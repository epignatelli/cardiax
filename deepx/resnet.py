import logging
from functools import partial
from typing import NamedTuple, Tuple

import cardiax
import jax
import jax.numpy as jnp
import wandb
from helx.methods import module, batch
from helx.types import Module, Optimiser, OptimizerState, Params, Shape
from jax.experimental import stax


def ResidualBlock(out_channels, kernel_size, stride, padding, input_format):
    double_conv = stax.serial(
        stax.GeneralConv(input_format, out_channels, kernel_size, stride, padding),
        stax.Elu,
        stax.GeneralConv(input_format, out_channels, kernel_size, stride, padding),
    )
    return Module(
        *stax.serial(
            stax.FanOut(2), stax.parallel(double_conv, stax.Identity), stax.FanInSum
        )
    )


def AddLastItem(axis):
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: inputs[0] + jax.lax.slice_in_dim(
        inputs[1], inputs[1].shape[axis] - 1, inputs[1].shape[axis], axis=axis
    )
    return init_fun, apply_fun


@module
def ResNet(hidden_channels, out_channels, depth):
    residual = stax.serial(
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), hidden_channels, (4, 3, 3), (1, 1, 1), "SAME"
        ),
        *[
            ResidualBlock(
                hidden_channels,
                (4, 3, 3),
                (1, 1, 1),
                "SAME",
                ("NCDWH", "IDWHO", "NCDWH"),
            )
            for _ in range(depth)
        ],
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), out_channels, (4, 3, 3), (1, 1, 1), "SAME"
        )
    )
    return Module(
        *stax.serial(
            stax.FanOut(2), stax.parallel(residual, stax.Identity), AddLastItem(1)
        )
    )


class HParams(NamedTuple):
    seed: int
    log_frequency: int
    debug: bool
    hidden_channels: int
    in_channels: int
    depth: int
    lr: float
    batch_size: int
    evaluation_steps: int
    epochs: int
    increase_at: float
    teacher_forcing_prob: float
    from_checkpoint: str
    root: str
    paramset: str
    size: int
    frames_in: int
    frames_out: int
    step: int
    refeed: int
    preload: bool


def compute_loss(y_hat, y):
    # zero derivative
    recon_loss = jnp.mean((y_hat - y) ** 2)  # mse

    # first derivative
    grad_y_hat_x = cardiax.solve.gradient(y_hat, -1)
    grad_y_hat_y = cardiax.solve.gradient(y_hat, -2)
    grad_y_x = cardiax.solve.gradient(y, -1)
    grad_y_y = cardiax.solve.gradient(y, -2)
    grad_loss_x = jnp.mean((grad_y_hat_x - grad_y_x) ** 2)  # mse
    grad_loss_y = jnp.mean((grad_y_hat_y - grad_y_y) ** 2)  # mse

    # second derivative
    del_y_hat_x = cardiax.solve.gradient(grad_y_hat_x, -1)
    del_y_hat_y = cardiax.solve.gradient(grad_y_hat_y, -1)
    del_y_x = cardiax.solve.gradient(grad_y_x, -1)
    del_y_y = cardiax.solve.gradient(grad_y_y, -1)
    del_loss_x = jnp.mean((del_y_hat_x - del_y_x) ** 2)  # mse
    del_loss_y = jnp.mean((del_y_hat_y - del_y_y) ** 2)  # mse

    return recon_loss + grad_loss_x + grad_loss_y + del_loss_x + del_loss_y


def preprocess(batch):
    raise NotImplementedError


@partial(jax.jit, static_argnums=(0,))
def forward(
    model: Module, params: Params, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[float, jnp.ndarray]:
    y_hat = model.apply(params, x)
    return (compute_loss(y_hat, y), y_hat)


@partial(jax.jit, static_argnums=(0,))
def backward(
    model: Module, params: Params, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[Tuple[float, jnp.ndarray], jnp.ndarray]:
    return jax.value_and_grad(forward, argnums=1, has_aux=True, allow_int=True)(
        model, params, x, y
    )


@partial(jax.jit, static_argnums=(0, 1))
def sgd_step(
    model: Module,
    optimiser: Optimiser,
    iteration: int,
    optimiser_state: OptimizerState,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[float, jnp.ndarray, OptimizerState]:
    params = optimiser.params(optimiser_state)
    (loss, y_hat), gradients = backward(model, params, x, y)
    return loss, y_hat, optimiser.update(iteration, gradients, optimiser_state)


@partial(jax.jit, static_argnums=(0, 1, 2))
def tbtt_step(
    model: Module,
    optimiser: Optimiser,
    refeed: int,
    iteration: int,
    optimiser_state: OptimizerState,
    x,
    y,
) -> Tuple[float, jnp.ndarray, OptimizerState]:
    def attach_diffusivity(xs, d):
        pass

    def detach_diffusivity(xs):
        pass

    def body_fun(inputs, i):
        _loss0, s0, optimiser_state = inputs
        s1 = y[:, i][:, None, :, :, :]
        _loss1, y_hat, optimiser_state = sgd_step(
            model, optimiser, iteration, optimiser_state, s0, s1
        )
        _loss = _loss0 + _loss1
        s0 = jnp.concatenate([s0[:, 1:], y_hat], axis=1)
        return (_loss, s0, optimiser_state), s0

    (loss, _, optimiser_state), ys = jax.lax.scan(
        body_fun, (0.0, x, optimiser_state), xs=jnp.arange(refeed), length=refeed
    )
    return (loss, ys, optimiser_state)


@partial(jax.jit, static_argnums=(0, 2))
def evaluate(
    model: Module,
    refeed: int,
    params,
    x,
    y,
) -> Tuple[float, jnp.ndarray]:
    def body_fun(i, inputs):
        _loss0, s0 = inputs
        s1 = y[:, i][:, None, :, :, :]
        _loss1, y_hat = forward(model, params, s0, s1)
        _loss = _loss0 + _loss1
        s0 = jnp.concatenate([s0[:, 1:], y_hat], axis=1)
        carry, y = (_loss, s0), s0
        return carry, y

    (loss, _), ys = jax.lax.scan(body_fun, (0.0, x), xs=None, length=refeed)
    return (loss, ys)


def log(loss, xs, ys_hat, ys, step, frequency, prefix=""):
    if step % frequency:
        return
    logging.debug(xs.shape)
    logging.debug(ys_hat.shape)
    logging.debug(ys.shape)

    # log loss
    wandb.log("{}/loss".format(prefix), loss, step=step)

    def log_states(array, name, **kw):
        states = [cardiax.solve.State(*a.squeeze()) for a in array]
        fig, _ = cardiax.plot.plot_states(
            states, figsize=(15, 2.5 * array.shape[1]), **kw
        )
        wandb.log("{}/{}".format(prefix, name), fig, step)

    # log input states
    log_states(xs, "inputs")
    log_states(ys_hat, "predictions")
    log_states(ys, "truth")
    log_states(jnp.abs(ys_hat - ys), "L1")
    # log_states(compute_loss(ys_hat, ys), "Our loss")
    return


log_train = partial(log, prexit="train")
log_val = partial(log, prexit="val")
