import logging
from functools import partial
from typing import NamedTuple, Tuple

import cardiax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from helx.methods import batch, module
from helx.types import Module, Optimiser, OptimizerState, Params, Shape
from jax.experimental import optimizers, stax


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

    return (
        recon_loss + 0.1 * (grad_loss_x + grad_loss_y) + 0.1 * (del_loss_x + del_loss_y)
    )


def preprocess(batch):
    raise NotImplementedError


@partial(jax.jit, static_argnums=(0,))
def forward(
    model: Module, params: Params, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[float, jnp.ndarray]:
    y_hat = model.apply(params, x)
    loss = compute_loss(y_hat, y)
    return (loss, y_hat)


@partial(jax.jit, static_argnums=(0,))
def backward(
    model: Module, params: Params, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[Tuple[float, jnp.ndarray], jnp.ndarray]:
    return jax.value_and_grad(forward, argnums=1, has_aux=True, allow_int=True)(
        model, params, x, y
    )


@jax.jit
def postprocess_gradients(gradients):
    return optimizers.clip_grads(gradients, 1.0)


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
    gradients = postprocess_gradients(gradients)
    return loss, y_hat, optimiser.update(iteration, gradients, optimiser_state)


@jax.jit
def roll_and_replace(
    x0,
    x1,
):
    return jnp.concatenate([x0[:, 1:, :-1], x1], axis=1)


@partial(jax.jit, static_argnums=(0, 1, 2))
def tbtt_step(
    model: Module,
    optimiser: Optimiser,
    refeed: int,
    iteration: int,
    optimiser_state: OptimizerState,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[float, jnp.ndarray, OptimizerState]:
    def body_fun(inputs, i):
        _loss0, xs, optimiser_state = inputs
        ds = xs[:, :, -1:]
        ys = y[:, i][:, None, :, :, :]
        _loss1, ys_hat, optimiser_state = sgd_step(
            model, optimiser, iteration, optimiser_state, xs, ys
        )
        _loss = _loss0 + _loss1
        xs = roll_and_replace(xs, ys_hat)  # roll and replace inputs with new prediction
        xs = jnp.concatenate([xs, ds], axis=2)
        return (_loss, xs, optimiser_state), ys_hat

    (loss, _, optimiser_state), ys_hat = jax.lax.scan(
        body_fun, (0.0, x, optimiser_state), xs=jnp.arange(refeed)
    )
    ys_hat = jnp.swapaxes(jnp.squeeze(ys_hat), 0, 1)
    return (loss, ys_hat, optimiser_state)


ptbtt_step = jax.pmap(tbtt_step, static_broadcasted_argnums=(0, 1, 2))


@partial(jax.jit, static_argnums=(0, 1))
def evaluate(
    model: Module,
    refeed: int,
    params: Params,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:
    def body_fun(inputs, i):
        _loss0, xs = inputs
        ds = xs[:, :, -1:]
        ys = y[:, i][:, None, :, :, :]
        _loss1, ys_hat = forward(model, params, xs, ys)
        _loss = _loss0 + _loss1
        xs = roll_and_replace(xs, ys_hat)
        xs = jnp.concatenate([xs, ds], axis=2)
        return (_loss, xs), ys_hat

    (loss, _), ys = jax.lax.scan(body_fun, (0.0, x), xs=jnp.arange(refeed))
    ys_hat = jnp.swapaxes(jnp.squeeze(ys), 0, 1)
    return (loss, ys_hat)


pevaluate = jax.pmap(evaluate, static_broadcasted_argnums=(0, 2))


def log(
    current_epoch,
    max_epochs,
    step,
    maxsteps,
    loss,
    xs,
    ys_hat,
    ys,
    log_frequency,
    global_step,
    prefix="",
):
    loss = float(loss)
    step = int(step)
    logging.info(
        "Epoch {}/{} - {} step {}/{} - Loss: {:.6f}\t\t\t".format(
            current_epoch, max_epochs, prefix, step, maxsteps, loss
        ),
    )

    if step % log_frequency:
        return

    # log loss
    wandb.log(
        {"{}/loss".format(prefix): loss, "epoch": current_epoch, "batch": step},
        step=global_step,
    )
    diffusivity = xs[:, :, -1:].squeeze()[0, -1]

    def log_states(array, name, **kw):
        # take only first element in batch and last frame
        a = array[0, -1]
        state = cardiax.solve.State(a[0], a[1], a[2])
        fig, _ = cardiax.plot.plot_state(state, diffusivity=diffusivity, **kw)
        wandb.log(
            {"{}/{}".format(prefix, name): fig, "epoch": current_epoch, "batch": step},
            step=global_step,
        )
        plt.close(fig)

    # log input states
    log_states(xs[:, :, :3], "inputs")
    log_states(ys_hat, "predictions")
    log_states(ys, "truth")
    log_states(jnp.abs(ys_hat - ys), "L1")
    # log_states(compute_loss(ys_hat, ys), "Our loss")
    return


log_train = partial(log, prefix="train")
log_val = partial(log, prefix="val")
