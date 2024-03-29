import logging
import os
import pickle
from functools import partial
from typing import NamedTuple, Tuple

import cardiax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from helx.distributed import redistribute_tree
from helx.nn.module import Module
from helx.optimise.optimisers import Optimiser
from helx.typing import Params
from jax.experimental import optimizers
from jax.experimental.optimizers import OptimizerState

from deepx.resnet import HParams


def compute_loss(y_hat, y):
    # zero derivative
    recon_loss = jnp.sqrt(jnp.mean((y_hat - y) ** 2))  # rmse

    # first derivative
    grad_y_hat_x = cardiax.solve.gradient(y_hat, -1)
    grad_y_hat_y = cardiax.solve.gradient(y_hat, -2)
    grad_y_x = cardiax.solve.gradient(y, -1)
    grad_y_y = cardiax.solve.gradient(y, -2)
    grad_loss_x = jnp.sqrt(jnp.mean((grad_y_hat_x - grad_y_x) ** 2))  # rmse
    grad_loss_y = jnp.sqrt(jnp.mean((grad_y_hat_y - grad_y_y) ** 2))  # rmse
    grad_loss = grad_loss_x + grad_loss_y
    return recon_loss + grad_loss


def preprocess(batch):
    xs, ys = batch
    xs = jnp.concatenate(
        [xs[:, :, :, :3], xs[:, :, :, -1:] * 500], axis=-3
    )  # rescale diffusivity to median of the other fields
    return (xs, ys)


def forward(
    model: Module, params: Params, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[float, jnp.ndarray]:
    y_hat = model.apply(params, x)
    loss = compute_loss(y_hat, y)
    return (loss, y_hat)


def postprocess_gradients(gradients):
    return optimizers.clip_grads(gradients, 1.0)


def sgd_step(
    model: Module,
    optimiser: Optimiser,
    iteration: int,
    optimiser_state: OptimizerState,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[float, jnp.ndarray, OptimizerState]:
    params = optimiser.params(optimiser_state)
    backward = jax.value_and_grad(forward, argnums=1, has_aux=True, allow_int=True)
    (loss, y_hat), gradients = backward(model, params, x, y)
    gradients = postprocess_gradients(gradients)
    return loss, y_hat, optimiser.update(iteration, gradients, optimiser_state)


def refeed(x0, x1):
    ds = x0[:, :, -1:]  # diffusivity channel
    x1 = jnp.concatenate([x0[:, 1:, :-1], x1], axis=1)
    x1 = jnp.concatenate([x1, ds], axis=2)
    return x1


@partial(
    jax.pmap,
    in_axes=(None, None, None, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 1, 2),
    axis_name="device",
)
def tbtt_step(
    model: Module,
    optimiser: Optimiser,
    n_refeed: int,
    iteration: int,
    optimiser_state: OptimizerState,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
) -> Tuple[float, jnp.ndarray, OptimizerState]:
    def body_fun(inputs, i):
        x, optimiser_state = inputs
        y = ys[:, i][:, None, :, :, :]
        loss, y_hat, optimiser_state = sgd_step(
            model, optimiser, iteration, optimiser_state, x, y
        )
        x = refeed(x, y_hat)  # roll and replace inputs with new prediction
        return (x, optimiser_state), (loss, y_hat)

    (_, optimiser_state), (losses, ys_hat) = jax.lax.scan(
        body_fun, (xs, optimiser_state), xs=jnp.arange(n_refeed)
    )
    ys_hat = jnp.swapaxes(jnp.squeeze(ys_hat), 0, 1)
    losses = jax.lax.pmean(losses, axis_name="device")
    return (sum(losses), ys_hat, optimiser_state)


@partial(
    jax.pmap,
    in_axes=(None, None, None, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 1, 2),
    axis_name="device",
)
def btt_step(
    model: Module,
    optimiser: Optimiser,
    n_refeed: int,
    iteration: int,
    optimiser_state: OptimizerState,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
) -> Tuple[float, jnp.ndarray, OptimizerState]:
    @jax.checkpoint
    def body_fun(inputs, i):
        x, _params = inputs
        y = ys[:, i][:, None, :, :, :]
        loss, y_hat = forward(model, _params, x, y)
        x = refeed(x, y_hat)  # roll and replace inputs with new prediction
        return (x, _params), (loss, y_hat)

    def f(xs, params):
        _, (losses, ys_hat) = jax.lax.scan(
            body_fun, (xs, params), xs=jnp.arange(n_refeed)
        )
        ys_hat = jnp.swapaxes(jnp.squeeze(ys_hat), 0, 1)
        return sum(losses), ys_hat

    params = optimiser.params(optimiser_state)
    btt = jax.value_and_grad(f, has_aux=True, argnums=1)
    (loss, ys_hat), grads = btt(xs, params)
    grads = postprocess_gradients(grads)
    grads = jax.lax.pmean(grads, axis_name="device")
    loss = jax.lax.pmean(loss, axis_name="device")
    optimiser_state = optimiser.update(iteration, grads, optimiser_state)
    return (loss, ys_hat, optimiser_state)


@partial(
    jax.pmap,
    in_axes=(None, None, 0, 0),
    static_broadcasted_argnums=(0, 1),
    axis_name="device",
)
def infer(model, n_refeed, params, xs):
    def body_fun(inputs, i):
        x = inputs
        y_hat = model.apply(params, x)
        x = refeed(x, y_hat)  #  add the new pred to the inputs
        return x, y_hat

    _, ys_hat = jax.lax.scan(body_fun, xs, xs=jnp.arange(n_refeed))
    ys_hat = jnp.swapaxes(jnp.squeeze(ys_hat), 0, 1)
    return ys_hat


@partial(
    jax.pmap,
    in_axes=(None, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 1),
    axis_name="device",
)
def evaluate(
    model: Module,
    n_refeed: int,
    params: Params,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:
    def body_fun(inputs, i):
        x = inputs
        y = ys[:, i][:, None, :, :, :]  #  ith target frame
        loss, y_hat = forward(model, params, x, y)
        x = refeed(x, y_hat)  #  add the new pred to the inputs
        return x, (loss, y_hat)

    _, (losses, ys_hat) = jax.lax.scan(body_fun, xs, xs=jnp.arange(n_refeed))
    ys_hat = jnp.swapaxes(jnp.squeeze(ys_hat), 0, 1)
    losses = jax.lax.pmean(losses, axis_name="device")
    return (sum(losses), ys_hat)


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
    train_state=None,
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
    diffusivity = xs[0, 0, -1, -1]  #  (device, batch, t, c, w, h)

    def log_states(array, name, **kw):
        # take only first element in batch and last frame
        a = array[0, 0, -1]  #  (device, batch, t, c, w, h)
        state = cardiax.solve.State(a[0], a[1], a[2])
        fig, _ = cardiax.plot.plot_state(state, diffusivity=diffusivity, **kw)
        wandb.log(
            {"{}/{}".format(prefix, name): fig, "epoch": current_epoch, "batch": step},
            step=global_step,
        )
        plt.close(fig)

    # log input states
    log_states(xs[:, :, :, :3], "inputs")
    log_states(ys_hat, "predictions")
    log_states(ys, "truth")
    log_states(jnp.abs(ys_hat - ys), "L1")
    # log_states(compute_loss(ys_hat, ys), "Our loss")

    if train_state is None or (step % log_frequency * 4):
        return

    params_path = os.path.join(
        wandb.run.dir, "train_state_{}.pickle".format(global_step)
    )
    train_state.save(params_path)
    wandb.save(params_path, base_path=wandb.run.dir)
    return


log_train = partial(log, prefix="train")
log_val = partial(log, prefix="val")
log_test = partial(log, prefix="test")


class TrainState(NamedTuple):
    rng: jnp.ndarray
    iteration: int
    opt_state: OptimizerState
    hparams: HParams

    def serialise(self):
        state = self._replace(
            opt_state=optimizers.unpack_optimizer_state(self.opt_state)
        )
        return pickle.dumps(state)

    @staticmethod
    def deserialise(obj):
        state = pickle.loads(obj)
        opt_state = optimizers.pack_optimizer_state(state.opt_state)
        state = state._replace(opt_state=redistribute_tree(opt_state))
        return state

    def save(self, filepath):
        with open(filepath, "wb") as f:
            state = self.serialise()
            f.write(state)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return TrainState.deserialise(f.read())

    @staticmethod
    def restore(wandb_address):
        if os.path.exists(wandb_address):
            return TrainState.load(wandb_address)
        run_id, step = wandb_address.strip().split("/")
        api = wandb.Api()
        run = api.run("epignatelli/deepx/{}".format(run_id))
        filename = "train_state_{}.pickle".format(step)
        run.file(filename).download(replace=True)
        return TrainState.load(filename)
