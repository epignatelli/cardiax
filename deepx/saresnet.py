import json
import logging
import math
import os
import pickle
import sys
from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import cardiax
from helx.types import Optimiser
from jax.experimental import optimizers, stax
from jax.experimental.stax import Elu, FanInSum, FanOut, GeneralConv, Identity

from .dataset import ConcatSequence, DataStream, imresize
from .jaxboard import SummaryWriter
from .utils import seed_experiment

from helx.methods import module
from helx.types import Params, Module


@module
def ReselBlock(n_heads, input_format):
    one = (1,) * len((1, 1))

    def init(rng, input_shape):
        return tuple(
            *(jax.vmap(GeneralConv(input_format, n_heads, one, one, "SAME")[0]))(
                jnp.stack(jax.random.split(rng, 4))
            )
        )

    def apply(params, inputs):
        dw_conv = lambda x, w, b: b + jax.lax.conv_general_dilated(
            x, w, one, one, one, one, dimension_numbers=input_format
        )
        fx, gx, hx = jax.vmap(dw_conv)(params[:3], inputs)
        sx = jnp.matmul(jnp.transpose(fx, (-2, -1)), gx)
        sx = jax.nn.softmax(sx, axis=-2)
        sx = jnp.matmul(sx, hx)
        vx = dw_conv(sx, *params[-1])
        return inputs + vx

    return (init, apply)


def ResNet(
    hidden_channels, out_channels, kernel_size, strides, padding, depth, input_format
):
    residual = stax.serial(
        GeneralConv(input_format, hidden_channels, kernel_size, strides, padding),
        *[ReselBlock(hidden_channels, input_format) for _ in range(depth)],
        GeneralConv(input_format, out_channels, kernel_size, strides, padding)
    )
    return Module(*stax.serial(FanOut(2), stax.parallel(residual, Identity), FanInSum))


@jax.jit
def loss(y_hat: jnp.ndarray, y: jnp.ndarray):
    grad_y_hat_1 = cardiax.solve.gradient(y_hat, -1)
    grad_y_hat_2 = cardiax.solve.gradient(y_hat, -2)
    grad_y_1 = cardiax.solve.gradient(y, -1)
    grad_y_2 = cardiax.solve.gradient(y, -2)
    grad_loss_1 = jnp.mean((grad_y_hat_1 - grad_y_1) ** 2)  # mse
    grad_loss_2 = jnp.mean((grad_y_hat_2 - grad_y_2) ** 2)  # mse
    grad_loss = grad_loss_1 + grad_loss_2
    recon_loss = jnp.mean((y_hat - y) ** 2)  # mse
    return grad_loss + recon_loss


@partial(jax.jit, static_argnums=0)
def forward(model, params, x, y):
    y_hat = model.apply(params, x)
    return (loss(y_hat, y), y_hat)


@partial(jax.jit, static_argnums=0)
def backward(model, params, x, y):
    return jax.value_and_grad(forward, argnums=1, has_aux=True)(model, params, x, y)


@partial(jax.jit, static_argnums=(0, 1))
def sgd_step(model, optimiser, iteration, optimiser_state, x, y):
    params = optimiser.params(optimiser_state)
    (loss, y_hat), gradients = backward(model, params, x, y)
    return loss, y_hat, optimiser.update(iteration, gradients, optimiser_state)


@partial(jax.jit, static_argnums=(0, 1, 2))
def update(model, optimiser, refeed, iteration, optimiser_state, x, y):
    # init state
    loss = 0.0
    u_t1 = x
    y_hat_stacked = []
    for t in range(refeed):
        u_t2 = y[:, t][:, None, :, :, :]
        j, y_hat, optimiser_state = sgd_step(
            model, optimiser, iteration, optimiser_state, u_t1, u_t2
        )
        # update state
        loss += j
        u_t1 = jnp.concatenate([u_t1[:, 1:], y_hat], axis=1)
        y_hat_stacked += [y_hat]
    return loss, jnp.concatenate(y_hat_stacked, axis=1), optimiser_state


@partial(jax.jit, static_argnums=(0, 1, 2))
def update_fast(model, optimiser, refeed, iteration, optimiser_state, x, y):
    def body_fun(i, inputs):
        loss, u_t1, optimiser_state = inputs
        u_t2 = y[:, i][:, None, :, :, :]
        loss, y_hat, optimiser_state = sgd_step(
            model, optimiser, iteration, optimiser_state, u_t1, u_t2
        )
        u_t1 = jnp.concatenate([u_t1[:, 1:], y_hat], axis=1)
        return (loss, u_t1, optimiser_state)

    return jax.lax.fori_loop(0, refeed, body_fun, (0.0, x, optimiser_state))


@partial(jax.jit, static_argnums=(0, 2))
def evaluate(model, params, refeed, x, y):
    # init state
    loss = 0.0
    u_t1 = x
    y_hat_stacked = []
    for t in range(refeed):
        # evaluate state
        u_t2 = y[:, t][:, None, :, :, :]
        j, y_hat = forward(model, params, u_t1, u_t2)
        # update state
        loss += j
        u_t1 = jnp.concatenate([u_t1[:, 1:], y_hat], axis=1)
        y_hat_stacked += [y_hat]
    return loss, jnp.concatenate(y_hat_stacked, axis=1)


def log(logger, loss, x, y_hat, y, step, frequency, prefix):
    if step % frequency:
        return
    logging.debug(x.shape)
    logging.debug(y_hat.shape)
    logging.debug(y.shape)
    # log loss
    logger.scalar("{}/loss".format(prefix), loss, step=step)
    # log input states
    x_states = [cardiax.solve.State(*s.squeeze()) for s in x[0]]
    fig, _ = cardiax.plot.plot_states(x_states, figsize=(15, 2.5 * x.shape[1]))
    logger.figure("{}/x".format(prefix), fig, step)
    # log predictions as images
    y_hat_states = [cardiax.solve.State(*s.squeeze()) for s in y_hat[0]]
    fig, _ = cardiax.plot.plot_states(y_hat_states, figsize=(15, 2.5 * y_hat.shape[1]))
    logger.figure("{}/y_hat".format(prefix), fig, step)
    # log truth
    y_states = [cardiax.solve.State(*s.squeeze()) for s in y[0]]
    fig, _ = cardiax.plot.plot_states(y_states, figsize=(15, 2.5 * y.shape[1]))
    logger.figure("{}/y".format(prefix), fig, step)
    # log error
    min_time = min(y.shape[1], y_hat.shape[1])
    error_states = [
        cardiax.solve.State(*s.squeeze()) for s in y_hat[0, :min_time] - y[0, :min_time]
    ]
    fig, _ = cardiax.plot.plot_states(
        error_states, vmin=-1, vmax=1, figsize=(15, 2.5 * y_hat.shape[1])
    )
    logger.figure("{}/y_hat - y".format(prefix), fig, step)
    # force update
    logger.flush()
    return


def main(hparams):
    # seed experiment
    seed_experiment(hparams.seed)
    # set logger
    logdir = os.path.join(hparams.logdir, hparams.experiment)
    logger = SummaryWriter(logdir)
    # save hparams
    logging.info(hparams)
    json.dump(vars(hparams), open(os.path.join(logger.log_dir, "hparams.json"), "w"))
    n_sequence_out = hparams.refeed * hparams.frames_out

    # get data streams
    in_shape = (
        hparams.batch_size,
        hparams.frames_in,
        hparams.n_channels,
        *hparams.size,
    )
    train_set = ConcatSequence(
        root=hparams.root,
        frames_in=hparams.frames_in,
        frames_out=n_sequence_out,
        step=hparams.step,
        transform=partial(imresize, size=hparams.size, method="bilinear"),
        keys=hparams.train_search_regex,
        preload=hparams.preload,
    )
    val_set = ConcatSequence(
        root=hparams.root,
        frames_in=hparams.frames_in,
        frames_out=hparams.evaluation_steps,
        step=hparams.step,
        transform=partial(imresize, size=hparams.size, method="bilinear"),
        keys=hparams.val_search_regex,
        preload=hparams.preload,
        perc=hparams.val_perc,
    )
    train_dataloader = DataStream(
        train_set, hparams.batch_size, jnp.stack, hparams.seed
    )
    val_dataloader = DataStream(val_set, hparams.batch_size, jnp.stack)

    # init model parameters
    rng = jax.random.PRNGKey(hparams.seed)
    resnet = ResNet(
        hidden_channels=hparams.n_filters,
        out_channels=1,
        kernel_size=hparams.kernel_size,
        strides=tuple([1] * len(hparams.kernel_size)),
        padding="SAME",
        depth=hparams.depth,
        input_format=(hparams.input_format, "IDWHO", hparams.input_format),
    )
    if hparams.from_checkpoint is not None and os.path.exists(hparams.from_checkpoint):
        with open(hparams.from_checkpoints, "rb") as f:
            params = pickle.load(f)
    else:
        _, params = resnet.init(rng, in_shape)

    # init optimiser
    optimiser = Optimiser(optimizers.adam(hparams.lr))
    optimiser_state = optimiser.init(params)

    train_iteration = 0
    val_iteration = 0
    for i in range(hparams.epochs):
        ## TRAINING
        train_loss = 0.0
        for j, batch in enumerate(train_dataloader):
            logging.debug(batch.shape)
            # prepare data
            x, y = batch[:, : hparams.frames_in], batch[:, hparams.frames_in :]
            # learning
            j_train, y_hat_stacked, optimiser_state = update_fast(
                resnet,
                optimiser,
                hparams.refeed,
                train_iteration,
                optimiser_state,
                x,
                y,
            )
            train_loss += j_train
            # logging
            log(
                logger,
                j_train,
                x,
                y_hat_stacked,
                y,
                train_iteration,
                hparams.log_frequency,
                "train",
            )
            # prepare next iteration
            print(
                "Epoch {}/{} - Training step {}/{} - Loss: {:.6f}\t\t\t".format(
                    i, hparams.epochs, j, train_dataloader.n_batches, j_train
                ),
                end="\r",
            )
            train_iteration = train_iteration + 1
            if j == 10 and hparams.debug:
                break
        # checkpoint model
        train_loss /= len(train_dataloader)
        logger.checkpoint("resnet", optimiser_state, train_iteration, train_loss)

        ## VALIDATING
        # we always validate on 20 times steps
        val_loss = 0.0
        params = optimiser.params(optimiser_state)
        for j, batch in enumerate(val_dataloader):
            # prepare data
            x, y = batch[:, : hparams.frames_in], batch[:, hparams.frames_in :]
            # learning
            j_val, y_hat_stacked = evaluate(
                resnet, params, hparams.evaluation_steps, x, y
            )
            val_loss += j_val
            # prepare next iteration
            print(
                "Epoch {}/{} - Evaluation step {}/{} - Loss: {:.6f}\t\t\t".format(
                    i, hparams.epochs, j, val_dataloader.n_batches, j_val
                ),
                end="\r",
            )
            val_iteration = val_iteration + 1
            if j == 10 and hparams.debug:
                break
        # logging
        val_loss /= len(val_dataloader)
        log(logger, val_loss, x, y_hat_stacked, y, val_iteration, val_iteration, "val")

        ## SCHEDULED UPDATES
        if (i != 0) and (train_loss <= hparams.increase_at) and (hparams.refeed < 20):
            hparams.refeed = hparams.refeed + 1
            train_dataloader.frames_out = hparams.refeed
            assert hparams.refeed == train_dataloader.frames_out
            logger.scalar("refeed", hparams.refeed, train_iteration)
            logging.info(
                "Increasing the amount of output frames to {} \t\t\t".format(
                    hparams.refeed
                )
            )
        # resetting flow
        train_dataloader.reset()
        val_dataloader.reset()

    return optimiser_state


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # model args
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_channels", type=int, default=5)
    parser.add_argument("--size", type=int, nargs="+", default=(256, 256))
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--n_filters", type=int, default=13)
    parser.add_argument(
        "--kernel_size",
        type=int,
        nargs="+",
        default=(5, 3, 3),
        help="CWH",
    )
    parser.add_argument("--frames_in", type=int, default=2)
    parser.add_argument("--frames_out", type=int, default=1)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--refeed", type=int, default=5)
    parser.add_argument("--input_format", type=str, default="NCDWH")
    parser.add_argument("--evaluation_steps", type=int, default=20)
    parser.add_argument("--val_perc", type=float, default=0.2)
    # optim
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--increase_at", type=float, default=0.0)
    parser.add_argument("--teacher_forcing_prob", type=float, default=0.0)
    # loader args
    parser.add_argument("--preload", action="store_true", default=False)
    parser.add_argument(
        "--root",
        type=str,
        default="/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/",
    )
    parser.add_argument(
        "--train_search_regex", type=str, default="^spiral.*_PARAMS5.hdf5"
    )
    parser.add_argument(
        "--val_search_regex", type=str, default="^spiral.*_PARAMS5.hdf5"
    )
    # program
    parser.add_argument("--logdir", type=str, default="logs/resnet/train")
    parser.add_argument("--experiment", type=str, default="../debug")
    parser.add_argument("--log_frequency", type=int, default=5)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # parse arguments
    hparams = parser.parse_args()
    # set logging level
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if hparams.debug else logging.INFO
    )
    main(hparams)
