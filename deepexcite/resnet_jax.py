import json
import logging
import math
import os
import random
import sys
from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import optimizers, stax
from jax.experimental.stax import Elu, FanInSum, FanOut, GeneralConv, Identity
from torch.utils.data import DataLoader
from tqdm import tqdm

import fenton_karma as fk
from dataset import ConcatSequence
from jaxboard import SummaryWriter
from utils import seed_experiment


class Module(NamedTuple):
    init: Callable[[jnp.ndarray, Tuple], Tuple[Tuple, jnp.ndarray]]
    apply: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def ResidualBlock(out_channels, kernel_size, stride, padding, input_format):
    double_conv = stax.serial(
        GeneralConv(input_format, out_channels, kernel_size, stride, padding),
        Elu,
        GeneralConv(input_format, out_channels, kernel_size, stride, padding),
    )
    return Module(
        *stax.serial(FanOut(2), stax.parallel(double_conv, Identity), FanInSum, Elu)
    )


def SumLastInputFrame():
    """Layer construction function for a fan-in sum layer."""
    init_fun = lambda rng, input_shape: (input_shape[0], ())
    apply_fun = lambda params, inputs, **kwargs: inputs[0] + inputs[1][:, -2:-1]
    return init_fun, apply_fun


def ResNet(
    hidden_channels, out_channels, kernel_size, strides, padding, depth, input_format
):
    return Module(
        *(
            stax.serial(
                GeneralConv(
                    (input_format), hidden_channels, kernel_size, strides, padding
                ),
                *[
                    ResidualBlock(
                        hidden_channels, kernel_size, strides, padding, input_format
                    )
                    for _ in range(depth)
                ],
                GeneralConv(input_format, out_channels, kernel_size, strides, padding)
            )
        )
    )


@jax.jit
def compute_loss(y_hat: jnp.ndarray, y: jnp.ndarray):
    grad_y_hat_1 = fk.model.gradient(y_hat, -1)
    grad_y_hat_2 = fk.model.gradient(y_hat, -2)
    grad_y_1 = fk.model.gradient(y, -1)
    grad_y_2 = fk.model.gradient(y, -2)
    grad_loss_1 = jnp.mean((grad_y_hat_1 - grad_y_1) ** 2)  # mse
    grad_loss_2 = jnp.mean((grad_y_hat_2 - grad_y_2) ** 2)  # mse
    grad_loss = grad_loss_1 + grad_loss_2
    recon_loss = jnp.mean((y_hat - y) ** 2)  # mse
    return (grad_loss + recon_loss)


@partial(jax.jit, static_argnums=0)
def forward(model, params, x, y):
    y_hat = model.apply(params, x)
    return (compute_loss(y_hat, y), y_hat)


@partial(jax.jit, static_argnums=0)
def backward(model, params, x, y):
    return jax.value_and_grad(forward, argnums=1, has_aux=True)(model, params, x, y)


@partial(jax.jit, static_argnums=(0, 1))
def update(model, optimiser, iteration, optimiser_state, x, y):
    params = optimiser.params_fn(optimiser_state)
    (loss, y_hat), gradients = backward(model, params, x, y)
    return loss, y_hat, optimiser.update_fn(iteration, gradients, optimiser_state)


@jax.jit
def preprocess_batch(batch):
    return batch


@partial(jax.jit, static_argnums=(0, 1, 2))
def training_step(model, optimiser, refeed, iteration, optimiser_state, x, y):
    # init state
    loss = 0.0
    u_t1 = x
    y_hat_stacked = []
    for t in range(refeed):
        u_t2 = y[:, t][:, None, :, :, :]
        j, y_hat, optimiser_state = update(
            model, optimiser, iteration, optimiser_state, u_t1, u_t2
        )
        # update state
        loss += j
        u_t1 = jnp.concatenate([u_t1[:, 1:], y_hat], axis=1)
        y_hat_stacked += [y_hat]
    return loss, jnp.concatenate(y_hat_stacked, axis=1), optimiser_state


@partial(jax.jit, static_argnums=(0, 1, 2))
def training_step_fast(model, optimiser, refeed, iteration, optimiser_state, x, y):
    def fun(i, inputs):
        loss, x, optimiser_state = inputs
        u_t2 = y[:, i][:, None, :, :, :]
        loss, y_hat, optimiser_state = update(model, optimiser, iteration, optimiser_state, x, u_t2)
        u_t1 = jnp.concatenate([x[:, 1:], y_hat], axis=1)
        return (loss, u_t1, optimiser_state)
    return jax.lax.fori_loop(0, refeed, fun, (0., x, optimiser_state))


@partial(jax.jit, static_argnums=(0, 2))
def validation_step(model, params, refeed, x, y):
    # init state
    loss = 0.0
    u_t1 = x
    y_hat_stacked = []
    for t in range(refeed):
        u_t2 = y[:, t][:, None, :, :, :]
        j, y_hat = forward(model, params, u_t1, u_t2)
        # update state
        loss += j
        u_t1 = jnp.concatenate([u_t1[:, 1:], y_hat], axis=1)
        y_hat_stacked += [y_hat]
    return loss, jnp.concatenate(y_hat_stacked, axis=1)


def logging_step(logger, loss, x, y_hat, y, step, frequency, prefix):
    if step % frequency:
        return
    figsize = (15, 2.5 * y_hat.shape[1])
    # log loss
    logger.scalar("{}/loss".format(prefix), loss, step=step)
    # log input states
    x_states = [fk.model.State(*s.squeeze()) for s in x[0]]
    fig, _ = fk.plot.plot_states(x_states, vmin=None, vmax=None, figsize=figsize)
    logger.figure("{}/x".format(prefix), fig, step)
    # log predictions as images
    y_hat_states = [fk.model.State(*s.squeeze()) for s in y_hat[0]]
    fig, _ = fk.plot.plot_states(y_hat_states, vmin=None, vmax=None, figsize=figsize)
    logger.figure("{}/y_hat".format(prefix), fig, step)
    # log truth
    y_states = [fk.model.State(*s.squeeze()) for s in y[0]]
    fig, _ = fk.plot.plot_states(y_states, vmin=None, vmax=None, figsize=figsize)
    logger.figure("{}/y".format(prefix), fig, step)
    # log error
    error_states = [fk.model.State(*s.squeeze()) for s in y_hat[0] - y[0, :y_hat.shape[1]]]
    fig, _ = fk.plot.plot_states(error_states, vmin=None, vmax=None, figsize=figsize)
    logger.figure("{}/y_hat .format(prefix)- y", fig, step)
    # y_hat_states = [fk.model.State(*s.squeeze()) for s in y_hat[0]]
    # y_states = [fk.model.State(*s.squeeze()) for s in y[0]]
    # fig, _ = fk.plot.compare_states(y_hat_states, y_states, vmin=None, vmax=None, figsize=figsize)
    # logger.figure("{}/comparison".format(prefix), fig, step)
    # force update
    logger.flush()
    return


def create_dataloader(
    root, batch_size, frames_in, frames_out, step, shape, search_regex, preload
):
    dataset = ConcatSequence(
        root,
        frames_in,
        frames_out,
        step,
        transform=partial(jax.image.resize, shape=shape, method="bilinear"),
        keys=search_regex,
        preload=preload,
    )
    logging.info(
        "Found training dataset at: {}".format([x.filename for x in dataset.datasets])
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=jnp.stack,
        shuffle=True,
        drop_last=True,
    )
    return loader


def main(hparams):
    # set logger
    logdir = os.path.join(hparams.logdir, hparams.experiment)
    logger = SummaryWriter(logdir)

    # save hparams
    logging.info(hparams)
    json.dump(vars(hparams), open(os.path.join(logger.log_dir, "hparams.json"), "w"))
    n_sequence_out = hparams.refeed * hparams.frames_out

    # get data
    train_dataloader = create_dataloader(
        root=hparams.root,
        batch_size=hparams.batch_size,
        frames_in=hparams.frames_in,
        frames_out=n_sequence_out,
        step=hparams.step,
        shape=(
            hparams.frames_in + n_sequence_out,
            hparams.n_channels,
            *hparams.size,
        ),
        search_regex=hparams.train_search_regex,
        preload=hparams.preload,
    )
    val_dataloader = create_dataloader(
        root=hparams.root,
        batch_size=hparams.batch_size,
        frames_in=hparams.frames_in,
        frames_out=n_sequence_out,
        step=hparams.step,
        shape=(
            hparams.frames_in + n_sequence_out,
            hparams.n_channels,
            *hparams.size,
        ),
        search_regex=hparams.val_search_regex,
        preload=hparams.preload,
    )
    #
    n_train_samples = len(train_dataloader.dataset)
    n_train_batches = math.floor(n_train_samples / hparams.batch_size)
    n_val_samples = len(val_dataloader.dataset)
    n_val_batches = math.floor(n_val_samples / hparams.batch_size)

    # init model parameters
    in_shape = (
        hparams.batch_size,
        hparams.frames_in,
        hparams.n_channels,
        *hparams.size,
    )
    rng = jax.random.PRNGKey(0)
    resnet = ResNet(
        hidden_channels=hparams.n_filters,
        out_channels=1,
        kernel_size=hparams.kernel_size,
        strides=tuple([1] * len(hparams.kernel_size)),
        padding="SAME",
        depth=hparams.depth,
        input_format=(hparams.input_format, "IDWHO", hparams.input_format),
    )
    out_shape, params = resnet.init(rng, in_shape)

    # init optimiser
    optimiser = optimizers.adam(hparams.lr)
    optimiser_state = optimiser.init_fn(params)

    train_iteration = 0
    val_iteration = 0
    for i in range(hparams.epochs):
        ## TRAINING
        train_loss = 0.0
        for j, batch in enumerate(train_dataloader):
            # prepare data
            x, y = batch[:, : hparams.frames_in], batch[:, hparams.frames_in :]
            # learning
            train_loss, y_hat_stacked, optimiser_state = training_step_fast(
                resnet, optimiser, hparams.refeed, train_iteration, optimiser_state, x, y
            )
            # logging
            logging_step(
                logger, train_loss, x, y_hat_stacked, y, train_iteration, hparams.log_frequency, "train"
            )
            # prepare next iteration
            print("Epoch {}/{} - Training step {}/{} - Loss: {:.6f}\t\t".format(i, hparams.epochs, j, n_train_batches, train_loss), end="\r")
            train_iteration = train_iteration + 1
        # checkpoint model
        logger.checkpoint("resnet", optimiser.params_fn(optimiser_state), train_iteration, train_loss)

        ## VALIDATING
        val_loss = 0.0
        params = optimiser.params_fn(optimiser_state)
        for j, batch in enumerate(val_dataloader):
            # prepare data
            x, y = batch[:, : hparams.frames_in], batch[:, hparams.frames_in :]
            # learning
            val_loss, y_hat_stacked = validation_step(resnet, params, hparams.refeed, x, y)
            # logging
            logging_step(
                logger, val_loss, x, y_hat_stacked, y, val_iteration, hparams.log_frequency, "val"
            )
            # prepare next iteration
            print("Epoch {}/{} - Evaluation step {}/{} - Loss: {:.6f}\t\t".format(i, hparams.epochs, j, n_val_batches, val_loss), end="\r")
            val_iteration = val_iteration + 1

        ## SCHEDULED UPDATES
        if (i != 0) and (train_loss <= hparams.increase_at) and (hparams.refeed < 20):
            hparams.refeed = hparams.refeed + 1
            train_dataloader.dataset.frames_out = hparams.refeed
            val_dataloader.dataset.frames_out = hparams.refeed
            assert hparams.refeed == train_dataloader.dataset.frames_out == val_dataloader.dataset.frames_out
            logging.info(
                "Increasing the amount of output frames to {}".format(hparams.refeed)
            )

    return optimiser_state


if __name__ == "__main__":
    from argparse import ArgumentParser
    seed_experiment(0)
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
    # optim
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--increase_at", type=float, default=0.)
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
    # parse arguments
    hparams = parser.parse_args()
    # set logging level
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if hparams.debug else logging.INFO
    )
    main(hparams)
