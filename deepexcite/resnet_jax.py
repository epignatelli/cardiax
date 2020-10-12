import sys
import os
from typing import NamedTuple, Callable, Tuple
from functools import partial
import math
import logging
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
from jax import random
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental.stax import (
    GeneralConv,
    Identity,
    FanInSum,
    FanOut,
    Elu,
    BatchNorm,
)
import jax.numpy as jnp
import torch
import random
import numpy as onp
from torch.utils.data import DataLoader
from dataset import MnistDataset, ConcatSequence
from jaxboard import SummaryWriter
import fenton_karma as fk


class Module(NamedTuple):
    init: Callable[[jnp.ndarray, Tuple], Tuple[Tuple, jnp.ndarray]]
    apply: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def ResidualBlock(out_channels, kernel_size, stride, padding, input_format):
    double_conv = stax.serial(
        GeneralConv(input_format, out_channels, kernel_size, stride, padding),
        # BatchNorm((0,)),
        Elu,
        GeneralConv(input_format, out_channels, kernel_size, stride, padding),
        # BatchNorm((0,)),
    )
    return Module(
        *stax.serial(FanOut(2), stax.parallel(double_conv, Identity), FanInSum, Elu)
    )


def ResNet(
    hidden_channels, out_channels, kernel_size, strides, padding, depth, input_format
):
    return Module(
        *(
            stax.serial(
                GeneralConv(
                    input_format, hidden_channels, kernel_size, strides, padding
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
def compute_loss(y_hat, y):
    grad_y_hat_1 = fk.model.gradient(y_hat, -1)
    grad_y_hat_2 = fk.model.gradient(y_hat, -2)
    grad_y_1 = fk.model.gradient(y, -1)
    grad_y_2 = fk.model.gradient(y, -2)
    grad_loss_1 = jnp.mean((grad_y_hat_1 - grad_y_1) ** 2)  # mse
    grad_loss_2 = jnp.mean((grad_y_hat_2 - grad_y_2) ** 2)  # mse
    grad_loss = grad_loss_1 + grad_loss_2
    recon_loss = jnp.mean((y_hat - y) ** 2)  # mse
    return grad_loss + recon_loss


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


def logging_step(logger, loss, x, y_hat, y, step, frequency):
    if step % frequency:
        return
    # log loss
    logger.scalar("train/loss", loss, step=step)
    # log input states
    x_states = [fk.model.State(*s.squeeze()) for s in x[0]]
    fig, ax = fk.plot.plot_states(x_states, vmin=None, vmax=None, figsize=(15, 5))
    logger.figure("train/x", fig, step)
    # log predictions as images
    y_hat_states = [fk.model.State(*s.squeeze()) for s in y_hat[0]]
    fig, ax = fk.plot.plot_states(y_hat_states, vmin=None, vmax=None, figsize=(15, 5))
    logger.figure("train/y_hat", fig, step)
    # log truth
    y_states = [fk.model.State(*s.squeeze()) for s in y[0]]
    fig, ax = fk.plot.plot_states(y_states, vmin=None, vmax=None, figsize=(15, 5))
    logger.figure("train/y", fig, step)
    # force update
    logger.flush()
    return


def main(hparams):
    # set logger
    logdir = os.path.join(hparams.logdir, hparams.experiment)
    logging.info("Started logger at: {}".format(logdir))
    logger = SummaryWriter(logdir)

    # save hparams
    logging.info(hparams)
    json.dump(vars(hparams), open(os.path.join(logger.log_dir, "hparams.json"), "w"))

    # get data
    in_shape = (
        hparams.batch_size,
        hparams.frames_in,
        hparams.n_channels,
        *hparams.size,
    )
    train_dataloader = create_dataloader(
        root=hparams.root,
        batch_size=hparams.batch_size,
        frames_in=hparams.frames_in,
        frames_out=hparams.frames_out,
        step=hparams.step,
        shape=(
            hparams.frames_in + hparams.frames_out,
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
        frames_out=hparams.frames_out,
        step=hparams.step,
        shape=(
            hparams.frames_in + hparams.frames_out,
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
    n_val_batches = math.floor(n_train_batches / hparams.batch_size)

    # init model parameters
    rng = jax.random.PRNGKey(0)
    resnet = ResNet(
        hidden_channels=hparams.n_filters,
        out_channels=1,
        kernel_size=hparams.kernel_size,
        strides=tuple([1] * len(hparams.kernel_size)),
        padding="SAME",
        depth=hparams.depth,
        input_format=("NCDWH", "IDWHO", "NCDWH"),
    )
    out_shape, params = resnet.init(rng, in_shape)

    # init optimiser
    optimiser = optimizers.adam(hparams.lr)
    optimiser_state = optimiser.init_fn(params)

    loss = 0.0
    step = 0
    for i in tqdm(range(hparams.epochs), desc="Epochs"):
        # training
        pbar = tqdm(train_dataloader, desc="Training samples", total=n_train_batches)
        for batch in pbar:
            batch = preprocess_batch(batch)
            x, y = batch[:, : hparams.frames_in], batch[:, hparams.frames_in :]

            loss = 0.0
            u_t1 = x
            y_hat_stacked = []
            for t in range(hparams.frames_out):
                u_t2 = y[:, t : t + 1]
                j, y_hat, optimiser_state = update(
                    resnet, optimiser, t, optimiser_state, u_t1, u_t2
                )
                y_hat_stacked += [y_hat]
                u_t1 = jnp.concatenate([u_t1[:, 1:], y_hat], axis=1)
                loss += j

            y_hat_stacked = jnp.stack(y_hat_stacked, axis=1)
            logging_step(logger, loss, x, y_hat_stacked, y, step, hparams.log_frequency)
            pbar.set_postfix_str("Loss: {:.6f}".format(loss))
            step = step + 1
    return loss, optimiser_state


if __name__ == "__main__":
    # seed experiment
    SEED = 0
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    onp.random.seed(SEED)
    torch.manual_seed(SEED)

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
        default=(11, 7, 7),
        help="CWH",
    )
    parser.add_argument("--frames_in", type=int, default=2)
    parser.add_argument("--frames_out", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)
    # optim
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
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
        "--val_search_regex", type=str, default="^three_points.*_PARAMS5.hdf5"
    )
    # program
    parser.add_argument("--logdir", type=str, default="logs/resnet/train")
    parser.add_argument("--log_frequency", type=int, default=5)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--experiment", type=str, default="")

    hparams = parser.parse_args()

    # set logging level
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if hparams.debug else logging.INFO
    )

    main(hparams)
