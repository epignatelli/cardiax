import sys
from typing import NamedTuple, Callable, Tuple
from functools import partial
import math
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
from jax import random
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental.stax import GeneralConv, Identity, FanInSum, FanOut, Elu, Conv
import jax.numpy as jnp
from torch.utils.data import DataLoader
from dataset import MnistDataset, ConcatSequence


class Module(NamedTuple):
    init: Callable[[jnp.ndarray, Tuple], Tuple[Tuple, jnp.ndarray]]
    apply: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def ResidualBlock(out_channels, kernel_size, stride, padding):
    double_conv = stax.serial(
        Conv(out_channels, kernel_size, stride, padding),
        Elu,
        Conv(out_channels, kernel_size, stride, padding),
    )
    return Module(
        *stax.serial(FanOut(2), stax.parallel(double_conv, Identity), FanInSum, Elu)
    )


def ResNet(hidden_channels, out_channels, kernel_size, strides, padding, depth):
    return Module(
        *(
            stax.serial(
                Conv(hidden_channels, kernel_size, strides, padding),
                *[
                    ResidualBlock(hidden_channels, kernel_size, strides, padding)
                    for _ in range(depth)
                ],
                Conv(out_channels, kernel_size, strides, padding)
            )
        )
    )


@jax.jit
def compute_loss(y_hat, y):
    return jnp.mean((y_hat - y) ** 2)


@partial(jax.jit, static_argnums=0)
def forward(model, params, batch):
    x = batch
    y = jnp.flip(x, axis=-2)
    y_hat = model.apply(params, x)
    return compute_loss(y_hat, y)


@partial(jax.jit, static_argnums=0)
def backward(model, params, batch):
    return jax.value_and_grad(forward, argnums=1)(model, params, batch)


@partial(jax.jit, static_argnums=(0, 1))
def update(model, optimiser, iteration, optimiser_state, batch):
    params = optimiser.params_fn(optimiser_state)
    loss, gradients = backward(model, params, batch)
    return loss, optimiser.update_fn(iteration, gradients, optimiser_state)


@jax.jit
def preprocess_batch(batch):
    return batch


def log_step(loss, y_hat, y, stage):
    return


def create_dataloader(root, batch_size, frames_in, frames_out, step, shape, search_regex, preload):
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


if __name__ == "__main__":
    from argparse import ArgumentParser

    # set logging level
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    parser = ArgumentParser()

    # model args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_channels", type=int, default=5)
    parser.add_argument("--size", type=Tuple[int], nargs="+", default=(64, 64))
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--n_filters", type=int, default=4)
    parser.add_argument("--kernel_size", type=Tuple[int], nargs="+", default=(3, 3))
    parser.add_argument("--frames_in", type=int, default=2)
    parser.add_argument("--frames_out", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)

    # optim
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="logs/resnet/train")

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

    hparams = parser.parse_args()
    logging.info(hparams)

    # get data
    in_shape = (hparams.batch_size, *hparams.size, hparams.n_channels)
    train_dataloader = create_dataloader(
        hparams.root,
        hparams.batch_size,
        hparams.frames_in,
        hparams.frames_out,
        hparams.step,
        in_shape,
        hparams.train_search_regex,
        hparams.preload,
    )
    val_dataloader = create_dataloader(
        hparams.root,
        hparams.batch_size,
        hparams.frames_in,
        hparams.frames_out,
        hparams.step,
        in_shape,
        hparams.val_search_regex,
        hparams.preload,
    )
    #
    n_train_samples = len(train_dataloader.dataset)
    n_train_batches = math.floor(n_train_samples / hparams.batch_size)
    n_val_samples = len(val_dataloader.dataset)
    n_val_batches = math.floor(n_train_batches / hparams.batch_size)

    # init model parameters
    rng = random.PRNGKey(0)
    resnet = ResNet(
        hidden_channels=hparams.n_filters,
        out_channels=hparams.frames_out,
        kernel_size=hparams.kernel_size,
        strides=tuple([1] * len(hparams.kernel_size)),
        padding="SAME",
        depth=hparams.depth,
    )
    out_shape, params = resnet.init(rng, in_shape)

    # init optimiser
    optimiser = optimizers.sgd(hparams.lr)
    optimiser_state = optimiser.init_fn(params)

    loss = 0.0
    for i in tqdm(range(hparams.epochs), desc="Epochs"):
        # training
        pbar = tqdm(train_dataloader, desc="Training samples", total=n_train_batches)
        for batch in pbar:
            batch = jnp.transpose(batch, axes=())[:, 0]
            # x, y = jnp.split(batch, (hparams.frames_in, hparams.frames_out), axis=1)
            loss, optimiser_state = update(resnet, optimiser, i, optimiser_state, batch)
            pbar.set_postfix_str("Loss: {}".format(loss))

        # # validating
        # pbar = tqdm(train_dataloader, desc="Validation samples", total=n_train_batches)
        # for batch in pbar:
        #     # x, y = jnp.split(batch, (hparams.frames_in, hparams.frames_out), axis=1)
        #     x = y = batch
        #     y_hat = resnet.apply(params, batch)[:, 0]
        #     loss = compute_loss(y_hat, y)
        #     pbar.set_postfix_str("Loss: {}".format(loss))
        #     log_step(loss, y_hat, y, "val")
