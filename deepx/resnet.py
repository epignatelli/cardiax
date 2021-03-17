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


class HParams(NamedTuple):
    seed: int
    log_frequency: int
    debug: bool
    hidden_channels: int
    in_channels: int
    depth: int
    lr: float
    batch_size: int
    lamb: float
    evaluation_steps: int
    epochs: int
    train_maxsteps: int
    val_maxsteps: int
    tbtt: bool
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

    @staticmethod
    def from_flags(flags):
        return HParams(
            seed=flags.seed,
            log_frequency=flags.log_frequency,
            debug=flags.debug,
            hidden_channels=flags.hidden_channels,
            in_channels=flags.in_channels,
            depth=flags.depth,
            lr=flags.lr,
            batch_size=flags.batch_size,
            lamb=flags.lamb,
            evaluation_steps=flags.evaluation_steps,
            epochs=flags.epochs,
            train_maxsteps=flags.train_maxsteps,
            val_maxsteps=flags.val_maxsteps,
            tbtt=flags.tbtt,
            increase_at=flags.increase_at,
            teacher_forcing_prob=flags.teacher_forcing_prob,
            from_checkpoint=flags.from_checkpoint,
            root=flags.root,
            paramset=flags.paramset,
            size=tuple(flags.size),
            frames_in=flags.frames_in,
            frames_out=flags.frames_out,
            step=flags.step,
            refeed=flags.refeed,
            preload=flags.preload,
        )


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


def Euler(axis=1):
    def init_fun(rng, input_shape):
        s0, _ = input_shape
        return (s0[:3] + (s0[3] - 1,) + s0[4:], ())  # remove 1 channel (diffusivity)

    def apply_fun(params, inputs, **kwargs):
        x0, x1 = inputs
        return x0[:, -1:, :-1] + x1

    return init_fun, apply_fun


@module
def ResNet(hidden_channels, out_channels, depth):
    residual = stax.serial(
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), hidden_channels, (4, 5, 5), (1, 1, 1), "SAME"
        ),
        *[
            ResidualBlock(
                hidden_channels,
                (4, 5, 5),
                (1, 1, 1),
                "SAME",
                ("NCDWH", "IDWHO", "NCDWH"),
            )
            for _ in range(depth)
        ],
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), out_channels, (4, 5, 5), (1, 1, 1), "SAME"
        ),
        stax.GeneralConv(("NDCWH", "IDWHO", "NDCWH"), 3, (3, 5, 5), (1, 1, 1), "SAME")
    )
    return stax.serial(stax.FanOut(2), stax.parallel(stax.Identity, residual), Euler())
