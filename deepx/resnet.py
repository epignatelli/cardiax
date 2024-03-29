from typing import NamedTuple

import jax.numpy as jnp
from helx.nn.module import pmodule, Module
from jax.experimental import stax


class HParams(NamedTuple):
    seed: int
    log_frequency: int
    debug: bool
    hidden_channels: int
    in_channels: int
    depth: int
    lr: float
    grad_norm: float
    normalise: bool
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
    test_refeed: int
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
            grad_norm=flags.grad_norm,
            normalise=flags.normalise,
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
            test_refeed=flags.test_refeed,
            preload=flags.preload,
        )


def ResidualBlock(out_channels, kernel_size, stride, padding, input_format):
    double_conv = stax.serial(
        stax.GeneralConv(input_format, out_channels, kernel_size, stride, padding),
        stax.Elu,
    )
    return Module(
        *stax.serial(
            stax.FanOut(2), stax.parallel(double_conv, stax.Identity), stax.FanInSum
        )
    )


def conv_params(input_format, out_channels, kernel_size, stride, padding):
    return stax.GeneralConv(input_format, out_channels, kernel_size, stride, padding)[0]


def MomentumBlock(out_channels, kernel_size, stride, padding, input_format, gamma=0.9):
    def init(rng, input_shape):
        params = conv_params(input_format, out_channels, kernel_size, stride, padding)(
            rng, input_shape
        )
        return params

    def apply(params, x, **kwargs):
        params


def Euler():
    def init(rng, input_shape):
        # (b, t, c, w, h)
        input_shape, _ = input_shape
        return (input_shape[0], 1) + input_shape[3:], ()

    def apply(params, inputs, **kwargs):
        x0, x1 = inputs
        return jnp.add(x0[:, -2:-1, :3], x1)

    return init, apply


@pmodule
def ResNet(hidden_channels, out_channels, depth):
    # time integration module
    backbone = stax.serial(
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
        ),
        stax.GeneralConv(("NDCWH", "IDWHO", "NDCWH"), 3, (3, 3, 3), (1, 1, 1), "SAME"),
    )

    #  euler scheme
    return stax.serial(stax.FanOut(2), stax.parallel(stax.Identity, backbone), Euler())
