from typing import NamedTuple

import jax
import jax.numpy as jnp
from helx.methods import pmodule
from helx.types import Module
from jax.experimental import stax


class HParams(NamedTuple):
    seed: int
    log_frequency: int
    debug: bool
    hidden_channels: int
    in_channels: int
    depth: int
    denseconv_step: int
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
            denseconv_step=flags.denseconv_step,
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


def ResidualBlock(out_channels):
    double_conv = stax.serial(
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), out_channels, (4, 3, 3), (1, 1, 1), "SAME"
        ),
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), out_channels, (1, 1, 1), (1, 1, 1), "SAME"
        ),
        stax.Elu,
    )
    return Module(
        *stax.serial(
            stax.FanOut(2),
            stax.parallel(
                double_conv,
                stax.Identity,
            ),
            stax.FanInSum,
        )
    )


def DenseConvBlock(out_channels, depth):
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(
            stax.serial(*[ResidualBlock(out_channels) for _ in depth]),
            stax.Identity,
        ),
        stax.FanInSum,
    )


def Euler():
    def init(rng, input_shape):
        # (b, t, c, w, h)
        input_shape, _ = input_shape
        return (input_shape[0], 1) + input_shape[3:], ()

    def apply(params, inputs, **kwargs):
        x0, x1 = inputs
        #  x0 is the input, so (b, 2, 4, w, h)
        #  x1 is the output, so (b, 1, 4, w, h)
        #  we discard the channel deriving from the diff map
        return jnp.add(x0[:, -2:-1, :3], x1[:, :, :3])

    return init, apply


@pmodule
def ResNet(hidden_channels, out_channels, depth, denseconv_step):
    denseconv_depth = depth // denseconv_step
    # time integration module
    backbone = stax.serial(
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), hidden_channels, (4, 3, 3), (1, 1, 1), "SAME"
        ),
        *[DenseConvBlock(hidden_channels, denseconv_depth) for _ in range(depth)],
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), out_channels, (4, 3, 3), (1, 1, 1), "SAME"
        ),
    )

    #  euler scheme
    return stax.serial(stax.FanOut(2), stax.parallel(stax.Identity, backbone), Euler())
