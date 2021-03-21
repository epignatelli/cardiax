import logging
from functools import partial
from typing import NamedTuple, Tuple

import cardiax
import jax
import jax.numpy as jnp
import wandb
from helx.methods import module, batch, pmodule
from helx.types import Module, Optimiser, OptimizerState, Params, Shape
from jax.experimental import stax


class HParams(NamedTuple):
    seed: int
    log_frequency: int
    debug: bool
    hidden_channels: int
    in_channels: int
    out_channels: int
    n_heads: int
    depth: int
    input_format: Tuple[str, str, str]
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


def SelfAttentionBlock(n_heads):
    conv_init, conv_apply = stax.GeneralConv(
        ("NCDWH", "IDWHO", "NCDWH"), n_heads, (4, 1, 1), (1, 1, 1), "SAME"
    )

    def init(rng, input_shape):
        rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
        _, f = conv_init(rng_1, input_shape)
        _, g = conv_init(rng_2, input_shape)
        _, h = conv_init(rng_3, input_shape)
        _, v = conv_init(rng_4, input_shape)
        return input_shape, (f, g, h, v)

    def apply(params, inputs, **kwargs):
        f, g, h, v = params
        # fx, gx, hx = jax.vmap(conv_apply, in_axes=1, out_axes=1)(
        #     jnp.stack(fgh), jnp.stack(inputs)
        # )
        fx = conv_apply(f, inputs)
        gx = conv_apply(g, inputs)
        hx = conv_apply(h, inputs)
        sx = jnp.matmul(jnp.swapaxes(fx, -2, -1), gx)
        sx = jax.nn.softmax(sx, axis=-2)
        sx = jnp.matmul(sx, hx)
        vx = conv_apply(v, sx)
        return vx

    return (init, apply)


def ConvBlock(out_channels):
    return stax.serial(
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), out_channels, (4, 3, 3), (1, 1, 1), "SAME"
        ),
        stax.Elu,
    )


def ResBlock(out_channels, n_heads):
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(stax.Identity, ConvBlock(out_channels)),
        stax.parallel(
            stax.Identity,
            SelfAttentionBlock(n_heads),
        ),
        stax.FanInSum,
    )


def ResMHDPABlock(hidden_channels):
    block = stax.serial(ConvBlock(hidden_channels), SelfAttentionBlock(4))
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(block, stax.Identity),
        stax.FanInSum,
    )


@pmodule
def SelfAttentionResNet(hidden_channels, out_channels, depth):
    # time integration module
    backbone = stax.serial(
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), hidden_channels, (4, 3, 3), (1, 1, 1), "SAME"
        ),
        *[ResMHDPABlock(hidden_channels) for _ in range(depth)],
        stax.GeneralConv(
            ("NCDWH", "IDWHO", "NCDWH"), out_channels, (4, 3, 3), (1, 1, 1), "SAME"
        ),
        stax.GeneralConv(("NDCWH", "IDWHO", "NDCWH"), 3, (3, 3, 3), (1, 1, 1), "SAME"),
    )

    #  euler scheme
    return stax.serial(
        stax.FanOut(2), stax.parallel(stax.Identity, backbone), stax.FanInSum
    )