import math
from typing import Tuple, NamedTuple, Callable, Any
import functools

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.stax import Elu, FanInSum, FanOut, GeneralConv, Identity


Params = Any
RNGKey = jnp.ndarray
Shape = Tuple[int]


class Module(NamedTuple):
    init: Callable[[RNGKey, Shape], Tuple[Shape, Params]]
    apply: Callable[[Params, jnp.ndarray], jnp.ndarray]
    inverse: Callable[[Params, jnp.ndarray], jnp.ndarray] = None


def module(module_maker):
    @functools.wraps(module_maker)
    def fabricate_module(*args, **kwargs):
        functions = module_maker(*args, **kwargs)
        return Module(*functions)

    return fabricate_module


@module
def Split(axis):
    def init(axis):
        return ()

    def apply(params, x):
        return jnp.split(x, 2, axis=axis)

    return init, apply


@module
def AddLastItem(axis):
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: inputs[0] + jax.lax.slice_in_dim(
        inputs[1], inputs[1].shape[axis] - 1, inputs[1].shape[axis], axis=axis
    )
    return init_fun, apply_fun


@module
def ConvBlock(out_channels, kernel_size, input_format):
    return stax.serial(
        GeneralConv(input_format, out_channels, kernel_size, 1, "SAME"),
        Elu,
        GeneralConv(input_format, out_channels, kernel_size, 1, "SAME"),
    )


@module
def ReversibleBlock(out_channels, kernel_size, input_format):
    f = ConvBlock(out_channels, kernel_size, input_format)
    g = ConvBlock(out_channels, kernel_size, input_format)

    def init_func(rng, input_shape):
        params_f, output_shape_f = f.init_func(rng, input_shape)
        params_g, output_shape_g = g.init_func(rng, input_shape)
        return ((params_f, params_g), (output_shape_f, output_shape_g))

    def apply_fun(params, x):
        params_f, params_g = params
        x1, x2 = x
        y1 = x1 + f.apply_fun(params_f, x2)
        y2 = x2 + g.apply_fun(params_g, y1)
        return (y1, y2)

    def inverse_fun(params, y):
        params_f, params_g = params
        y1, y2 = y
        x2 = y2 - g.apply_fun(params_g, y1)
        x1 = y1 - f.apply_fun(params_f, x2)
        return (x1, x2)

    return init_func, apply_fun, inverse_fun


def RevNet(
    hidden_channels, out_channels, kernel_size, strides, padding, depth, input_format
):
    residual = stax.serial(  #
        Split(input_format[0].lower().index("c")),
        GeneralConv(input_format, hidden_channels, kernel_size, strides, padding),
        *[
            ReversibleBlock(hidden_channels, kernel_size, input_format)
            for _ in range(depth)
        ],
        GeneralConv(input_format, out_channels, kernel_size, strides, padding)
    )
    return Module(
        *stax.serial(FanOut(2), stax.parallel(residual, Identity), AddLastItem(1))
    )