import os
import timeit

import jax
from jax import numpy as jnp

import cardiax
import deepx


def speed_test(shape):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = 0

    #  init
    seed = 0
    rng = jax.random.PRNGKey(seed)
    diffusivity = deepx.generate.random_diffusivity(rng, shape)
    p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
    s1 = [
        cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.SOUTH, 0.2, 20.0, p1)
    ]
    stimuli = s1
    state = cardiax.solve.init(shape)
    ts = jnp.arange(0, 10_000, 500)
    dt = 0.01
    dx = 0.01
    paramset = cardiax.params.PARAMSET_3

    cardiax_forward = lambda: cardiax.solve.forward(
        state, ts, paramset, diffusivity, stimuli, dx, dt, plot_while=False
    )

    model = deepx.resnet.ResNet(16, 1, 20)
    _, params = model.init(rng, (1, 2, 4) + shape)
    x = jnp.array(*state)[None, None]
    deepx_forward = lambda: deepx.optimise.infer(model, len(ts * dt / 5), params)

    #  warm up
    cardiax_forward()
    deepx_forward()

    #  measure
    cardiax_times = timeit.timeit(cardiax_forward, number=10)
    deepx_times = timeit.timeit(deepx_forward, number=10)


if __name__ == "__main__":
    speed_test((256, 256))