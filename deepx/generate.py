from functools import partial
from typing import Any, Dict, Sequence, Tuple

import cardiax
import jax
import jax.numpy as jnp

from . import utils_scars as ipu

Shape = Tuple[int, ...]
Key = jnp.ndarray
Domain = Tuple[int, int]


def random_protocol(
    rng: Key,
    min_start: int = 0,
    max_start: int = 1000,
    min_period: int = 400,
    max_period: int = 1e9,
) -> cardiax.stimulus.Protocol:
    rng_1, rng_2 = jax.random.split(rng)
    start = jax.random.randint(rng_1, (1,), min_start, max_start)
    duration = 2  # always instantaneous
    period = jax.random.randint(rng_2, (1,), min_period, max_period)
    return cardiax.stimulus.Protocol(start, duration, period)


def random_rectangular_stimulus(
    rng: Key, shape: Shape, protocol: cardiax.stimulus.Protocol, modulus: float = 0.6
) -> cardiax.stimulus.Stimulus:
    rng_1, rng_2 = jax.random.split(rng)
    size = jax.random.randint(rng_2, (2,), max(shape[0] // 100, 10), shape[0] // 3)
    centre = jax.random.randint(rng_1, (2,), size.min(), shape[0])
    return cardiax.stimulus.rectangular(shape, centre, size, modulus, protocol)


def random_linear_stimulus(
    rng: Key, shape: Shape, protocol: cardiax.stimulus.Protocol, modulus: float = 0.6
) -> cardiax.stimulus.Stimulus:
    rng_1, rng_2 = jax.random.split(rng)
    direction = jax.random.randint(rng_1, (1,), 0, 3)
    coverage = jax.random.normal(rng_2, (1,))
    return cardiax.stimulus.linear(shape, direction, coverage * 0.5, modulus, protocol)


def random_triangular_stimulus(
    rng: Key, shape: Shape, protocol: cardiax.stimulus.Protocol, modulus: float = 0.6
) -> cardiax.stimulus.Stimulus:
    rng_1, _ = jax.random.split(rng)
    angle, coverage = jax.random.normal(rng_1, (2,))
    direction = jax.random.randint(rng_1, (1,), 0, 3)
    return cardiax.stimulus.triangular(
        shape, direction, angle * 90, coverage * 0.5, modulus, protocol
    )


def random_stimulus(
    rng: Key, shape: Shape, min_start: int = 0
) -> cardiax.stimulus.Stimulus:
    stimuli_fn = (
        random_rectangular_stimulus,
        random_triangular_stimulus,
        random_linear_stimulus,
    )
    rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
    protocol = random_protocol(rng_1, min_start=min_start)
    modulus = 20.0
    stimulus_fn = partial(
        stimuli_fn[jax.random.choice(rng_3, jnp.arange(0, len(stimuli_fn)))],
        shape=shape,
        protocol=protocol,
        modulus=modulus,
    )
    return stimulus_fn(rng_4)


def random_gaussian_1d(rng: Key, length: int):
    rngs = jax.random.split(rng, 3)
    mu = jax.random.normal(rngs[0], (1,)) * 0.2
    sigma = jax.random.normal(rngs[1], (1,)) * 0.3
    x = jnp.linspace(-1, 1, length)
    return jnp.exp(-jnp.power(x - mu, 2.0) / (2 * jnp.power(sigma, 2.0)))


def random_gaussian_2d(rng: Key, shape: Shape):
    rngs = jax.random.split(rng, len(shape))
    x0 = random_gaussian_1d(rngs[0], shape[0])
    x1 = random_gaussian_1d(rngs[1], shape[1])
    return 1 - (jnp.ones(shape) * x0).T * x1


def random_gaussian_mixture(rng: Key, shape: Shape, n_gaussians: int) -> jnp.ndarray:
    rngs = jax.random.split(rng, n_gaussians)
    mixture = [random_gaussian_2d(rngs[i], shape) for i in range(n_gaussians)]
    return sum(mixture) / n_gaussians


def random_diffusivity(
    rng: Key, shape: Shape, domain: Domain = (0.0001, 0.001)
) -> jnp.ndarray:
    c = ipu.random_diffusivity_scar(rng, shape)
    return cardiax.convert.diffusivity_rescale(c, domain)


def rng_sequence(start_seed=0):
    rng = jax.random.PRNGKey(start_seed)
    while True:
        yield jax.random.split(rng)[0]


def random_sequence(
    rng: Key,
    params: cardiax.params.Params,
    filepath: str,
    shape: Shape = (1200, 1200),
    n_stimuli: int = 2,
    start: int = 0,
    stop: int = 1000,
    step: int = 1,
    dt: float = 0.01,
    dx: float = 0.01,
    reshape: Shape = (256, 256),
):
    # generate random stimuli
    rngs = jax.random.split(rng, n_stimuli)
    min_start = jnp.arange(1, stop, 400 * (n_stimuli - 1))
    stimuli = [
        random_stimulus(rngs[i], shape, min_start=min_start[i])
        for i in range(n_stimuli)
    ]

    # generate diffusivity map
    diffusivity = random_diffusivity(rngs[-1], shape)

    # generate sequence
    return sequence(
        start=cardiax.convert.ms_to_units(start, dt),
        stop=cardiax.convert.ms_to_units(stop, dt),
        step=cardiax.convert.ms_to_units(step, dt),
        dt=dt,
        dx=dx,
        params=params,
        diffusivity=diffusivity,
        stimuli=stimuli,
        filename=filepath,
        reshape=reshape,
    )


def sequence(
    start,
    stop,
    step,
    dt,
    dx,
    params,
    diffusivity,
    stimuli,
    filename,
    reshape=None,
):
    # check shapes
    for s in stimuli:
        assert (
            diffusivity.shape == s.field.shape
        ), "Inconsistent stimulus shapes {} and diffusivity {}".format(
            diffusivity.shape, s.field.shape
        )

    # checkpoints
    checkpoints = jnp.arange(int(start), int(stop), int(step))

    # shapes
    shape = diffusivity.shape
    tissue_size = cardiax.convert.shape_to_realsize(shape, dx)

    # print and plot
    print("Tissue size", tissue_size, "Grid size", diffusivity.shape)
    print("Checkpointing at:", checkpoints)
    print("Cell parameters", params)
    cardiax.plot.plot_diffusivity(diffusivity)
    cardiax.plot.plot_stimuli(stimuli)

    # init storage
    init_size = shape if reshape is None else reshape
    hdf5 = cardiax.io.init(
        filename, init_size, n_iter=len(checkpoints), n_stimuli=len(stimuli)
    )
    cardiax.io.add_params(hdf5, params, diffusivity, dt, dx, shape=reshape)
    cardiax.io.add_stimuli(hdf5, stimuli, shape=reshape)
    cardiax.io.add_diffusivity(hdf5, diffusivity, shape=reshape)

    states_dset = hdf5["states"]
    state = cardiax.solve.init(shape)
    for i in range(len(checkpoints) - 1):
        print(
            "Solving at: %dms/%dms\t\t"
            % (
                cardiax.convert.units_to_ms(checkpoints[i + 1], dt),
                cardiax.convert.units_to_ms(checkpoints[-1], dt),
            ),
            end="\r",
        )
        state = cardiax.solve._forward_euler(
            state,
            checkpoints[i],
            checkpoints[i + 1],
            params,
            diffusivity,
            stimuli,
            dt,
            dx,
        )
        cardiax.io.add_state(states_dset, state, i, shape=(len(state), *reshape))

    print()
    hdf5.close()
