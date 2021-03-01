from functools import partial
from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
import cardiax

from deepx import utils_scars as ipu

Shape = Tuple[int, ...]
Key = jnp.ndarray


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
    xs = jax.random.randint(rng_1, (2,), 0, shape[0] // 3)
    ys = jax.random.randint(rng_2, (2,), 0, shape[0] // 3)
    centre = (xs[0], ys[0])
    size = (xs[1], ys[1])
    return cardiax.stimulus.rectangular(shape, centre, size, modulus, protocol)


def random_linear_stimulus(
    rng: Key, shape: Shape, protocol: cardiax.stimulus.Protocol, modulus: float = 0.6
) -> cardiax.stimulus.Stimulus:
    rng_1, rng_2 = jax.random.split(rng)
    direction = jax.random.randint(rng_1, (1,), 0, 3)
    coverage = jax.random.normal(rng_2, (1,))
    return cardiax.stimulus.linear(shape, direction, coverage, modulus, protocol)


def random_triangular_stimulus(
    rng: Key, shape: Shape, protocol: cardiax.stimulus.Protocol, modulus: float = 0.6
) -> cardiax.stimulus.Stimulus:
    rng_1, _ = jax.random.split(rng)
    angle, coverage = jax.random.normal(rng_1, (2,))
    direction = jax.random.randint(rng_1, (1,), 0, 3)
    return cardiax.stimulus.triangular(
        shape, direction, angle, coverage, modulus, protocol
    )


def random_stimulus(
    rng: Key, shape: Shape, max_start: int = 0
) -> cardiax.stimulus.Stimulus:
    stimuli_fn = (
        random_rectangular_stimulus,
        random_triangular_stimulus,
        random_linear_stimulus,
    )
    rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
    protocol = random_protocol(rng_1, max_start=max_start)
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
    rng: Key, shape: Shape, n_gaussians: int = 3, domain=(0.0001, 0.001)
) -> jnp.ndarray:
    c = random_gaussian_mixture(rng, shape, n_gaussians)
    a, b = c.min(), c.max()
    y, z = domain[0], domain[1]
    return (c - a) * (z - y) / (b - a) + y

def random_diffusivity_scar(params: dict = ipu.def_params, SAVE_SCAR: bool = False):
    
    VALID_SCAR = False
    CentroidSpline = ipu.CreateSplineCentroids(params)

    # Create individual blobs, scale them up and combine them
    while not VALID_SCAR:
        try:
            res_dict = ipu.MakeAndSumCompositeBlob(params, CentroidSpline)
        
            # taper the edges
            SoftenedComposite, avg_edge_size_pixel, avg_edge_size_prop, GaussShape = ipu.SoftenPolyAndSplineCurve(
                res_dict['CompositeSplineMask'], GaussShape = None, GaussSigma= params['GaussSigma'], 
                AvgEdgeSize = params['RequiredAvgEdgeSize'])
            
            VALID_SCAR = True                    
        
        except ValueError as err:
            print('Attempt at generating random scar map failed because of a ValueError. Trying again')
            VALID_SCAR = False
            
        except:
            print('Attempt at generating random scar map failed because of an unexpected error. Trying again')
            VALID_SCAR = False
    
    assert(isinstance(SoftenedComposite, (np.ndarray, np.generic)))
    
    if SAVE_SCAR:
        ipu.save_scar_as_array(SoftenedComposite, params = params, 
                            root_file_name = ipu.def_root_file_name)
        
    # returned as a numpy array
    return SoftenedComposite

def random_diffusivity_load_scar(
    shortID: str = ipu.def_shortID, 
    root_file_name: str = ipu.def_root_file_name):
    #load scar from file
    SoftenedComposite = ipu.load_scar_as_array(shortID = shortID, 
                                            root_file_name = root_file_name)
    return SoftenedComposite
    
    
def random_sequence(
    rng: Key,
    cell_parameters: cardiax.params.Params,
    filepath: str,
    shape: Sequence[int] = (1200, 1200),
    n_stimuli: int = 3,
    n_scars: int = 3,
    start: int = 0,
    stop: int = 1000,
    dt: float = 0.01,
    dx: float = 0.01,
    reshape: Sequence[int] = (256, 256),
    save_interval_ms: int = 1,
):
    # generate random stimuli
    rngs = jax.random.split(rng, n_stimuli)
    maxstarts = [1, 300, 500]
    stimuli = [random_stimulus(rngs[i], shape, maxstarts[i]) for i in range(n_stimuli)]

    # generate diffusivity map
    diffusivity = random_diffusivity(rngs[-1], shape, n_scars)

    # generate sequence
    return sequence(
        start=cardiax.convert.ms_to_units(start, dt),
        stop=cardiax.convert.ms_to_units(stop, dt),
        dt=dt,
        dx=dx,
        cell_parameters=cell_parameters,
        diffusivity=diffusivity,
        stimuli=stimuli,
        filename=filepath,
    )


def sequence(
    start,
    stop,
    dt,
    dx,
    cell_parameters,
    diffusivity,
    stimuli,
    filename,
    reshape=None,
    save_interval_ms=1,
):
    # check shapes
    for s in stimuli:
        assert (
            diffusivity.shape == s.field.shape
        ), "Inconsistent stimulus shapes {} and diffusivity {}".format(
            diffusivity.shape, s.field.shape
        )

    # checkpoints
    checkpoints = jnp.arange(
        int(start), int(stop), int(save_interval_ms / dt)
    )  # this guarantees a checkpoint every ms

    # shapes
    shape = diffusivity.shape
    tissue_size = cardiax.convert.shape_to_realsize(shape, dx)

    # print and plot
    print("Tissue size", tissue_size, "Grid size", diffusivity.shape)
    print("Checkpointing at:", checkpoints)
    print("Cell parameters", cell_parameters)
    cardiax.plot.plot_diffusivity(diffusivity)
    cardiax.plot.plot_stimuli(*stimuli)

    # init storage
    init_size = reshape or shape
    hdf5 = cardiax.io.init(
        filename, init_size, n_iter=len(checkpoints), n_stimuli=len(stimuli)
    )
    cardiax.io.add_params(hdf5, cell_parameters, diffusivity, dt, dx, shape=reshape)
    cardiax.io.add_stimuli(hdf5, stimuli, shape=reshape)

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
        state = cardiax.solve._forward(
            state,
            checkpoints[i],
            checkpoints[i + 1],
            cell_parameters,
            diffusivity,
            stimuli,
            dt,
            dx,
        )
        cardiax.io.add_state(states_dset, state, i, shape=(len(state), *reshape))

    print()
    hdf5.close()
