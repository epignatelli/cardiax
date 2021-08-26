import jax
import cardiax
import deepx


shape = (1200, 1200)
reshape = (256, 256)
dt = 0.01
dx = 0.01


# Chaotic
state = cardiax.solve.init(shape)
start = 0
step = 5000
paramset = cardiax.params.PARAMSET_3

p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
s1 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.SOUTH, 0.2, 20.0, p1)]
p2 = cardiax.stimulus.Protocol(40000, 2, 1e9)
s2 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.EAST, 0.2, 20.0, p2)]
stimuli = s1 + s2


seed = 12
rng = jax.random.PRNGKey(seed)
diffusivity = deepx.generate.random_diffusivity(rng, shape)
stop = 250_000
filename = "data/cardiax_{}.hdf5".format(seed)

deepx.generate.sequence(
    start,
    stop,
    step,
    dt,
    dx,
    paramset,
    diffusivity,
    stimuli,
    filename,
    reshape=reshape,
    use_memory=True,
    plot_while=True,
)


# Break-up
state = cardiax.solve.init(shape)
start = 0
step = 5000
paramset = cardiax.params.PARAMSET_5

p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
s1 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.WEST, 0.2, 20.0, p1)]
p2 = cardiax.stimulus.Protocol(40000, 2, 1e9)
s2 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.NORTH, 0.2, 20.0, p2)]
stimuli = s1 + s2

seed = 31
rng = jax.random.PRNGKey(seed)
diffusivity = deepx.generate.random_diffusivity(rng, shape)
stop = 100_000
filename = "data/cardiax_{}.hdf5".format(seed)

deepx.generate.sequence(
    start,
    stop,
    step,
    dt,
    dx,
    paramset,
    diffusivity,
    stimuli,
    filename,
    reshape=reshape,
    use_memory=True,
    plot_while=True,
)


# Spiral
state = cardiax.solve.init(shape)
start = 0
step = 5000
paramset = cardiax.params.PARAMSET_5

p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
s1 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.WEST, 0.2, 20.0, p1)]
p2 = cardiax.stimulus.Protocol(40000, 2, 1e9)
s2 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.NORTH, 0.2, 20.0, p2)]
stimuli = s1 + s2

seed = 1990
rng = jax.random.PRNGKey(seed)
diffusivity = deepx.generate.random_diffusivity(rng, shape)
stop = 100_000
filename = "data/cardiax_{}.hdf5".format(seed)

deepx.generate.sequence(
    start,
    stop,
    step,
    dt,
    dx,
    paramset,
    diffusivity,
    stimuli,
    filename,
    reshape=reshape,
    use_memory=True,
    plot_while=True,
)


# linear wave
state = cardiax.solve.init(shape)
start = 0
step = 5000
paramset = cardiax.params.PARAMSET_5

p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
s1 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.WEST, 0.2, 20.0, p1)]
p2 = cardiax.stimulus.Protocol(40000, 2, 1e9)
s2 = [cardiax.stimulus.linear(shape, cardiax.stimulus.Direction.SOUTH, 0.2, 20.0, p2)]
stimuli = s1 + s2

seed = 5
rng = jax.random.PRNGKey(seed)
diffusivity = deepx.generate.random_diffusivity(rng, shape)
stop = 50_000
filename = "data/cardiax_{}.hdf5".format(seed)

deepx.generate.sequence(
    start,
    stop,
    step,
    dt,
    dx,
    paramset,
    diffusivity,
    stimuli,
    filename,
    reshape=reshape,
    use_memory=True,
    plot_while=True,
)
