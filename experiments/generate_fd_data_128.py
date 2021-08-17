import jax
import cardiax
import deepx
import jax.numpy as jnp


shape = (600, 600)
reshape = (128, 128)
state = cardiax.solve.init(shape)
start = 0
stop = 200_000
step = 500  #  5 milliseconds
dt = 0.01
dx = 0.01
paramset = cardiax.params.PARAMSET_3
n_refeed = 100


if __name__ == "__main__":
    #  setup
    filename = "data/heterogeneous_spiral_128_{}-{}.hdf5"
    #  Finite difference simulation
    p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
    p2 = cardiax.stimulus.Protocol(30_000, 2, 1e9)
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        angle = int(
            jax.random.randint(
                rng,
                (1,),
                0,
                180,
            )
        )
        s1 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
        )
        s2 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle + 90, 0.2, 20, p2
        )
        stimuli = [s1, s2]
        diffusivity = deepx.generate.random_diffusivity(rng, shape)
        deepx.generate.sequence(
            start,
            stop,
            step,
            dt,
            dx,
            paramset,
            diffusivity,
            stimuli,
            filename.format(seed, "fd"),
            reshape=reshape,
            use_memory=True,
            plot_while=False,
        )

    filename = "data/heterogeneous_linear_128_{}-{}.hdf5"
    #  Finite difference simulation
    p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
    p2 = cardiax.stimulus.Protocol(30_000, 2, 1e9)
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        angle = int(
            jax.random.randint(
                rng,
                (1,),
                0,
                180,
            )
        )
        s1 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
        )
        s2 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p2
        )
        stimuli = [s1, s2]
        diffusivity = deepx.generate.random_diffusivity(rng, shape)
        deepx.generate.sequence(
            start,
            stop,
            step,
            dt,
            dx,
            paramset,
            diffusivity,
            stimuli,
            filename.format(seed, "fd"),
            reshape=reshape,
            use_memory=True,
            plot_while=False,
        )

    filename = "data/homogeneous_spiral_128_{}-{}.hdf5"
    #  Finite difference simulation
    p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
    p2 = cardiax.stimulus.Protocol(30_000, 2, 1e9)
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        angle = int(
            jax.random.randint(
                rng,
                (1,),
                0,
                180,
            )
        )
        s1 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
        )
        s2 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle + 90, 0.2, 20, p2
        )
        stimuli = [s1, s2]
        diffusivity = jnp.ones(shape) * 0.001
        deepx.generate.sequence(
            start,
            stop,
            step,
            dt,
            dx,
            paramset,
            diffusivity,
            stimuli,
            filename.format(seed, "fd"),
            reshape=reshape,
            use_memory=True,
            plot_while=False,
        )

    filename = "data/homogeneous_linear_128_{}-{}.hdf5"
    #  Finite difference simulation
    p1 = cardiax.stimulus.Protocol(0, 2, 1e9)
    p2 = cardiax.stimulus.Protocol(30_000, 2, 1e9)
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        angle = int(
            jax.random.randint(
                rng,
                (1,),
                0,
                180,
            )
        )
        s1 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
        )
        s2 = cardiax.stimulus.triangular(
            shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p2
        )
        stimuli = [s1, s2]
        diffusivity = jnp.ones(shape) * 0.001
        deepx.generate.sequence(
            start,
            stop,
            step,
            dt,
            dx,
            paramset,
            diffusivity,
            stimuli,
            filename.format(seed, "fd"),
            reshape=reshape,
            use_memory=True,
            plot_while=False,
        )
