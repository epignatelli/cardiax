import jax
import jax.numpy as jnp

import cardiax
from cardiax import params
from deepx import generate


def test_fk_step():
    # set hparams
    rng = jax.random.PRNGKey(0)
    shape = (32, 32)
    state = cardiax.solve.init(shape)
    parameter_set = params.PARAMSET_3
    diffusivity = generate.random_diffusivity(rng, shape, 3)
    stimuli = [
        cardiax.stimulus.linear(
            shape,
            cardiax.stimulus.Direction.NORTH,
            0.2,
            0.6,
            cardiax.stimulus.Protocol(0, 2, 1e9),
        )
    ]
    dt = 0.01
    dx = 0.01

    # test python fun
    cardiax.solve.step(state, 0, parameter_set, diffusivity, stimuli, dt, dx)

    # test xla fun
    jax.jit(cardiax.solve.step, static_argnums=(2,))(
        state, 0, parameter_set, diffusivity, stimuli, dt, dx
    )
    return


def test_fk_forward():
    # set hparams
    rng = jax.random.PRNGKey(0)
    shape = (1200, 1200)
    state = cardiax.solve.init(shape)
    parameter_set = params.PARAMSET_3
    diffusivity = generate.random_diffusivity(rng, shape)
    stimuli = [
        cardiax.stimulus.linear(
            shape,
            cardiax.stimulus.Direction.NORTH,
            0.2,
            0.6,
            cardiax.stimulus.Protocol(0, 2, 1e9),
        )
    ]
    dx = 0.1
    dt = 0.1

    # test
    cardiax.solve.forward(
        state,
        jnp.arange(0, 200000, 12000),
        parameter_set,
        diffusivity,
        stimuli,
        dt,
        dx,
        plot_while=False,
    )
    return


if __name__ == "__main__":
    test_fk_step()
    test_fk_forward()