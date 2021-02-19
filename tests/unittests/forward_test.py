import jax
import jax.numpy as jnp

from cardiax import fk, ode
from cardiax.fk import params
from cardiax.ode.conditions import neumann
from cardiax.ode import stimulus
from deepx import generate


def test_fk_step():
    # set hparams
    rng = jax.random.PRNGKey(0)
    shape = (32, 32)
    state = fk.solve.init(shape)
    boundary = neumann()
    parameter_set = params.PARAMSET_3
    diffusivity = generate.random_diffusivity(rng, shape, 3)
    stimuli = [
        stimulus.linear(
            shape, stimulus.Direction.NORTH, 0.2, 0.6, stimulus.Protocol(0, 2, 1e9)
        )
    ]
    dx = 0.01

    # test python fun
    fk.solve.step(state, 0, boundary, parameter_set, diffusivity, stimuli, dx)

    # test xla fun
    jax.jit(fk.solve.step, static_argnums=(2,))(
        state, 0, boundary, parameter_set, diffusivity, stimuli, dx
    )
    return


def test_fk_forward():
    # set hparams
    rng = jax.random.PRNGKey(0)
    shape = (1200, 1200)
    state = fk.solve.init(shape)
    boundary = neumann()
    parameter_set = params.PARAMSET_3
    diffusivity = generate.random_diffusivity(rng, shape)
    stimuli = [
        stimulus.linear(
            shape, stimulus.Direction.NORTH, 0.2, 0.6, stimulus.Protocol(0, 2, 1e9)
        )
    ]
    dx = 0.1
    dt = 0.1

    # test
    ode.solve.forward(
        fk.solve.step,
        state,
        jnp.arange(0, 200000, 12000),
        boundary,
        parameter_set,
        diffusivity,
        stimuli,
        dt,
        dx,
        plot=True,
    )
    return


if __name__ == "__main__":
    # test_fk_step()
    test_fk_forward()