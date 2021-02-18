import jax
import jax.numpy as jnp

from cardiax import fk, ode
from cardiax.fk.params import PARAMSET_5
from cardiax.ode.conditions import neumann
from deepx import generate


def test_fk_step():
    # set hparams
    rng = jax.random.PRNGKey(0)
    shape = (32, 32)
    state = fk.solve.init(shape)
    boundary = neumann()
    params = PARAMSET_5
    diffusivity = generate.random_diffusivity(rng, shape, 3)
    stimuli = [generate.random_stimulus(rng, shape)]
    dx = 0.01

    # test python fun
    fk.solve.step(state, 0, boundary, params, diffusivity, stimuli, dx)

    # test xla fun
    jax.jit(fk.solve.step, static_argnums=(2,))(
        state, 0, boundary, params, diffusivity, stimuli, dx
    )
    return


def test_fk_forward():
    # set hparams
    rng = jax.random.PRNGKey(0)
    shape = (32, 32)
    state = fk.solve.init(shape)
    boundary = neumann()
    params = PARAMSET_5
    diffusivity = generate.random_diffusivity(rng, shape, 3)
    stimuli = [generate.random_stimulus(rng, shape)]
    dx = 0.01
    dt = 0.01

    # test
    ode.forward.forward(
        fk.solve.step,
        state,
        (0, 100, 200, 300, 400),
        boundary,
        params,
        diffusivity,
        stimuli,
        dt,
        dx,
        plot=False,
    )
    return


if __name__ == "__main__":
    test_fk_step()
    test_fk_forward()