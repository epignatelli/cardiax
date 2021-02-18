import functools
from typing import Callable, Dict, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from . import plot
from .stimulus import Stimulus

Params = NamedTuple  # physical parameters of the equation


def forward(
    step_fn: Callable,
    integrator: Callable,
    x0: jnp.ndarray,
    t_checkpoints: Sequence[int],
    boundary_conditions,
    physical_parameters: Params,
    diffusivity: jnp.ndarray,
    stimuli: Sequence[Stimulus],
    dt: float,
    dx: float,
):
    x = x0
    states = []
    for i in range(len(t_checkpoints) - 1):
        print(
            "Solving at: %dms/%dms\t\t with %d passages"
            % (
                t_checkpoints[i + 1] * dt,
                t_checkpoints[-1] * dt,
                t_checkpoints[i + 1] - t_checkpoints[i],
            ),
            end="\r",
        )
        x = _forward(
            step_fn,
            integrator,
            x,
            t_checkpoints[i],
            t_checkpoints[i + 1],
            boundary_conditions,
            physical_parameters,
            jnp.ones_like(x) * diffusivity,
            stimuli,
            dt,
            dx,
        )
        plot.plot_state(x)
        plt.show()
        states.append(x)
    return states


@functools.partial(jax.jit, static_argnums=(0, 1))
def _forward(
    step_fn,
    integrator,
    x,
    t,
    t_end,
    boundary_conditions,
    physical_parameters,
    diffusivity,
    stimuli,
    dt,
    dx,
):
    def body_fn(i, x):
        x = integrator(
            step_fn,
            x,
            i,
            boundary_conditions,
            physical_parameters,
            diffusivity,
            stimuli,
            dt,
            dx,
        )
        return x

    return jax.lax.fori_loop(t, t_end, body_fn, init_val=x)


# @functools.partial(jax.jit, static_argnums=(0, 1))
# def _forward_stack(
#     step_fn,
#     integrator,
#     x,
#     t,
#     t_end,
#     physical_parameters,
#     diffusion,
#     stimuli,
#     dt,
#     dx,
# ):
#     def body_fun(x, i):
#         new_state = integrator(step_fn, x, i, physical_parameters, diffusion, stimuli, dt, dx)
#         return (new_state, new_state)

#     xs = jnp.arange(t, t_end)
#     _, states = jax.lax.scan(body_fun, x, xs)
#     return states
