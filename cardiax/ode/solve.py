import functools
from typing import Callable, Dict, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from . import integrate
from .conditions import Boundary
from .plot import plot_state, plot_diffusivity, plot_stimuli
from .stimulus import Stimulus

Params = NamedTuple  # physical parameters of the equation


def forward(
    step_fn: Callable,
    x0: jnp.ndarray,
    t_checkpoints: Sequence[int],
    boundary_conditions: Boundary,
    physical_parameters: Params,
    diffusivity: jnp.ndarray,
    stimuli: Sequence[Stimulus],
    dt: float,
    dx: float,
    plot=True,
):
    if plot:
        plot_diffusivity(diffusivity)
        plot_stimuli(stimuli)
        plt.show()

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
        # x = _forward(
        #     step_fn,
        #     x,
        #     t_checkpoints[i],
        #     t_checkpoints[i + 1],
        #     boundary_conditions,
        #     physical_parameters,
        #     diffusivity,
        #     stimuli,
        #     dt,
        #     dx,
        # )

        x = integrate.rk45(
            step_fn,
            x,
            t_checkpoints,
            boundary_conditions,
            physical_parameters,
            diffusivity,
            stimuli,
            dt,
            dx,
        )

        states.append(x)
        if plot:
            plot_state(x)
            plt.show()
    return states


# @functools.partial(jax.jit, static_argnums=(0, 4))
def _forward(
    step_fn: Callable,
    x: jnp.ndarray,
    t: int,
    t_end: int,
    boundary_conditions: Boundary,
    physical_parameters: Params,
    diffusivity: jnp.ndarray,
    stimuli: Sequence[Stimulus],
    dt: float,
    dx: float,
):
    def body_fn(t, x):
        return integrate.rk45(
            step_fn,
            x,
            t,
            boundary_conditions,
            physical_parameters,
            diffusivity,
            stimuli,
            dx,
            dt=dt,
        )

    return jax.lax.fori_loop(t, t_end, body_fn, init_val=x)
