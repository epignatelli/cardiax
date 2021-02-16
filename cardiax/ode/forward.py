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
    cell_parameters: Params,
    diffusivity: jnp.ndarray,
    stimuli: Sequence[Stimulus],
    dt: float,
    dx: float,
):
    """
    Solves the function with the integration scheme of choice.
    Units are adimensional.
    Args:
        shape (Tuple[int, int]): The shape of the finite difference grid
        t_checkpoints (iter): An iterable that contains time steps in simulation units, at which pause, and display the state of the system
        cell_parameters (Dict[string, float]): Dictionary of physiological parameters as illustrated in Fenton, Cherry, 2002.
        diffusivity (float): Diffusivity of the cardiac tissue
        stimuli (List[Dict[string, object]]): A list of stimuli to provide energy to the tissue
        dt (float): time infinitesimal to use in the euler stepping scheme
        dx (float): space infinitesimal to use in the spatial gradient calculation
    Returns:
        (List[jax.numpy.ndarray]): The list of states at each checkpoint
    """

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
            cell_parameters,
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
    params,
    D,
    stimuli,
    dt,
    dx,
):
    odeint = lambda i, state: integrator(step_fn, x, i, params, D, stimuli, dt, dx)
    return jax.lax.fori_loop(t, t_end, odeint, init_val=x)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _forward_stack(
    step_fn,
    integrator,
    x,
    t,
    t_end,
    params,
    diffusion,
    stimuli,
    dt,
    dx,
):
    def body_fun(x, i):
        new_state = integrator(step_fn, x, i, params, diffusion, stimuli, dt, dx)
        return (new_state, new_state)

    xs = jnp.arange(t, t_end)
    _, states = jax.lax.scan(body_fun, x, xs)
    return states
