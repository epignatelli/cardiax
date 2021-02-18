import functools
from typing import Callable, Any

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(0,))
def euler(
    step_fn: Callable,
    x: jnp.ndarray,
    t: float,
    *args: Any,
):
    return jax.numpy.add(x, step_fn(x, t, *args) * dt)
    # def apply(x):
    #     return jax.numpy.add(x, step_fn(x) * dt)

    # return jax.tree_multimap(apply, x, dt)


@functools.partial(jax.jit, static_argnums=(0,))
def rk(
    step_fn: Callable,
    x: jnp.ndarray,
    t: float,
    *args: Any,
):
    return jax.experimental.ode.odeint(step_fn, x, t, *args)
