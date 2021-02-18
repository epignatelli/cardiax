import functools
from typing import Callable, Any

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnums=(0,))
def euler(
    f: Callable, x: jnp.ndarray, t: float, *f_args: Any, **integrator_kwargs: Any
):
    dt = integrator_kwargs.pop("dt")
    return jax.numpy.add(x, f(x, t, *f_args) * dt)

    # def apply(x):
    #     return jax.numpy.add(x, f(x) * dt)

    # return jax.tree_multimap(apply, x, dt)


@functools.partial(jax.jit, static_argnums=(0,))
def rk45(f: Callable, x: jnp.ndarray, t: float, *f_args: Any, **integrator_kwargs: Any):
    return jax.experimental.ode.odeint(f, x, t, *f_args)
