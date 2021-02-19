import functools
from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.experimental.ode


def euler(
    f: Callable, x: jnp.ndarray, t: float, *f_args: Any, **integrator_kwargs: Any
):
    dt = integrator_kwargs.pop("dt")
    grads = f(x, t, *f_args)
    return jax.tree_multimap(lambda v, dv: jnp.add(v, dv * dt), x, grads)


def rk45(f: Callable, x: jnp.ndarray, t: float, *f_args: Any, **integrator_kwargs: Any):
    return jax.experimental.ode.odeint(f, x, t, *f_args)
