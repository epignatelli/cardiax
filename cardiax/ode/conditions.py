from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Boundary(NamedTuple):
    apply: Callable
    restore: Callable


def neumann(order=2) -> jnp.ndarray:
    def apply(x):
        return jax.tree_map(lambda v: jnp.pad(v, 2, mode="edge"), x)

    def restore(x):
        def restore_axis(v, axis):
            v = jax.lax.slice_in_dim(v, order, v.shape[axis] - order, axis=axis)
            return v

        def restore_axes(y):
            y = jnp.apply_over_axes(restore_axis, y, jnp.arange(0, len(y.shape)))
            return y

        return jax.tree_map(restore_axes, x)

    return Boundary(apply, restore)