from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Boundary(NamedTuple):
    apply: Callable
    restore: Callable


def neumann(order: int = 3) -> jnp.ndarray:
    def apply(x):
        return jax.tree_map(lambda v: jnp.pad(v, 2, mode="edge"), x)

    def restore(x):
        return jax.tree_map(lambda v: v[1:-1], x)

    return Boundary(apply, restore)