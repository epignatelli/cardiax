from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Boundary(NamedTuple):
    apply: Callable
    restore: Callable


def neumann(x: jnp.ndarray, func: Callable, order: int = 3) -> jnp.ndarray:
    def apply(x):
        return jnp.pad(x, 2, mode="edge")

    def restore(x):
        return jax.tree_map(lambda v: v[1:-1], x)

    return apply, restore