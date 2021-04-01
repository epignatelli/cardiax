import jax
import jax.numpy as jnp


def Neumann(window=1):
    def apply(pytree):

        return jax.tree_map(lambda x: jnp.pad(x, (window, window), mode="edge"), pytree)

    def restore(pytree):
        return jax.tree_map(lambda x: jnp.pad(x, (-window, -window)), pytree)

    return apply, restore