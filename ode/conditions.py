import jax
import jax.numpy as jnp


@jax.jit
def neumann(X: jnp.ndarray) -> jnp.ndarray:
    def apply(X: jnp.ndarray):
        return np.pad(X, 1, mode="edge")
    def restore(X: jnp.ndarray) -> X: jnp.ndarray:
        return jnp.take_along_axis(X, slice(1, -1), axis=range(X.ndim))