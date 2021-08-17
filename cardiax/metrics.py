import jax
import jax.numpy as jnp


def adp(x, perc):
    pass


def restitution(x, perc):
    pass


def electrogram(x, point):
    """Returns the value of the electrogram at the specified point.
    x is a 3-dimensional tensor where the first dimension is time."""
    #  generate grid
    c_y, c_x = jnp.ogrid[: x.shape[-1], : x.shape[-2]]
    #  inverse radius proportionality
    dist = jnp.sqrt((c_x - point[0]) ** 2 + (c_y - point[1]) ** 2)
    print(dist)
    #  integrate
    return jnp.sum(x * dist, axis=(-1, -2))


def spiral_centres(x):
    pass
