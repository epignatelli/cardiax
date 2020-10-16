import jax
import jax.numpy as jnp


@jax.jit
def laplacian_kernel():
    ux_kernel = [
        [1/12, -2/3, 0, 2/3, 1/12],
        [1/12, -2/3, 0, 2/3, 1/12],
        [1/12, -2/3, 0, 2/3, 1/12],
        [1/12, -2/3, 0, 2/3, 1/12],
        [1/12, -2/3, 0, 2/3, 1/12],
    ]
    uy_kernel = jnp.transpose(ux_kernel, (1, 0))
    return ux_kernel, uy_kernel


@functools.partial(jax.jit, static_argnums=1)
def gradient(a, axis):
    sliced = functools.partial(jax.lax.slice_in_dim, a, axis=axis)
    a_grad = np.concatenate((
        # 3th order edge
        ((-11/6) * sliced(0, 2) + 3 * sliced(1, 3) - (3/2) * sliced(2, 4) + (1/3) * sliced(3, 5)),
        # 4th order inner
        ((1/12) * sliced(None, -4) - (2/3) * sliced(1, -3) + (2/3) * sliced(3, -1) - (1/12) * sliced(4, None)),
        # 3th order edge
        ((-1/3) * sliced(-5, -3) + (3/2) * sliced(-4, -2) - 3 * sliced(-3, -1) + (11/6) * sliced(-2, None))
    ), axis)
    return a_grad