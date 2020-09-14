import jax
import jax.numpy as np
import functools
import matplotlib.pyplot as plt
import plot


def forward(shape,
            checkpoints,
            diffusivity,
            params,
            stimulus,
            dt, dx):
    """
    Solves the Fitzhugh-Nagumo model using second order finite difference and explicit euler update scheme.
    Neumann conditions are considered at the boundaries.
    Units are adimensional.
    Args:
        shape (Tuple[int, int]): The shape of the finite difference grid
        checkpoints (iter): An iterable that contains time steps in simulation units, at which pause, and display the state of the system
        params (Dict[string, float]): Dictionary of physiological parameters
        dt (float): time infinitesimal to use in the euler stepping scheme
        dx (float): space infinitesimal to use in the spatial gradient calculation
    Returns:
        (List[jax.numpy.ndarray]): The list of states at each checkpoint
    """
    state = init(shape)

    states = []

    # add initial state
    plot.plot_state(state)
    plt.show()
    states.append(state[2])
    for i in range(len(checkpoints) - 1):
        print("Solving at: %dms/%dms\t\t" % (checkpoints[i + 1], checkpoints[-1]), end="\r")
        state = _forward(state, checkpoints[i], checkpoints[i + 1], diffusivity, params, stimulus, dt, dx)
        plot.plot_state(state)
        plt.show()
        states.append(state[2])
    return states


@jax.jit
def _forward(state, t, t_end, diffusivity, params, stimulus, dt, dx):
    # iterate
    state = jax.lax.fori_loop(t, t_end, lambda i, state: step(state, i, diffusivity, params, stimulus, dt, dx), state)
    return state


@functools.partial(jax.jit, static_argnums=(1, 2))
def _forward_stack(state, t, t_end, diffusivity, params, stimulus, dt, dx):
    # iterate
    def _step(state, i):
        new_state = step(state, i, diffusivity, params, stimulus, dt, dx)
        return (new_state, new_state)
    xs = np.arange(t, t_end)
    last_state, states = jax.lax.scan(_step, state, xs)
    return states


@functools.partial(jax.jit, static_argnums=0)
def init(shape):
    rng = jax.random.PRNGKey(0)
    v = np.ones(shape) + jax.random.normal(rng, shape, dtype=np.float32) * 0.1
    w = np.zeros(shape)
    state = (v, w)
    state = np.asarray(state)
    return state


@jax.jit
def step(state, t, diffusivity, params, stimulus, dt, dx):
    v, w = state

    # apply boundary conditions
    v = neumann(v)
    w = neumann(w)

    # compute laplacian
    v_x = gradient(v, 0) / dx
    v_y = gradient(v, 1) / dx
    v_xx = gradient(v_x, 0) / dx
    v_yy = gradient(v_y, 1) / dx
    laplacian_v = (v_xx + v_yy) + v * (v - params.a) * (1 - v)

    # compute time gradients
    d_v = laplacian_v + v + v ** 3 + w
    d_w = params.a * v + params.b * w

    # euler update
    v += d_v * dt
    w += d_w * dt
    return np.asarray((v, w))


@functools.partial(jax.jit, static_argnums=1)
def gradient(a, axis):
    sliced = functools.partial(jax.lax.slice_in_dim, a, axis=axis)
    a_grad = jax.numpy.concatenate((
        # 3th order edge
        ((-11 / 6) * sliced(0, 2) + 3 * sliced(1, 3) - (3 / 2) * sliced(2, 4) + (1 / 3) * sliced(3, 5)),
        # 4th order inner
        ((1 / 12) * sliced(None, -4) - (2 / 3) * sliced(1, -3) + (2 / 3) * sliced(3, -1) - (1 / 12) * sliced(4, None)),
        # 3th order edge
        ((-1 / 3) * sliced(-5, -3) + (3 / 2) * sliced(-4, -2) - 3 * sliced(-3, -1) + (11 / 6) * sliced(-2, None))
    ), axis)
    return a_grad


@jax.jit
def stimulus_at_t(t, stimulus):
    return stimulus.mod * np.equal(stimulus.t, t)


@jax.jit
def neumann(X):
    X = jax.ops.index_update(X, jax.ops.index[0], X[1])
    X = jax.ops.index_update(X, jax.ops.index[-1], X[-2])
    X = jax.ops.index_update(X, jax.ops.index[..., 0], X[..., 1])
    X = jax.ops.index_update(X, jax.ops.index[..., -1], X[..., -2])
    return X
