from typing import NamedTuple
import functools
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
from . import plot


class State(NamedTuple):
    v: np.ndarray
    w: np.ndarray
    u: np.ndarray
    l: np.ndarray
    j: np.ndarray


def forward(shape,
            checkpoints,
            cell_parameters,
            diffusivity,
            stimuli,
            dt, dx):
    """
    Solves the Fenton-Karma model using second order finite difference and explicit euler update scheme.
    Neumann conditions are considered at the boundaries.
    Units are adimensional.
    Args:
        shape (Tuple[int, int]): The shape of the finite difference grid
        checkpoints (iter): An iterable that contains time steps in simulation units, at which pause, and display the state of the system
        cell_parameters (Dict[string, float]): Dictionary of physiological parameters as illustrated in Fenton, Cherry, 2002.
        diffusivity (float): Diffusivity of the cardiac tissue
        stimuli (List[Dict[string, object]]): A list of stimuli to provide energy to the tissue
        dt (float): time infinitesimal to use in the euler stepping scheme
        dx (float): space infinitesimal to use in the spatial gradient calculation
    Returns:
        (List[jax.numpy.ndarray]): The list of states at each checkpoint
    """
    state = init(shape)
    states = []
    for i in range(len(checkpoints) - 1):
        print("Solving at: %dms/%dms\t\t with %d passages" % (checkpoints[i + 1] * dt, checkpoints[-1] * dt, checkpoints[i + 1] - checkpoints[i]), end="\r")
        state = _forward(state, checkpoints[i], checkpoints[i + 1], cell_parameters, np.ones(shape) * diffusivity, stimuli, dt, dx)
        plot.plot_state(state)
        plt.show()
        states.append(state)
    return states


@functools.partial(jax.jit, static_argnums=0)
def init(shape):
    v = np.ones(shape)
    w = np.ones(shape)
    u = np.zeros(shape)
    l = np.zeros(shape)
    j = np.zeros(shape)
    return State(v, w, u, l, j)


# @functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
@jax.jit
def _forward(state, t, t_end, params, D, stimuli, dt, dx):
    # iterate
    state = jax.lax.fori_loop(t, t_end, lambda i, state: step(state, i, params, D, stimuli, dt, dx), init_val=state)
    return state


# @functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6))
@jax.jit
def step(state, t, params, D, stimuli, dt, dx):
    # apply stimulus
    u_stimulated = stimulate(t, state.u, stimuli)

    # neumann boundary conditions
    v = np.pad(state.v, 1, mode="edge")
    w = np.pad(state.w, 1, mode="edge")
    u = np.pad(u_stimulated, 1, mode="edge")
    D = np.pad(D, 1, mode="edge")

    # reaction term
    p = np.greater_equal(u, params.V_c)
    q = np.greater_equal(u, params.V_v)
    tau_v_minus = (1 - q) * params.tau_v1_minus + q * params.tau_v2_minus

    j_fi = - v * p * (u - params.V_c) * (1 - u) / params.tau_d
    j_so = (u * (1 - p) / params.tau_0) + (p / params.tau_r)
    j_si = - (w * (1 + np.tanh(params.k * (u - params.V_csi)))) / (2 * params.tau_si)
    j_ion = -(j_fi + j_so + j_si) / params.Cm

    # diffusion term
    u_x = gradient(u, 0) / dx
    u_y = gradient(u, 1) / dx
    u_xx = gradient(u_x, 0) / dx
    u_yy = gradient(u_y, 1) / dx
    D_x = gradient(D, 0) / dx
    D_y = gradient(D, 1) / dx
    del_u = D * (u_xx + u_yy) + (D_x * u_x) + (D_y * u_y)

    d_v = ((1 - p) * (1 - v) / tau_v_minus) - ((p * v) / params.tau_v_plus)
    d_w = ((1 - p) * (1 - w) / params.tau_w_minus) - ((p * w) / params.tau_w_plus)
    d_u = del_u + j_ion

    # euler update
    v = state.v + d_v[1:-1, 1:-1] * dt
    w = state.w + d_w[1:-1, 1:-1] * dt
    u = u_stimulated + d_u[1:-1, 1:-1] * dt
    return State(v, w, u, del_u[1:-1, 1:-1], j_ion[1:-1, 1:-1])


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


@jax.jit
def neumann(X):
    X = np.pad(X, 1, mode="edge")
    return X


@jax.jit
def stimulate(t, X, stimuli):
    stimulated = np.zeros_like(X)
    for stimulus in stimuli:
        active = np.greater_equal(t, stimulus.protocol.start)
        active &= (np.mod(stimulus.protocol.start - t + 1, stimulus.protocol.period) < stimulus.protocol.duration)
        stimulated = np.where(stimulus.field * (active), stimulus.field, stimulated)
    return np.where(stimulated != 0, stimulated, X)


@functools.partial(jax.jit, static_argnums=(1, 2))
def _forward_stack(state, t, t_end, params, diffusion, stimuli, dt, dx):
    # iterate
    def _step(state, i):
        new_state = step(state, i, params, diffusion, stimuli, dt, dx)
        return (new_state, new_state)
    xs = np.arange(t, t_end)
    last_state, states = jax.lax.scan(_step, state, xs)
    return states
