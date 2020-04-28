import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import functools
import params
import convert


@functools.partial(jax.jit, static_argnums=0)
def init(shape):
    v = np.ones(shape)
    w = np.ones(shape)
    u = np.zeros(shape)
    state = (v, w, u)
    return np.asarray(state)


@jax.jit
def step(state, t, params, D, stimuli, dt, dx):
    v, w, u = state

    # apply stimulus
    u = stimulate(t, u, stimuli)

    # apply boundary conditions
    v = neumann(v)
    w = neumann(w)
    u = neumann(u)

    # gate variables
    p = np.greater_equal(u, params["V_c"])
    q = np.greater_equal(u, params["V_v"])
    tau_v_minus = (1 - q) * params["tau_v1_minus"] + q * params["tau_v2_minus"]

    d_v = ((1 - p) * (1 - v) / tau_v_minus) - ((p * v) / params["tau_v_plus"])
    d_w = ((1 - p) * (1 - w) / params["tau_w_minus"]) - ((p * w) / params["tau_w_plus"])

    # currents
    J_fi = - v * p * (u - params["V_c"]) * (1 - u) / params["tau_d"]
    J_so = (u * (1 - p) / params["tau_0"]) + (p / params["tau_r"])
    J_si = - (w * (1 + np.tanh(params["k"] * (u - params["V_csi"])))) / (2 * params["tau_si"])

    I_ion = -(J_fi + J_so + J_si) / params["Cm"]

    # voltage01
    u_x, u_y = np.gradient(u)
    u_x /= dx
    u_y /= dx
    u_xx = np.gradient(u_x, axis=0)
    u_yy = np.gradient(u_y, axis=1)
    u_xx /= dx
    u_yy /= dx
    D_x, D_y = np.gradient(D)
    D_x /= dx
    D_y /= dx
    d_u = D * (u_xx + u_yy) + (D_x * u_x + D_y * u_y) + I_ion

    # euler update
    v += d_v * dt
    w += d_w * dt
    u += d_u * dt
    return np.asarray((v, w, u))


@jax.jit
def neumann(X):
    X = jax.ops.index_update(X, jax.ops.index[0], X[1])
    X = jax.ops.index_update(X, jax.ops.index[-1], X[-2])
    X = jax.ops.index_update(X, jax.ops.index[..., 0], X[..., 1])
    X = jax.ops.index_update(X, jax.ops.index[..., -1], X[..., -2])
    return X


@jax.jit
def stimulate(t, X, stimuli):
    for stimulus in stimuli:
        active = t > stimulus["start"]
        active &= t < stimulus["start"] + stimulus["duration"]
        # for some weird reason checks for cyclic stimuli does not work
        active = (np.mod(t - stimulus["start"], stimulus["period"]) < stimulus["duration"])  # cyclic
        X = np.where(stimulus["field"] * (active), stimulus["field"], X)
    return X


@jax.jit
def _forward(state, t, t_end, params, diffusion, stimuli, dt, dx):
    # iterate
    state = jax.lax.fori_loop(t, t_end, lambda i, state: step(state, i, params, diffusion, stimuli, dt, dx), state)
    return state


def _forward_and_stack(state, t, t_end, params, diffusion, stimuli, dt, dx):
    # iterate
#     states = np.empty((t_end - t, *(state.shape)))
    def _step(state, i):
        new_state = step(state, i, params, diffusion, stimuli, dt, dx)
        return (new_state, new_state)
    xs = np.arange(t, t_end)
#     print(xs)
    last_state, states = jax.lax.scan(_step, state, xs)
    return states


def show(state, **kwargs):
    fig, ax = plt.subplots(1, 3, figsize=(kwargs.pop("figsize", None) or (10, 3)))
    vmin = kwargs.pop("vmin", -1)
    vmax = kwargs.pop("vmax", 1)
    cmap = kwargs.pop("cmap", "RdBu")
    im = ax[0].imshow(state[0], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title("v")
    im = ax[1].imshow(state[1], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("w")
    im = ax[2].imshow(state[2], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title("u")
    plt.show()
    return fig, ax


def show_stimuli(*stimuli, **kwargs):
    fig, ax = plt.subplots(1, len(stimuli), figsize=(kwargs.pop("figsize", None) or (10, 3)))
    vmin = kwargs.pop("vmin", -1)
    vmax = kwargs.pop("vmax", 1)
    cmap = kwargs.pop("cmap", "RdBu")
    for i, stimulus in enumerate(stimuli):
        im = ax[i].imshow(stimulus["field"], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title("Stimulus %d" % i)
    plt.show()
    return


def forward(tissue_size=None,
            cell_parameters=None,
            diffusion=0.001,
            stimuli=[],
            dt=0.01,
            dx=0.025,
            end_time=100,
            checkpoints=None):
    if tissue_size is None:
        tissue_size = (12, 12)
    shape = convert.field_to_shape(tissue_size, dx)
    
    n_iter = convert.ms_to_units(end_time, dt)

    if cell_parameters is None:
        cell_parameters = params.params_test()

    diffusion = np.ones(shape) * diffusion
    
    if checkpoints is None:
        checkpoints = [0, n_iter]
    elif isinstance(checkpoints, list):
        checkpoints = [convert.ms_to_units(ck, dt) for ck in checkpoints]

    print("Starting simulation with %s dof for %dms (%d iterations with dt %4f)" % (tissue_size, end_time, n_iter, dt) )
    print("Checkpointing at", checkpoints)

    state = init(shape)
    for i in range(len(checkpoints) - 1):
        state = _forward(state, checkpoints[i], checkpoints[i + 1], cell_parameters, diffusion, stimuli, dt, dx)  # dt = 10000
        print(checkpoints[i + 1])
        show(state)
    return state
