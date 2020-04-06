import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import functools
import params


@functools.partial(jax.jit, static_argnums=0)
def init(shape):
    v = np.ones(shape)
    w = np.ones(shape)
    u = np.zeros(shape)
    state = (v, w, u)
    return state


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
    u_x, u_y = np.gradient(u, dx)
    u_xx = np.gradient(u_x, dx, axis=0)
    u_yy = np.gradient(u_y, dx, axis=1)
    D_x, D_y = np.gradient(D, dx)
    d_u = D * (u_xx + u_yy) + (D_x * u_x + D_y * u_y) + I_ion

    # euler update
    v += d_v * dt
    w += d_w * dt
    u += d_u * dt
    return (v, w, u)


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
        # active = (np.mod(t - stimulus["start"], stimulus["period"]) < stimulus["duration"])  # cyclic
        X = np.where(stimulus["field"] * (active), stimulus["field"], X)
    return X


@jax.jit
def _forward(state, t, t_end, params, diffusion, stimuli, dt, dx):
    # iterate
    state = jax.lax.fori_loop(t, t_end, lambda i, state: step(state, i * dt, params, diffusion, stimuli, dt, dx), state)
    return state


def _forward_by_step(shape, n_iter, params, D, stimuli, dt, dx, log_at=10):
    state = init(shape)
    for t in np.arange(0, n_iter, log_at):
        state = jax.lax.fori_loop(t, t + log_at, lambda i, state: step(state, i * dt, params, D, stimuli, dt, dx), state)
        print("t: %s" % ((t + log_at) * dt))
        show(state)
    return state


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
    return


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
            field_size=None,
            cell_parameters=None,
            diffusion=None,
            stimuli=[],
            dt=0.01,
            dx=0.025,
            end_time=100,
            log_at=None):
    if field_size is None and tissue_size is not None:
        field_size = (int(tissue_size[0] / dx), int(tissue_size[1] / dx))

    n_iter = int(end_time / dt)

    if cell_parameters is None:
        cell_parameters = params.params_test()

    if diffusion is None:
        diffusion = np.ones(field_size) * 0.05
    elif isinstance(diffusion, float) or isinstance(diffusion, int):
        diffusion = np.ones(field_size) * diffusion

    assert diffusion.shape == field_size

    print("Starting simulation with %s dof for %dms (%d iterations with dt %4f)" % (field_size, end_time, n_iter, dt) )

    if log_at is None:
        state = _forward(field_size,
                         n_iter,
                         cell_parameters,
                         diffusion,
                         stimuli,
                         dt,
                         dx)
    else:
        state = _forward_by_step(field_size,
                                 n_iter,
                                 cell_parameters,
                                 diffusion,
                                 stimuli,
                                 dt,
                                 dx,
                                 log_at)
    return state
