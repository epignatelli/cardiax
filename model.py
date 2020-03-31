import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import functools

@functools.partial(jax.jit, static_argnums=0)
def init(shape):
    v = np.ones(shape) * 0.99
    w = np.ones(shape) * 0.99
    u = np.zeros(shape)
    state = (v, w, u)   
    return state


@jax.jit
def step(state, t, params, D, stimuli, dt):
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

    # voltage
    u_x, u_y = np.gradient(u)
    u_xx = np.gradient(u_x, axis=0)
    u_yy = np.gradient(u_y, axis=1)
    D_x, D_y = np.gradient(D)
    d_u = 4 * D * (u_xx + u_yy) + ((D_x * u_x) + (D_y * u_y)) + I_ion
    return euler((v, w, u), (d_v, d_w, d_u), dt)


@jax.jit
def euler(state, grad, dt):
    v, w, u = state
    d_v, d_w, d_u = grad

    # explicit euler update
    v += d_v * dt
    w += d_w * dt
    u += d_u * dt
    return (v, w, u)


@jax.jit
def neumann(X):
    X = jax.ops.index_update(X, [0], X[1])
    X = jax.ops.index_update(X, [-1], X[-2])
    X = jax.ops.index_update(X, [..., 0], X[..., 1])
    X = jax.ops.index_update(X, [..., -1], X[..., -2])
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


@functools.partial(jax.jit, static_argnums=0)
def _forward(shape, n_iter, params, diffusion, stimuli, dt):
    # iterate
    state = init(shape)
    state = jax.lax.fori_loop(0, n_iter, lambda i, state: step(state, i * dt, params, diffusion, stimuli, dt), state)
    return state


def _forward_by_step(state, *, n_iter, params, D, stimuli, dt, log_at=10):
    for t in np.arange(0, n_iter, log_at):
        state = jax.lax.fori_loop(t, t + log_at, lambda i, state: step(state, i * dt, params, D, stimuli, dt), state)
        print("t: %s" % (t + log_at))
        show(state)
        
    # check if there is a leftover from the for loop
    if not n_iter % log_at:
        state = jax.lax.fori_loop(t, n_iter % log_at, lambda i, state: step(state, i * dt, params, D, stimuli, dt), state)
        print("t: %s" % t)
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


def forward(tissue_size=None,
            field_size=None,
            cell_parameters=None,
            diffusion=None,
            stimuli=[],
            dt=0.01,
            dx=0.025,
            end_time=1000,
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
    
    if log_at is None:
        state = _forward(field_size,
                        n_iter,
                        cell_parameters,
                        diffusion, 
                        stimuli,
                        dt)
    else:
        state = _forward_by_step(field_size,
                               n_iter,
                               cell_parameters,
                               diffusion,
                               stimuli,
                               dt,
                               log_at)
    state[0].block_until_ready()
    return state
