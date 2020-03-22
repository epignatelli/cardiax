import jax
import jax.numpy as np
import matplotlib.pyplot as plt


def init(width, height):
    shape = (width, height)
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
        active &= (np.mod(t - stimulus["start"], stimulus["period"]) < stimulus["duration"])  # cyclic
        X = np.where(stimulus["field"] * (active), stimulus["field"], X)
    return X


@jax.jit
def forward(length, params, D, stimuli, dt, log_at=10):
    # iterate
    state = init(128, 128)
    state = jax.lax.fori_loop(0, length, lambda i, state: step(state, i * dt, params, D, stimuli, dt), state)
    return state


def show(state, **kwargs):
    fig, ax = plt.subplots(1, 3, figsize=(kwargs.pop("figsize", None) or (10, 3)))
    im = ax[0].imshow(state[0], **kwargs)
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title("v")
    im = ax[1].imshow(state[1], **kwargs)
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("w")
    im = ax[2].imshow(state[2], **kwargs)
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title("u")
    plt.show()
    return
