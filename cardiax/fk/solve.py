import functools
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ..ode import plot
from ..ode.gradients import fd
from ..ode.stimulus import Stimulus
from .params import Params


class State(NamedTuple):
    v: jnp.ndarray
    w: jnp.ndarray
    u: jnp.ndarray


def init(shape):
    v = jnp.ones(shape)
    w = jnp.ones(shape)
    u = jnp.zeros(shape)
    return State(v, w, u)


def step(
    state: State,
    time: float,
    boundaries_conditions: Callable,
    params: Params,
    diffusivity: jnp.ndarray,
    stimuli: Sequence[Stimulus],
    dx: float,
):
    """
    Solves the gradients of the fenton karma equation
    """
    # apply stimulus
    state = state._replace(u=stimulate(time, state.u, stimuli))

    # neumann boundary conditions
    state = boundaries_conditions.apply(state)
    diffusivity = boundaries_conditions.apply(diffusivity)
    v, w, u = state

    # reaction term
    p = jnp.greater_equal(u, params.V_c)
    q = jnp.greater_equal(u, params.V_v)
    tau_v_minus = (1 - q) * params.tau_v1_minus + q * params.tau_v2_minus

    j_fi = -v * p * (u - params.V_c) * (1 - u) / params.tau_d
    j_so = (u * (1 - p) / params.tau_0) + (p / params.tau_r)
    j_si = -(w * (1 + jnp.tanh(params.k * (u - params.V_csi)))) / (2 * params.tau_si)
    j_ion = -(j_fi + j_so + j_si) / params.Cm

    # diffusion term
    u_x = fd(u, 0) / dx
    u_y = fd(u, 1) / dx
    u_xx = fd(u_x, 0) / dx
    u_yy = fd(u_y, 1) / dx
    D_x = fd(diffusivity, 0) / dx
    D_y = fd(diffusivity, 1) / dx
    del_u = diffusivity * (u_xx + u_yy) + (D_x * u_x) + (D_y * u_y)

    d_v = ((1 - p) * (1 - v) / tau_v_minus) - ((p * v) / params.tau_v_plus)
    d_w = ((1 - p) * (1 - w) / params.tau_w_minus) - ((p * w) / params.tau_w_plus)
    d_u = del_u + j_ion

    # restore from boundary manipultions
    grads = (d_v, d_w, d_u)
    grads = boundaries_conditions.restore(grads)
    return grads


def stimulate(time, X, stimuli):
    stimulated = jnp.zeros_like(X)
    for stimulus in stimuli:
        active = jnp.greater_equal(time, stimulus.protocol.start)
        active &= (
            jnp.mod(stimulus.protocol.start - time + 1, stimulus.protocol.period)
            < stimulus.protocol.duration
        )
        stimulated = jnp.where(stimulus.field * (active), stimulus.field, stimulated)
    return jnp.where(stimulated != 0, stimulated, X)
