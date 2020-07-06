import jax
import jax.numpy as np


def params_to_units(params, dx, dt):
    params["Cm"] /= (dx ** 2) / dt
    params["tau_d"] /= dt
    params["tau_v1_minus"] /= dt
    params["tau_v2_minus"] /= dt
    params["tau_v_plus"] /= dt
    params["tau_0"] /= dt
    params["tau_r"] /= dt
    params["tau_si"] /= dt
    params["tau_w_minus"] /= dt
    params["tau_w_plus"] /= dt
    return params


def diffusivity_to_units(d, dt):
    return d / dt


def realsize_to_shape(field, dx):
    return (int(field[0] / dx), int(field[1] / dx))


def shape_to_realsize(field, dx):
    return (int(field[0] * dx), int(field[1] * dx))


def cm_to_units(value, dx):
    return int(value / dx)


def units_to_cm(value, dx):
    return value * dx


def ms_to_units(value, dt):
    return int(value / dt)

    
def units_to_ms(value, dt):
    return value * dt


def stimuli_to_units(stimuli, dx, dt):
    stimuli = list(stimuli)
    for i in range(len(stimuli)):
        stimuli[i]["start"] = ms_to_units(stimuli[i]["start"], dt)
        stimuli[i]["duration"] = ms_to_units(stimuli[i]["duration"], dt)
        stimuli[i]["period"] = ms_to_units(stimuli[i]["period"], dt)
    return stimuli


def u_to_V(u, V0=-85, Vfi=15):
    return ((Vfi - V0) * u) + V0


def V_to_u(V, V0=-85, Vfi=15):
    return (V - V0) / (Vfi - V0)
    