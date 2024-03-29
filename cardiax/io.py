import json
import os

import h5py
import jax
import jax.numpy as jnp
from skimage.io import imsave

from .params import Params
from .stimulus import Protocol, Stimulus


def init(path, shape, n_iter, n_stimuli, n_variables=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdf5 = h5py.File(path, "w")
    if "states" not in hdf5:
        # shape is (t, n_variables, w, h)
        hdf5.create_dataset(
            "states", shape=(n_iter, n_variables, *shape), dtype="float32"
        )
    if "stimuli" not in hdf5:
        hdf5.create_dataset("stimuli", shape=(n_stimuli, *shape), dtype="float32")
    return hdf5


def add_params(hdf5, params, diffusivity, dt, dx, shape=None):
    # reshape
    if shape is not None:
        diffusivity = imresize(diffusivity, shape)
    # store
    hdf5.create_dataset("params/D", data=diffusivity)
    hdf5.create_dataset("params/dt", data=dt)
    hdf5.create_dataset("params/dx", data=dx)
    for i in range(len(params)):
        hdf5.create_dataset("params/" + params._fields[i], data=params[i])
    return True


def add_diffusivity(hdf5, diffusivity, shape=None):
    # reshape
    if shape is not None:
        diffusivity = imresize(diffusivity, shape)
    hdf5.create_dataset("diffusivity", data=diffusivity)
    return True


def add_stimuli(hdf5, stimuli, shape=None):
    # reshape
    if shape is not None:
        fields = [imresize(stimuli[i].field, shape) for i in range(len(stimuli))]
    else:
        fields = [stimuli[i].field for i in range(len(stimuli))]
    hdf5.create_dataset("field", data=fields)
    # store
    hdf5.create_dataset(
        "start", data=[stimuli[i].protocol.start for i in range(len(stimuli))]
    )
    hdf5.create_dataset(
        "duration", data=[stimuli[i].protocol.duration for i in range(len(stimuli))]
    )
    hdf5.create_dataset(
        "period", data=[stimuli[i].protocol.period for i in range(len(stimuli))]
    )
    return True


def add_state(dset, state, t, shape=None):
    if shape is not None:
        array = jnp.array(tuple(state))
        state = imresize(array, shape)
    dset[t] = state
    return True


def add_states(dset, states, start, end):
    # shape is (t, 3, w, h), where 3 is the tree fk variable
    dset[start:end] = states
    return True


def load(path, start=None, end=None, step=None):
    with h5py.File(path, "r") as file:
        return [file[dset][start:end:step] for dset in file]


def load_state(dset, start, end, step):
    return dset[start:end:step]


def load_stimuli(file):
    stimuli = []
    for i in range(len(file["field"])):
        protocol = Protocol(file["start"][i], file["duration"][i], file["period"][i])
        s = Stimulus(protocol, file["field"][i])
        stimuli.append(s)
    return stimuli


def load_params(filepath):
    params = {}
    D = None
    with h5py.File(filepath, "r") as file:
        stored_params = file["params"]
        for key in stored_params:
            if key == "D":
                D = stored_params[key][...]
            else:
                params[key] = stored_params[key][...]
    params = Params(*list(params.values()))
    return params, D


def load_diffusivity(filepath):
    with h5py.File(filepath, "r") as f:
        return f["diffusivity"][:]


def imresize(a, size, method="bilinear"):
    """
    Args:
        a (jnp.ndarray): 2D or 3D array
    """
    out_shape = a.shape[:-2] + size
    return jax.image.resize(a, out_shape, method)
