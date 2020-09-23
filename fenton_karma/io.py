import h5py
import os
import jax.numpy as np
from PIL import Image


def init(path, shape, n_iter, n_stimuli):
    hdf5 = h5py.File(path, "w")
    if "states" not in hdf5:
        # shape is (t, 3, w, h), where 3 is the tree fk variable
        dset_states = hdf5.create_dataset("states", shape=(n_iter, 3, *shape), dtype="float32")
    if "stimuli" not in hdf5:
        dset_stim = hdf5.create_dataset("stimuli", shape=(n_stimuli, *shape), dtype="float32")
    return hdf5


def add_params(hdf5, params, diffusivity, dt, dx, shape=None):
    #reshape
    if shape is not None:
        diffusivity = imresize(diffusivity, shape)
    # store
    hdf5.create_dataset("params/D", data=diffusivity)
    hdf5.create_dataset("params/dt", data=dt)
    hdf5.create_dataset("params/dx", data=dx)
    for key in params:
        hdf5.create_dataset("params/" + key, data=params[key])
    return True


def add_stimuli(hdf5, stimuli, shape=None):
    # reshape
    if shape is not None:
        fields = [imresize(stimuli[i]["field"], shape) for i in range(len(stimuli))]
    else:
        fields = [stimuli[i]["field"] for i in range(len(stimuli))]
    hdf5.create_dataset("field", data=fields)
    # store
    hdf5.create_dataset("start", data=[stimuli[i]["start"] for i in range(len(stimuli))])
    hdf5.create_dataset("duration", data=[stimuli[i]["duration"] for i in range(len(stimuli))])
    hdf5.create_dataset("period", data=[stimuli[i]["period"] for i in range(len(stimuli))])
    return True
        

def add_state(dset, state, t, shape=None):
    if shape is not None:
        state = imresize(state, shape)
    dset[t] = state
    return True

def append_states(dset, states, start, end):
    # shape is (t, 3, w, h), where 3 is the tree fk variable
    dset[start:end] = states
    return True
        
    
def load(path, dataset, start, end, step=None):
    with h5py.File(path, "r") as file:
        return [file[dset][start:end:step] for dset in file]
    
    
def load_state(dset, start, end, step):
    return dset[start:end:step]
    
    
def load_stimuli(file):
    stimuli = []
    for i in range(len(file["field"])):
        s = {}
        s["field"] = file["field"][i]
        s["start"] = file["start"][i]
        s["duration"] = file["duration"][i]
        s["period"] = file["period"][i]
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
    return params, D
    

def imresize(a, size):
    return torch.nn.functional.interpolate(torch.tensor(a).unsqueeze(0), size=size, mode="bilinear").squeeze().numpy()
    
