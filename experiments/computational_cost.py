import jax
import jax.numpy as jnp
from jax.experimental import optimizers
import os
import h5py
import numpy as onp
import time
import cardiax
import deepx
from deepx import optimise
import helx
from helx.optimise import Optimiser
import json
import wandb
import pickle
import IPython
from IPython.display import display
from IPython.display import display_javascript
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
import helx
from matplotlib import rc
from functools import partial
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import timeit

import matplotlib.patches as mpatches

rc("animation", html="jshtml")
rc("text", usetex=False)
# rc['animation.embed_limit'] = 2**128
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


FILENAME = "/home/epignatelli/repos/cardiax/experiments/data/spiral.hdf5"
N_REFEED = 2
URL = "p3aetobr/10351"  # TOP
N_RUNS = 10


def take(filename, t, length, normalise=False):
    with h5py.File(filename, "r") as sequence:
        states = onp.array(sequence["states"][t : t + (2 + length)])
        diffusivity = onp.array(sequence["diffusivity"])
        if normalise:
            diffusivity = diffusivity * 500
        diffusivity = onp.tile(diffusivity[None, None], (2, 1, 1, 1))
        xs = onp.concatenate([states[:2], diffusivity], axis=-3)
        ys = states[2:]
        xs = xs[None, None]
        ys = ys[None, None]
        print(xs.shape, ys.shape)
        return xs, ys


def load_model(url):
    train_state = deepx.optimise.TrainState.restore(url)
    hparams = train_state.hparams
    model = deepx.resnet.ResNet(hparams.hidden_channels, 1, hparams.depth)
    opt = optimizers.adam(0.001)
    params = opt.params_fn(train_state.opt_state)
    return model, hparams, params


def init(shape):
    xs, ys = take(FILENAME, 100, N_REFEED, True)
    xs = cardiax.io.imresize(xs, shape)
    print(xs.shape)
    return xs


def preprocess_cardiax(xs):
    d = xs[:, :, :, -1:]
    xs = xs[:, :, :, :-1]
    state = xs.squeeze()[-1]
    state = cardiax.solve.State(*state)
    diffusivity = d.squeeze()[-1]
    return state, diffusivity


def preprocess_deepx(xs):
    return xs


def speed_test(shape):

    #  init
    print("Initialising states...")
    xs = init(shape)
    xs_cardiax, diffusivity = preprocess_cardiax(xs)
    xs_deepx = preprocess_deepx(xs)

    print("Setting up times...")
    ts = jnp.arange(0, 10_000, 500)
    dt = 0.01
    dx = 0.01
    paramset = cardiax.params.PARAMSET_3
    cardiax_forward = lambda: cardiax.solve.forward(
        xs_cardiax, ts, paramset, diffusivity, [], dx, dt, plot_while=False
    )

    print("Loading model...")
    model, hparams, params = load_model(URL)
    deepx_forward = lambda: deepx.optimise.infer(model, len(ts) - 1, params, xs_deepx)

    #  warm up
    print("Warming up...")
    cardiax_forward()
    deepx_forward()

    #  measure
    print("Testing...")
    cardiax_times = timeit.repeat(cardiax_forward, repeat=N_RUNS, number=1)
    deepx_times = timeit.repeat(deepx_forward, repeat=N_RUNS, number=1)
    return cardiax_times, deepx_times


if __name__ == "__main__":
    shapes = [
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]
    folder = "/home/epignatelli/repos/cardiax/experiments/results/performance"
    os.makedirs(folder, exist_ok=True)

    c_times = {}
    d_times = {}
    with open(os.path.join(folder, "cardiax-results.pickle"), "wb") as cf:
        with open(os.path.join(folder, "deepx-results.pickle"), "wb") as df:
            # headers
            for shape in shapes:
                print("Testing for shape ", shape)
                cardiax_times, deepx_times = speed_test(shape)
                c_times[shape] = cardiax_times
                d_times[shape] = deepx_times
            pickle.dump(c_times, cf)
            pickle.dump(d_times, df)
