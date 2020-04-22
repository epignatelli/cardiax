import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import math
from . import convert


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


def show(state, **kwargs):
    fig, ax = plt.subplots(1, 3, figsize=(kwargs.pop("figsize", None) or (15, 5)))
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


def show_grid(states, times=[], figsize=None, rows=5):
    cols = math.ceil(len(states) / rows)
    fig, ax = plt.subplots(cols, rows, figsize=figsize)
    idx = 0
    while idx < len(states):
        for col in range(cols):
            for row in range(rows):
                ax[col, row].imshow(states[idx], cmap="magma", vmin=0, vmax=1,)
                if idx + 1 < len(times):
                    iteration = convert.ms_to_units(times[idx + 1], dt)
                    ax[col, row].set_title("Iter: " + str(iteration) + " (%.3fms)" % times[idx + 1])
                idx += 1
    return fig, ax