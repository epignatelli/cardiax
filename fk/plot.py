import jax
import jax.numpy as np
import numpy as onp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import IPython
from IPython.display import HTML
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


def show3d(state, rcount=200, ccount=200, zlim=None, figsize=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    r = list(range(0, len(state)))
    x, y = np.meshgrid(r, r)
    plot = ax.plot_surface(x, y, state, rcount=rcount, ccount=ccount, cmap="magma")
    cbar = fig.colorbar(plot)
    cbar.set_label("mV", rotation=0)
    if zlim is not None:
        ax.set_zlim3d(zlim[0], zlim[1])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("Voltage [mV]")
    ax.set_xticks([7, 7])
    fig.tight_layout()
    return fig, ax


def animate(states, times=None, figsize=None, channel=None, vmin=0, vmax=1):
    backend = matplotlib.get_backend()
    matplotlib.use("nbAgg")
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    def init():
        im = ax.imshow(states[0, channel].squeeze(), animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im)
        cbar.set_label("mV", rotation=0)
        return [im]

    def update(iteration):
        print("Rendering {}\t".format(iteration + 1), end="\r")
        im = ax.imshow(states[iteration, channel].squeeze(), animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        if times is not None:
            ax.set_title("t: %d" % times[iteration])
        return [ax]

    animation = FuncAnimation(fig, update, frames=range(len(states)), init_func=init, blit=True)
    matplotlib.use(backend)
    return HTML(animation.to_html5_video())


def animate3d(states, times=None, figsize=None):
    backend = matplotlib.get_backend()
    matplotlib.use("nbAgg")
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    def init():
        im = show3d(states[0, channel].squeeze(), animated=True, cmap="magma", vmin=0, vmax=1)
        cbar = fig.colorbar(im)
        cbar.set_label("mV", rotation=0)
        return [im]

    def update(iteration):
        print("Rendering {}\t".format(iteration + 1), end="\r")
        im = ax.imshow(states[iteration, channel].squeeze(), animated=True, cmap="magma", vmin=0, vmax=1)
        if times is not None:
            ax.set_title("t: %d" % times[iteration])
        return [ax]

    animation = FuncAnimation(fig, update, frames=range(len(states)), init_func=init, blit=True)
    matplotlib.use(backend)
    return HTML(animation.to_html5_video())