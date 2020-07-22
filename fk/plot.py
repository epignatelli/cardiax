import jax
import jax.numpy as np
import numpy as onp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
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
    # plot v
    im = ax[0].imshow(state[0], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title("v")
    # plot w
    im = ax[1].imshow(state[1], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("w")
    # plot u
    im = ax[2].imshow(state[2], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title("u")
    # set colorbar
#     cbar = fig.colorbar(im)
#     cbar.ax.set_title("mV")
    # format axes
    for i in range(len(ax)):
        ax[i].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax[i].set_xlabel("x [cm]")
        ax[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax[i].set_ylabel("y [cm]")
    fig.tight_layout()
    plt.show()
    return fig, ax


def show_grid(states, times=[], figsize=None, rows=5, font_size=10):
    cols = math.ceil(len(states) / rows)
    rows = max(2, min(rows, len(states)))
    fig, ax = plt.subplots(cols, rows, figsize=figsize)
    ax = ax.flatten()
    idx = 0
    
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
    
    for idx in range(len(states)):
        im = ax[idx].imshow(states[idx], cmap="magma", vmin=-85, vmax=15,)
        ax[idx].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax[idx].set_xlabel("x [cm]")
        ax[idx].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax[idx].set_ylabel("y [cm]")
        cbar = fig.colorbar(im, ax=ax[idx])
        cbar.ax.set_title("mV") 
        if idx + 1 < len(times):
            ax[idx].set_title("t: {:d}".format(times[idx]))
    fig.tight_layout()
    return fig, ax


def show3d(state, rcount=200, ccount=200, zlim=None, figsize=None):
    # setup figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    # make surface plot
    r = list(range(0, len(state)))
    x, y = np.meshgrid(r, r)
    plot = ax.plot_surface(x, y, state, rcount=rcount, ccount=ccount, cmap="magma")
    # add colorbar
    cbar = fig.colorbar(plot)
    cbar.ax.set_title("mV")
    if zlim is not None:
        ax.set_zlim3d(zlim[0], zlim[1])
    # format axes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
    ax.set_xlabel("x [cm]")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("Voltage [mV]")
    # crop image
    fig.tight_layout()
    return fig, ax


def animate(states, times=None, figsize=None, channel=None, vmin=0, vmax=1):
    backend = matplotlib.get_backend()
    matplotlib.use("nbAgg")
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    def init():
        im = ax.imshow(states[0, channel].squeeze(), animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax.set_xlabel("x [cm]")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax.set_ylabel("y [cm]")
        cbar = fig.colorbar(im)
        cbar.ax.set_title("mV")
        fig.tight_layout()
        return [im]

    def update(iteration):
        print("Rendering {}\t".format(iteration + 1), end="\r")
        im = ax.imshow(states[iteration, channel].squeeze(), animated=True, cmap="magma", vmin=vmin, vmax=vmax)
        if times is not None:
            ax.set_title("t: %d" % times[iteration])
        return [ax]

    animation = FuncAnimation(fig, update, frames=range(len(states)), init_func=init, blit=True)
    matplotlib.use(backend)
    return animation


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
    return animation