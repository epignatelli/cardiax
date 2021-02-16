import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math


def plot_state(state, **kwargs):
    fig, ax = plt.subplots(1, len(state), figsize=(kwargs.pop("figsize", None) or (15, 5)))
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    cmap = kwargs.pop("cmap", "RdBu")
    titles = kwargs.pop("names", ["Variable {}".format(i) for i in range(len(ax))])

    for i in range(len(ax)):
        im = ax[i].imshow(state[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title(titles[i])
        ax[i].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax[i].set_xlabel("x [cm]")
        ax[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
        ax[i].set_ylabel("y [cm]")
    return fig, ax


def plot_variable_3d(variable, **kwargs):
    # setup figure
    fig = plt.figure(figsize=(kwargs.pop("figsize", None) or (15, 5)))
    ax = fig.add_subplot(projection="3d")

    # get function args
    zlim = kwargs.pop("zlim", None)
    rcount = kwargs.pop("rcount", 200)
    ccount = kwargs.pop("ccount", 200)

    # make surface plot
    r = list(range(0, len(variable)))
    x, y = np.meshgrid(r, r)
    plot = ax.plot_surface(x, y, variable, rcount=rcount, ccount=ccount, cmap="magma")

    # add colorbar
    cbar = fig.colorbar(plot)
    cbar.ax.set_title("mV")

    # format axes
    if zlim is not None:
        ax.set_zlim3d(zlim[0], zlim[1])
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


def plot_time_variable(states, times=[], figsize=None, rows=5, font_size=10, vmin=-85, vmax=15, cmap="magma"):
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
        im = ax[idx].imshow(states[idx], cmap=cmap, vmin=vmin, vmax=vmax,)
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
