import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter


def plot_stimuli(stimuli, **kwargs):
    fig, ax = plt.subplots(
        1, len(stimuli), figsize=(kwargs.pop("figsize", None) or (25, 5))
    )
    if len(stimuli) <= 1:
        ax = [ax]
    vmin = kwargs.pop("vmin", -30)
    vmax = kwargs.pop("vmax", 30)
    cmap = kwargs.pop("cmap", "RdBu")
    for i, stimulus in enumerate(stimuli):
        im = ax[i].imshow(stimulus.field, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title(
            "Protocol:\nstart: {}, \nduration: {}, \nperiod: {}".format(
                int(stimulus.protocol.start),
                int(stimulus.protocol.duration),
                int(stimulus.protocol.period),
            ),
        )
    fig.tight_layout()
    return fig, ax


def plot_diffusivity(diff, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(kwargs.pop("figsize", (25, 5))))
    im = ax.imshow(diff, cmap="gray")
    ax.set_title("Diffusivity map")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    clb = plt.colorbar(im, ax=ax)
    clb.ax.set_title("[cm^2/ms]")
    return fig, ax


def plot_state(state, **kwargs):
    array = tuple(state)
    fig, ax = plt.subplots(
        1, len(array), figsize=(kwargs.pop("figsize", None) or (25, 5))
    )
    vmin = kwargs.pop("vmin", 0)
    vmax = kwargs.pop("vmax", 1)
    cmap = kwargs.pop("cmap", "RdBu")

    for i in range(len(ax)):
        if array[i] is None:
            continue
        im = ax[i].imshow(array[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title(state._fields[i])
        ax[i].xaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
        )
        ax[i].set_xlabel("x [cm]")
        ax[i].yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
        )
        ax[i].set_ylabel("y [cm]")
    return fig, ax


def plot_states(states, **kwargs):
    fig, ax = plt.subplots(
        len(states),
        len(states[0]),
        figsize=(kwargs.pop("figsize", None) or (25, 5 * len(states))),
    )
    ax = ax if ax.ndim > 1 else ax.reshape(1, -1)
    vmin = kwargs.pop("vmin", 0)
    vmax = kwargs.pop("vmax", 1)
    cmap = kwargs.pop("cmap", "RdBu")

    for t, state in enumerate(states):
        array = tuple(state)
        for i in range(len(ax[t])):
            if array[i] is None:
                continue
            im = ax[t, i].imshow(array[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            plt.colorbar(im, ax=ax[t, i])
            ax[t, i].set_title(state._fields[i])
            ax[t, i].xaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i].set_xlabel("x [cm]")
            ax[t, i].yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i].set_ylabel("y [cm]")
    return fig, ax


def compare_states(states_a, states_b, **kwargs):
    fig, ax = plt.subplots(
        len(states_a),
        len(states_a[0]) * 3,
        figsize=(kwargs.pop("figsize", None) or (50, 5)),
    )
    ax = ax if ax.ndim > 1 else ax.reshape(1, -1)
    vmin = kwargs.pop("vmin", 0)
    vmax = kwargs.pop("vmax", 1)
    cmap = kwargs.pop("cmap", "RdBu")

    for t, state in enumerate(states_a):
        state_a, state_b = states_a[t], states_b[t]
        a, b = tuple(states_a[t]), tuple(states_b[t])
        for i in range(0, len(ax), 2):
            if a[i] is None or b[i] is None:
                continue
            im = ax[t, i].imshow(a[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            ax[t, i].set_title(state_a._fields[i])
            ax[t, i].xaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i].set_xlabel("x [cm]")
            ax[t, i].yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i].set_ylabel("y [cm]")
            im = ax[t, i + 1].imshow(b[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
            plt.colorbar(im, ax=ax[t, i + 1])
            ax[t, i + 1].set_title("\hat{%s}" % state_b._fields[i])
            ax[t, i + 1].xaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i + 1].set_xlabel("x [cm]")
            ax[t, i + 1].yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i + 1].set_ylabel("y [cm]")
            im = ax[t, i + 2].imshow(
                a[i] - b[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
            )
            plt.colorbar(im, ax=ax[t, i + 2])
            ax[t, i + 2].set_title(
                "\hat{%s} - {%s}" % (state_b._fields[i], state_b._fields[i])
            )
            ax[t, i + 2].xaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i + 2].set_xlabel("x [cm]")
            ax[t, i + 2].yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
            )
            ax[t, i + 2].set_ylabel("y [cm]")
    return fig, ax


def animate_state(states, times=None, **kwargs):
    cached_backend = matplotlib.get_backend()
    matplotlib.use("nbAgg")
    fig, ax = plt.subplots(
        1, len(states[0]), figsize=(kwargs.pop("figsize", None) or (25, 5))
    )
    vmin = kwargs.pop("vmin", 0)
    vmax = kwargs.pop("vmax", 1)
    cmap = kwargs.pop("cmap", "RdBu")
    times = times or range(len(states))

    # setup figure
    state = states[0]
    graphics = []
    for i in range(len(ax)):
        im = ax[i].imshow(state[i], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title(state._fields[i])
        ax[i].xaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
        )
        ax[i].set_xlabel("x [cm]")
        ax[i].yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
        )
        ax[i].set_ylabel("y [cm]")
        fig.title = "time: {}".format(times[0])
        graphics.append(im)

    def update(t):
        state = states[t]
        for i in range(len(ax)):
            im = graphics[i].set_data(state[i])
            fig.title = "time: {}".format(times[t])
        return graphics

    animation = FuncAnimation(fig, update, frames=range(len(states)), blit=True)
    matplotlib.use(cached_backend)
    return animation


def show_grid(
    states,
    times=[],
    figsize=None,
    rows=5,
    font_size=10,
    vmin=-85,
    vmax=15,
    cmap="magma",
):
    cols = math.ceil(len(states) / rows)
    rows = max(2, min(rows, len(states)))
    fig, ax = plt.subplots(cols, rows, figsize=figsize)
    ax = ax.flatten()
    idx = 0

    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title

    for idx in range(len(states)):
        im = ax[idx].imshow(
            states[idx],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax[idx].xaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
        )
        ax[idx].set_xlabel("x [cm]")
        ax[idx].yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "{:.1f}".format(y / 100))
        )
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
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}".format(y / 100)))
    ax.set_xlabel("x [cm]")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.1f}".format(y / 100)))
    ax.set_ylabel("y [cm]")
    ax.set_zlabel("Voltage [mV]")
    # crop image
    fig.tight_layout()
    return fig, ax
