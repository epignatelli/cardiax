import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def compare(y_hat, y):
    fig, ax = plt.subplots(len(y_hat), 3, figsize=(15, 5 * len(y_hat)))
    if y_hat.size(0) == 1:
        ax = [ax]

    vmin = min(y_hat.min(), y.min())
    vmax = max(y_hat.max(), y.max())
    for i in range(len(ax)):
        # prediction
        im = ax[i][0].imshow(y_hat[i].squeeze(), vmin=vmin, vmax=vmax, cmap="gray")
        ax[i][0].set_title("y_hat")
        plt.colorbar(im, ax=ax[i][0])

        # truth
        im = ax[i][1].imshow(y[i].squeeze(), vmin=vmin, vmax=vmax, cmap="gray")
        ax[i][1].set_title("y")
        plt.colorbar(im, ax=ax[i][1])

        # error
        error = y_hat[i] - y[i]
        im = ax[i][2].imshow(error.squeeze(), cmap="gray")
        ax[i][2].set_title("y_hat - y")
        plt.colorbar(im, ax=ax[i][2])

        # both
        for j in range(len(ax[i])):
            ax[i][j].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
            ax[i][j].set_xlabel("x [cm]")
            ax[i][j].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1f}'.format(y / 100)))
            ax[i][j].set_ylabel("y [cm]")

    fig.tight_layout()
    return fig, ax