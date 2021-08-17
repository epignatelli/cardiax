import jax
import jax.numpy as jnp
from jax.experimental import optimizers
import os
import h5py
import numpy as onp
import time
import cardiax
import deepx
from IPython.display import display
import matplotlib.pyplot as plt
import scipy.io
from functools import partial
from jax._src.numpy.lax_numpy import _promote_args_inexact


def load_model(url):
    train_state = deepx.optimise.TrainState.restore(url)
    hparams = train_state.hparams
    model = deepx.resnet.ResNet(hparams.hidden_channels, 1, hparams.depth)
    opt = optimizers.adam(0.001)
    params = opt.params_fn(train_state.opt_state)
    return model, hparams, params


def infer(xs, n_refeed=10, a_min=None, a_max=None):
    @partial(
        jax.pmap,
        in_axes=(None, None, 0, 0),
        static_broadcasted_argnums=(0, 1),
        axis_name="device",
    )
    def _infer(model, n_refeed, params, xs):
        def body_fun(inputs, i):
            x = inputs
            y_hat = model.apply(params, x)
            y_hat = jnp.clip(x, a_min, a_max)
            x = deepx.optimise.refeed(x, y_hat)  #  add the new pred to the inputs
            return x, y_hat

        _, ys_hat = jax.lax.scan(body_fun, xs, xs=jnp.arange(n_refeed))
        ys_hat = jnp.swapaxes(jnp.squeeze(ys_hat), 0, 1)
        return ys_hat

    model, hparams, params = load_model("p3aetobr/9336")
    #     model, hparams, params = load_model("p3aetobr/9996")
    #     model, hparams, params = load_model("p3aetobr/10326")
    #     model, hparams, params = load_model("p3aetobr/10976")
    start = time.time()
    if (a_min is None) and (a_max is None):
        ys_hat = deepx.optimise.infer(model, n_refeed, params, xs)
    else:
        ys_hat = _infer(model, n_refeed, params, xs)
    print(
        "Solved forward propagation to {}ms in: {}s".format(
            n_refeed * 5, time.time() - start
        )
    )
    if ys_hat.shape[0] == 1:
        ys_hat = jnp.swapaxes(ys_hat, -3, -4)[None]
    return ys_hat


def evaluate(xs, ys):
    ys_hat = infer(xs)
    assert (
        ys_hat.shape == ys.shape
    ), "ys_hat and ys are of different shapes {} and {}".format(ys_hat.shape, ys.shape)
    loss = jnp.mean((ys_hat - ys) ** 2, axis=(0, 1, 4, 5))
    return ys_hat, loss


def read(filename, t, normalise=False):
    with h5py.File(filename, "r") as sequence:
        states = onp.array(sequence["states"][t:])
        diffusivity = onp.array(sequence["diffusivity"])
        if normalise:
            diffusivity = diffusivity * 500
        diffusivity = onp.tile(diffusivity[None, None], (len(states), 1, 1, 1))
        xs = onp.concatenate([states, diffusivity], axis=-3)[None, None]
        return xs


def save_mat(xs, filename):
    mat_filename = os.path.splitext(filename)[0] + ".mat"
    assert xs.shape[1] == 4, "Missing field. Expected 4, got {}".format(xs.shape[1])
    mdict = {"v": xs[:, 0], "w": xs[:, 1], "u": xs[:, 2], "d": xs[:, 3]}
    return scipy.io.savemat(mat_filename, mdict)


def animate_state(a, d, filename=None, figsize=None):
    states_a = [cardiax.solve.State(*x) for x in a.squeeze()]
    anim = cardiax.plot.animate_state(
        states_a, d, cmap="Blues", vmin=0, vmax=1, figsize=figsize
    )
    display(anim)
    if filename is not None:
        anim.save("data/{}".format(filename))
    return anim


def get_xy_pair(filename, seed, t=0, n_refeed=None):
    print("Reading {}".format(filename.format(seed, "fd")))
    xs = read(filename.format(seed, "fd"), t, normalise=True)
    if n_refeed is None:
        n_refeed = xs.shape[2]

    inputs = xs[:, :, :2]
    d = jnp.round(inputs[:, :, :, -1:], 6)
    inputs = jnp.concatenate([inputs[:, :, :, :3], d], axis=-3)

    n_input_frames = 2
    xs = xs[:, :, n_input_frames : n_refeed + n_input_frames, :3]

    ys_hat = infer(inputs, n_refeed=n_refeed).squeeze()

    xs, ys_hat, d = xs.squeeze(), ys_hat.squeeze(), d[0].squeeze()
    print(
        "x shape is {}; y shape is {}; d shape is {}".format(
            xs.shape, ys_hat.shape, d.shape
        )
    )
    return xs, ys_hat, d


def save_mat_pair(filename, seed, start_t, n_refeed=10):
    # get x and y
    xs, ys_hat, d = get_xy_pair(filename, seed, start_t, n_refeed=n_refeed)

    # add diffusivity map
    d = jnp.broadcast_to(d, xs.shape)
    xs = jnp.concatenate([xs.squeeze(), d], axis=-3)
    ys_hat = jnp.concatenate([ys_hat.squeeze(), d], axis=-3)

    #  save files
    save_mat(xs.squeeze(), filename.format(seed, "fd"))
    save_mat(ys_hat.squeeze(), filename.format(seed, "nn"))
    return xs, ys_hat


def normalise(x, axis=None):
    """To a 0-1 domain"""
    minimum = x.min(axis=axis)[..., None, None]
    maximum = x.max(axis=axis)[..., None, None]
    if jnp.array_equal(maximum, minimum):
        return x - minimum
    return (x - minimum) / (maximum - minimum)


def xlog2y(x, y):
    x, y = _promote_args_inexact("xlog2y", x, y)
    x_ok = x != 0.0
    safe_x = jnp.where(x_ok, x, 1.0)
    safe_y = jnp.where(x_ok, y, 1.0)
    return jnp.where(x_ok, jax.lax.mul(safe_x, jnp.log2(safe_y)), jnp.zeros_like(x))


def kld(a, b, axis=None):
    "Kullback-Leiber divergence"
    a = a + 1e-9
    b = b + 1e-9
    return jnp.mean(xlog2y(a, a / b), axis=axis)


def jsd(a, b, axis=None):
    "Jensen-Shannon divergence"
    a = normalise(a, axis=axis)
    b = normalise(b, axis=axis)
    m = 0.5 * (a + b)
    return 0.5 * (kld(a, m, axis=axis) + kld(b, m, axis=axis))


def mse(x, axis=None):
    return jnp.mean(jnp.square(x), axis=axis)


def rmse(x, y=None, axis=None):
    if y is not None:
        x = y - x
    return jnp.sqrt(mse(x))


def rnmse(a, b, axis=None):
    v = mse(a - b, axis=axis)
    v_bar = mse(b)
    return jnp.sqrt(v / v_bar)


def plot_sequence(v, std=None, ax=None, ymax=None):
    # transpode T and C axes: [T, C] to [C, T]
    v = v.transpose((1, 0))
    if ax is None:
        _, ax = plt.subplots()
    for i, d in enumerate(v):
        ax.plot(v[i])
        if std is not None:
            ax.fill_between(
                jnp.arange(len(d)), v[i] - std[:, i], v[i] + std[:, i], alpha=0.3
            )
    ax.set_xlim([0, len(v[i])])
    if ymax is not None:
        ax.set_ylim([0, ymax])
    plt.legend(["v", "w", "u"])
    plt.show()
    return ax


def max_lyapunov_exp(x):
    return jnp.mean(jnp.log(jnp.abs(jnp.diff(x, axis=0))), axis=(-1, -2))
