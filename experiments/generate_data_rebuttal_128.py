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
from helx.optimise.optimisers import Optimiser
import json
import wandb
import pickle
import IPython
from IPython.display import display
from IPython.display import display_javascript
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.animation as animation
import helx
from matplotlib import rc, rcParams
import scipy.io
from functools import partial

rc("animation", html="jshtml")
rc("text", usetex=False)
rcParams["animation.embed_limit"] = 2 ** 128


shape = (600, 600)
reshape = (128, 128)
state = cardiax.solve.init(shape)
start = 0
stop = 200_000
step = 500  #  5 milliseconds
dt = 0.01
dx = 0.01
paramset = cardiax.params.PARAMSET_3
n_refeed = 100


def load_model(url):
    train_state = deepx.optimise.TrainState.restore(url)
    hparams = train_state.hparams
    model = deepx.resnet.ResNet(hparams.hidden_channels, 1, hparams.depth)
    opt = optimizers.adam(0.001)
    params = opt.params_fn(train_state.opt_state)
    return model, hparams, params


def infer(xs, a_min=None, a_max=None):
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


def save_mat_pair(filename, seed, start_t, dt=0.01):
    xs = read(filename.format(seed, "fd"), start_t, normalise=True)

    # make sure the d field is not stripy
    # by replacing the current d with a rounded one at digits
    d_round = jnp.round(xs[:, :, :, -1:], 6)
    xs_no_d = xs[:, :, :, :3]
    xs = jnp.concatenate([xs_no_d, d_round], axis=-3)

    ys_hat = infer(xs[:, :, :2]).squeeze()
    # extract D and unnormalise
    d = xs[0, 0, 2:102, -1:] / 500
    #  clear xs from the D field
    xs = xs.squeeze()[2:102, :3]
    #  check the shapes are okay
    print(seed, d.shape, xs.shape, ys_hat.shape)
    #  add the d field to xs
    xs = jnp.concatenate([xs.squeeze(), d], axis=-3)
    #  add the d field to ys_hat
    ys_hat = jnp.concatenate([ys_hat.squeeze(), d], axis=-3)
    #  save files
    save_mat(xs.squeeze(), filename.format(seed, "fd"))
    save_mat(ys_hat.squeeze(), filename.format(seed, "nn"))
    return xs, ys_hat


#  setup
filename = "rebuttal/heterogeneous_linear_128_{}-{}.hdf5"
#  Finite difference simulation
p1 = cardiax.stimulus.Protocol(40_000 * 0, 2, 1e9)
p2 = cardiax.stimulus.Protocol(40_000 * 1, 2, 1e9)
p3 = cardiax.stimulus.Protocol(40_000 * 2, 2, 1e9)
p4 = cardiax.stimulus.Protocol(40_000 * 3, 2, 1e9)
for seed in range(10):
    rng = jax.random.PRNGKey(seed)
    angle = int(
        jax.random.randint(
            rng,
            (1,),
            0,
            180,
        )
    )
    s1 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
    )
    s2 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p2
    )
    s3 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p3
    )
    s4 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p4
    )
    stimuli = [s1, s2, s3, s4]
    diffusivity = deepx.generate.random_diffusivity(rng, shape)
    deepx.generate.sequence(
        start,
        stop,
        step,
        dt,
        dx,
        paramset,
        diffusivity,
        stimuli,
        filename.format(seed, "fd"),
        reshape=reshape,
        use_memory=True,
        plot_while=False,
    )


#  setup
filename = "rebuttal/heterogeneous_spiral_128_{}-{}.hdf5"
#  Finite difference simulation
stop = 100_000
p1 = cardiax.stimulus.Protocol(40_000 * 0, 2, 1e9)
p2 = cardiax.stimulus.Protocol(40_000 * 0 + 40_000, 2, 1e9)
for seed in range(10):
    rng = jax.random.PRNGKey(seed)
    angle = int(
        jax.random.randint(
            rng,
            (1,),
            0,
            180,
        )
    )
    s1 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
    )
    s2 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle + 90, 0.8, 20, p2
    )
    stimuli = [s1, s2]
    diffusivity = deepx.generate.random_diffusivity(rng, shape)
    deepx.generate.sequence(
        start,
        stop,
        step,
        dt,
        dx,
        paramset,
        diffusivity,
        stimuli,
        filename.format(seed, "fd"),
        reshape=reshape,
        use_memory=True,
        plot_while=False,
    )


#  setup
filename = "rebuttal/homogeneous_linear_128_{}-{}.hdf5"
#  Finite difference simulation
p1 = cardiax.stimulus.Protocol(40_000 * 0, 2, 1e9)
p2 = cardiax.stimulus.Protocol(40_000 * 1, 2, 1e9)
p3 = cardiax.stimulus.Protocol(40_000 * 2, 2, 1e9)
p4 = cardiax.stimulus.Protocol(40_000 * 3, 2, 1e9)
for seed in range(10):
    rng = jax.random.PRNGKey(seed)
    angle = int(
        jax.random.randint(
            rng,
            (1,),
            0,
            180,
        )
    )
    s1 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
    )
    s2 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p2
    )
    s3 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p3
    )
    s4 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p4
    )
    stimuli = [s1, s2, s3, s4]
    diffusivity = jnp.ones(shape) * 0.001
    deepx.generate.sequence(
        start,
        stop,
        step,
        dt,
        dx,
        paramset,
        diffusivity,
        stimuli,
        filename.format(seed, "fd"),
        reshape=reshape,
        use_memory=True,
        plot_while=False,
    )


#  setup
filename = "rebuttal/homogeneous_spiral_128_{}-{}.hdf5"
#  Finite difference simulation
stop = 100_000
p1 = cardiax.stimulus.Protocol(40_000 * 0, 2, 1e9)
p2 = cardiax.stimulus.Protocol(40_000 * 0 + 40_000, 2, 1e9)
for seed in range(0, 10):
    rng = jax.random.PRNGKey(seed)
    angle = int(
        jax.random.randint(
            rng,
            (1,),
            0,
            180,
        )
    )
    s1 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle, 0.2, 20, p1
    )
    s2 = cardiax.stimulus.triangular(
        shape, cardiax.stimulus.Direction.NORTH, angle + 90, 0.8, 20, p2
    )
    stimuli = [s1, s2]
    diffusivity = jnp.ones(shape) * 0.001
    deepx.generate.sequence(
        start,
        stop,
        step,
        dt,
        dx,
        paramset,
        diffusivity,
        stimuli,
        filename.format(seed, "fd"),
        reshape=reshape,
        use_memory=True,
        plot_while=False,
    )
