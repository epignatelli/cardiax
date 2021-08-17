import numpy as onp
import scipy.io
import util
import scipy.stats


n_refeed = 200
n_runs = 10
divergences, divergences_control = [], []
rnmses, rnmse_controls = [], []
diff = []
legend_names = [
    "Heterogeneous spiral",
    "Heterogeneous linear",
    "Homogeneous spiral",
    "Homogeneous linear",
    "Heterogeneous spiral-control",
    "Heterogeneous linear-control",
    "Homogeneous spiral-control",
    "Homogeneous linear-control",
]


filename = "rebuttal/heterogeneous_spiral_128_{}-{}.hdf5"
for seed in range(n_runs):
    # get the x, y pairs
    t = int(45_000 * 0.01 / 5)
    xs, ys, ds = util.get_xy_pair(filename, seed, t=t, n_refeed=n_refeed)
    gray = onp.ones_like(xs) * 0.5

    xs = onp.array(xs)
    ys = onp.array(ys)
    ds = onp.array(ds)

    div = onp.array(util.jsd(xs, ys, axis=(2, 3)))
    div_control = onp.array(util.jsd(xs, gray, axis=(2, 3)))
    rmse = onp.array(util.rnmse(xs, ys, axis=(2, 3)))
    rmse_control = onp.array(util.rnmse(xs, gray, axis=(2, 3)))

    divergences.append(div)
    divergences_control.append(div_control)
    rnmses.append(rmse)
    rnmse_controls.append(rmse_control)


filename = "rebuttal/heterogeneous_linear_128_{}-{}.hdf5"
for seed in range(n_runs):
    # get the x, y pairs
    t = int(55_000 * 0.01 / 5)
    xs, ys, ds = util.get_xy_pair(filename, seed, t=t, n_refeed=n_refeed)
    gray = onp.ones_like(xs) * 0.5

    xs = onp.array(xs)
    ys = onp.array(ys)
    ds = onp.array(ds)

    div = onp.array(util.jsd(xs, ys, axis=(2, 3)))
    div_control = onp.array(util.jsd(xs, gray, axis=(2, 3)))
    rmse = onp.array(util.rnmse(xs, ys, axis=(2, 3)))
    rmse_control = onp.array(util.rnmse(xs, gray, axis=(2, 3)))

    divergences.append(div)
    divergences_control.append(div_control)
    rnmses.append(rmse)
    rnmse_controls.append(rmse_control)


filename = "rebuttal/homogeneous_spiral_128_{}-{}.hdf5"
for seed in range(n_runs):
    # get the x, y pairs
    t = int(45_000 * 0.01 / 5)
    xs, ys, ds = util.get_xy_pair(filename, seed, t=t, n_refeed=n_refeed)
    gray = onp.ones_like(xs) * 0.5

    xs = onp.array(xs)
    ys = onp.array(ys)
    ds = onp.array(ds)

    div = onp.array(util.jsd(xs, ys, axis=(2, 3)))
    div_control = onp.array(util.jsd(xs, gray, axis=(2, 3)))
    rmse = onp.array(util.rnmse(xs, ys, axis=(2, 3)))
    rmse_control = onp.array(util.rnmse(xs, gray, axis=(2, 3)))

    divergences.append(div)
    divergences_control.append(div_control)
    rnmses.append(rmse)
    rnmse_controls.append(rmse_control)


filename = "rebuttal/homogeneous_linear_128_{}-{}.hdf5"
for seed in range(n_runs):
    # get the x, y pairs
    t = int(55_000 * 0.01 / 5)
    xs, ys, ds = util.get_xy_pair(filename, seed, t=t, n_refeed=n_refeed)
    gray = onp.ones_like(xs) * 0.5

    xs = onp.array(xs)
    ys = onp.array(ys)
    ds = onp.array(ds)

    div = onp.array(util.jsd(xs, ys, axis=(2, 3)))
    div_control = onp.array(util.jsd(xs, gray, axis=(2, 3)))
    rmse = onp.array(util.rnmse(xs, ys, axis=(2, 3)))
    rmse_control = onp.array(util.rnmse(xs, gray, axis=(2, 3)))

    divergences.append(div)
    divergences_control.append(div_control)
    rnmses.append(rmse)
    rnmse_controls.append(rmse_control)


scipy.io.savemat(
    "tissue_size_512.mat",
    {
        "divergences": divergences,
        "divergences_control": divergences_control,
        "rnmses": rnmses,
        "rnmse_controls": rnmse_controls,
        "diff": diff,
    },
)
