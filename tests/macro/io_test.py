import h5py
import os
import jax
import jax.numpy as np
import fk
import functools


# simulation inputs (real values)
field_size = (12, 12)  # cm
d = 0.001  # (cm^2/ms)
cell_parameters = fk.params.params5()

# infinitesimals
dx = 0.015  # (cm/units) - Fenton 1998 recommends ~200, 300 micron/gridunit (~0.02, 0.03), smaller dx means finer grid
dt = 0.02  # (ms) - Fenton 1998 recommends few hundreds of ms (~0.01, 0.04)

# shape
shape = fk.convert.field_to_shape(field_size, dx)

# diffusivity
diffusivity = np.ones(shape) * d

# stimuli
stripe_size = int(shape[0] / 10)
protocol1 = fk.stimulus.protocol(start=0, duration=2, period=0)
s1 = fk.stimulus.rectangular(shape, jax.ops.index[:, -stripe_size:], 1., protocol1)
protocol2 = fk.stimulus.protocol(start=fk.convert.ms_to_units(400, dt), duration=2, period=0)
s2 = fk.stimulus.rectangular(shape, jax.ops.index[-fk.convert.cm_to_units(6, dx):], 1., protocol2)
stimuli = [s1, s2]

# checkpoints
checkpoints = np.arange(0, 100000, int(1/dt))  # this guarantees a checkpoint every ms

# show
print(shape)
print("real tissue size:", field_size)
print("Checkpointing at:", checkpoints)
print(cell_parameters)
fk.model.show_stimuli(*stimuli)

# init storage
hdf5 = fk.io.init("test.hdf5", shape, n_iter=len(checkpoints), n_stimuli=len(stimuli))
fk.io.add_params(hdf5, cell_parameters, diffusivity, dt, dx)
fk.io.add_stimuli(hdf5, stimuli)

states_dset = hdf5["states"]
state = fk.model.init(shape)
for i in range(len(checkpoints) - 1):
    print("Solving at: %dms/%dms\t\t" % (fk.convert.units_to_ms(checkpoints[i + 1], dt), fk.convert.units_to_ms(checkpoints[-1], dt)), end="\r")
    state = fk.model._forward(state, checkpoints[i], checkpoints[i + 1], cell_parameters, diffusivity, stimuli, dt, dx)
    fk.io.add_state(states_dset, state, i)

    print()
    hdf5.close()

with h5py.File("test.hdf5", "r") as file:
    states = file["states"][::int(1/dt) * 10]
    dt = file["params/dt"][...]
    print(file["states"].shape)

fig, ax = fk.model.show_grid(states[:, 2], range(len(states) + 1), (10, 10), dt)
fig.savefig("test.png")