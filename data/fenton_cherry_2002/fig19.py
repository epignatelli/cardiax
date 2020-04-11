import os
import math
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import fk


# simulation inputs (real values)
field_size = (11.25, 11.25)  # cm
d = 0.001  # (cm^2/ms)
cell_parameters = fk.params.params6()
cell_parameters["tau_d"] = 0.388

# infinitesimals
dx = 0.015  # (cm/units) - Fenton 1998 recommends ~200, 300 micron/gridunit (~0.02, 0.03), smaller dx means finer grid
dt = 0.02  # (ms) - Fenton 1998 recommends few hundreds of ms (~0.01, 0.04)

# to computational units
shape = fk.convert.field_to_shape(field_size, dx)

stripe_size = int(shape[0] / 10)

protocol1 = fk.stimulus.protocol(start=0, duration=2, period=0)
s1 = fk.stimulus.rectangular(shape, jax.ops.index[:, -stripe_size:], 1., protocol1)

protocol2 = fk.stimulus.protocol(start=fk.convert.ms_to_units(300, dt), duration=2, period=0)
s2 = fk.stimulus.rectangular(shape, jax.ops.index[-fk.convert.cm_to_units(6, dx):], 1., protocol2)

stimuli = [s1, s2]

checkpoints = [0, 350, 385, 405, 425, 445, 460, 475, 495, 510, 525, 575, 625, 910, 1115, 1160, 1210, 1260, 1300, 1340, 1390, 1420, 1480, 1530, 1600, 1670]
checkpoints = [fk.convert.ms_to_units(ck, dt) for ck in checkpoints]

print(shape)
print("real tissue size:", field_size)
print("Checkpointing at:", checkpoints)
print(cell_parameters)

state = fk.model.init(shape)
states = []
for i in range(len(checkpoints) - 1):
    state = fk.model._forward(state, checkpoints[i], checkpoints[i + 1], cell_parameters, np.ones(shape) * d, stimuli, dt, dx)
    u = state[2]
    states.append(u)
          
fk.model.show_grid(states, checkpoints, (20, 20), dt)
fig = plt.gcf()
filename = os.path.realpath(__file__)
filename = os.path.splitext(filename)[0] + ".png"
fig.savefig(filename)
