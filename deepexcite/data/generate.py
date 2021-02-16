import jax.numpy as np
import fenton_karma as fk


def generate_fk(start, stop, dt, dx, cell_parameters, diffusivity, stimuli, filename, reshape=None, save_interval_ms=1):
    """
    Generates a new spatio-temporal sequence of fenton-karma simulations.
    Args:
        start (float): start time of the simulation in millisecons
        stop (float): end time of the simulation in milliseconds
        dt (float): time infinitesimal for temporal gradients calculation (nondimensional)
        dx (float): space infinitesimal for spatial gradients calculation (nondimensional)
        cell_parameters (Dict[string, float]): set of electrophysiological parameters to use for the simulation
                                             Units are cm for space, ms for time, mV for potential difference, uF for capacitance
        diffusivity (np.ndarray): diffusivity map for tissue conduction velocity. Spatial units are in simulation units,
                                  and the value is the dimensional value, usually set to 0.001 cm^2/ms
        stimuli (List[Dict[string, object]]): The stimulus as a dictionary containing:
                                              "field": (np.ndarray) [mV/unitgrid],
                                              "start": (int) [ms],
                                              "duration": (int) [ms],
                                              "period": (int) [ms]
        filename (str): file path to save the simulation to. Simulation is saved with steps of 1ms
    """
    # check shapes
    for s in stimuli:
        assert diffusivity.shape == s.field.shape, "Inconsistend stimulus shapes {} and diffusivity {}".format(
        diffusivity.shape, s.field.shape)

    # checkpoints
    checkpoints = np.arange(int(start), int(stop), int(save_interval_ms / dt))  # this guarantees a checkpoint every ms

    # shapes
    shape = diffusivity.shape
    tissue_size = fk.convert.shape_to_realsize(shape, dx)

    # show
    print("Tissue size", tissue_size, "Grid size", diffusivity.shape)
    print("Checkpointing at:", checkpoints)
    print("Cell parameters", cell_parameters)
    fk.plot.plot_stimuli(*stimuli)

    # init storage
    init_size = reshape or shape
    hdf5 = fk.io.init(filename, init_size, n_iter=len(checkpoints), n_stimuli=len(stimuli))
    fk.io.add_params(hdf5, cell_parameters, diffusivity, dt, dx, shape=reshape)
    fk.io.add_stimuli(hdf5, stimuli, shape=reshape)

    states_dset = hdf5["states"]
    state = fk.model.init(shape)
    for i in range(len(checkpoints) - 1):
        print("Solving at: %dms/%dms\t\t" % (fk.convert.units_to_ms(checkpoints[i + 1], dt), fk.convert.units_to_ms(checkpoints[-1], dt)), end="\r")
        state = fk.model._forward(state, checkpoints[i], checkpoints[i + 1], cell_parameters, diffusivity, stimuli, dt, dx)
        fk.io.add_state(states_dset, state, i, shape=(len(state), *reshape))

    print()
    hdf5.close()
