import jax.numpy as np
import fk


def test_stimulate():
    shape = (800, 800)
    
    A = fk.stimulus.protocol(start=0, duration=2, period=50)
    A = fk.stimulus.linear(shape, direction="up", coverage=.05, modulus=1., protocol=A)
    B = fk.stimulus.protocol(start=10, duration=2, period=50)
    B = fk.stimulus.linear(shape, direction="left", coverage=0.5, modulus=1., protocol=B)
    C = fk.stimulus.protocol(start=30, duration=2, period=1000000)
    C = fk.stimulus.linear(shape, direction="right", coverage=0.5, modulus=1., protocol=C)    
    stimuli = [A, B, C]
    
    A_is_active_at = set([0, 1, 50, 51, 100, 101, 150, 151, 200, 201, 250, 251])
    B_is_active_at = set([10, 11, 60, 61, 110, 111, 160, 161, 210, 211, 260, 261])
    C_is_active_at = set([30, 31])
    times = set(range(0, 300))
    
    X = np.zeros(shape, dtype="float32")
    for t in times:
        X = fk.model.stimulate(t, X, stimuli)
        if t in A_is_active_at or t in B_is_active_at or t in C_is_active_at:
            # stimulus active
            assert np.sum(np.nonzero(X)) != 0, "Failed stimulus test at time {}, nonzero is {}".format(t, np.nonzero(X))
        else:
            # stimulus non active
            assert np.sum(np.nonzero(X)) == 0, "Failed stimulus test at time {}, nonzero is {}".format(t, np.nonzero(X))
        X = np.zeros(shape, dtype="float32")
    return


if __name__ == "__main__":
    test_stimulate()
    
#     def stimulate(t, X, stimuli):
#     stimulated = np.zeros_like(X)
#     for stimulus in stimuli:
#         active = t > stimulus["start"]
#         active &= t < stimulus["start"] + stimulus["duration"]
#         # for some weird reason checks for cyclic stimuli does not work
#         active &= (np.mod(t - stimulus["start"], stimulus["period"]) < stimulus["duration"])  # cyclic
#         stimulated = np.where(stimulus["field"] * (active), stimulus["field"], stimulated)
#     return np.where(stimulated != 0, stimulated, X)