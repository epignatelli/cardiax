import jax
import jax.numpy as np


def protocol(start, duration, period):
    return {
        "start": int(start),
        "duration": int(duration),
        "period": int(period)
    }


def rectangular(shape, jax_indices, modulus, protocol):
    mask = np.zeros(shape, dtype="float32")
    mask = jax.ops.index_update(mask, jax_indices, modulus)
    stimulus = {"field": mask}
    return {**stimulus, **protocol}


def circular(shape, centre, radius, protocol):
    raise NotImplementedError
