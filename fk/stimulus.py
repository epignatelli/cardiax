import jax
import jax.numpy as np


def protocol(start, duration, period=0):
    """
    Generates a time protocol to manage the simulus by start time, duration and eventual period.
    Args:
        start (int):
        duration (int):
        period (int): 
    """
    return {
        "start": int(start),
        "duration": int(duration),
        "period": int(period)
    }


def rectangular(shape, centre, size, modulus, protocol):
    """
    Generates a rectangular stimulus given the center and the dimension of the rectangle.
    Args:
        shape (Tuple[int, int]): the shape of the stimulus array in simulation units (not cm)
        centre (Tuple[int, int]): the carthesian coordinates of the centre of the rectangle in simulation units
        size (Tuple[int, int]): the width and height dimensions of the rectangle in simulation units
        modulus (float): the amplitude of the stimulus in mV
        protocol (Dict[str, int]): the time protocol used to manage the stimuli.
                                   It's values are in simulation units (not ms)
    Returns:
        (Dict[str, object]): The stimulus as a dictionary containing:
                             "field": (np.ndarray),
                             "start": (int),
                             "duration": (int),
                             "period": (int)
    """
    mask = np.zeros(shape, dtype="float32")
    x1 = (int(centre[0] - size[0] / 2))
    x2 = (int(centre[0] + size[0] / 2))
    y1 = (int(centre[1] - size[1] / 2))
    y2 = (int(centre[1] + size[1] / 2))
    mask = jax.ops.index_update(mask, jax.ops.index[x1:x2, y1:y2], modulus)
    stimulus = {"field": mask}
    return {**stimulus, **protocol}


def linear(shape, direction, coverage, modulus, protocol):
    """
    Generates a linear wave stimulus.
    Args:
        shape (Tuple[int, int]): the shape of the stimulus array in simulation units (not cm)
        direction (str): Direction of the wave as a string. Can be either:
                         'left', 'right', 'up', or 'down'
        coverage (float): percentage of the field that the wave will cover.
                        It must be between 0 and 1
        protocol (Dict[str, int]): the time protocol used to manage the stimuli.
                                   It's values are in simulation units (not ms)
    Returns:
        (Dict[str, object]): The stimulus as a dictionary containing:
                             "field": (np.ndarray),
                             "start": (int),
                             "duration": (int),
                             "period": (int)
    """
    direction = direction.lower()
    stripe_size = int(shape[0] * coverage)
    stripe = None
    if direction == "left":
        stripe = jax.ops.index[:, :stripe_size]
    elif direction == "right":
        stripe = jax.ops.index[:, -stripe_size:]
    elif direction == "up":
        stripe = jax.ops.index[:stripe_size, :]
    elif direction == "down":
        stripe = jax.ops.index[-stripe_size:, :]
    else:
        raise ValueError("direction mus be either 'left', 'right', 'up', or 'down' not %s" % direction)
        
    mask = np.zeros(shape, dtype="float32")
    mask = jax.ops.index_update(mask, stripe, modulus)
    stimulus = {"field": mask}
    return {**stimulus, **protocol}


def circular(shape, centre, radius, protocol):
    raise NotImplementedError
