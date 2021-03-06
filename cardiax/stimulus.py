import random
from typing import NamedTuple, Tuple
from enum import IntEnum

import jax
import jax.numpy as np
from scipy.ndimage.interpolation import rotate

Shape = Tuple[int, ...]
Point2D = Tuple[int, int]


class Protocol(NamedTuple):
    start: int
    duration: int
    period: int


class Stimulus(NamedTuple):
    protocol: Protocol
    field: np.ndarray


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


def rectangular(
    shape: Shape,
    centre: Point2D,
    size: Point2D,
    modulus: float,
    protocol: Protocol,
):
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
    x1 = int(centre[0] - size[0] / 2)
    x2 = int(centre[0] + size[0] / 2)
    y1 = int(centre[1] - size[1] / 2)
    y2 = int(centre[1] + size[1] / 2)
    mask = jax.ops.index_update(mask, jax.ops.index[x1:x2, y1:y2], modulus)
    return Stimulus(protocol, mask)


def linear(
    shape: Shape,
    direction: Direction,
    coverage: float,
    modulus: float,
    protocol: Protocol,
):
    """
    Generates a linear wave stimulus.
    Args:
        shape (Tuple[int, int]): the shape of the stimulus array in simulation units (not cm)
        direction (str): Direction of the wave as a string. Can be either:
                         'left', 'right', 'up', or 'down'
        coverage (float): percentage of the field that the wave will cover.
                        It must be between 0 and 1
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
    stripe_size = int(shape[0] * coverage)
    stripe = None
    if direction == Direction.WEST:
        stripe = jax.ops.index[:, :stripe_size]
    elif direction == Direction.EAST:
        stripe = jax.ops.index[:, -stripe_size:]
    elif direction == Direction.NORTH:
        stripe = jax.ops.index[:stripe_size, :]
    elif direction == Direction.SOUTH:
        stripe = jax.ops.index[-stripe_size:, :]
    else:
        raise ValueError(
            "direction mus be either 'left', 'right', 'up', or 'down' not %s"
            % direction
        )

    field = np.zeros(shape, dtype="float32")
    field = jax.ops.index_update(field, stripe, modulus)
    return Stimulus(protocol, field)


def triangular(
    shape: Shape,
    direction: float,
    angle: float,
    coverage: float,
    modulus: float,
    protocol: Protocol,
):
    """
    Generates a linear wave at a custom angle.
    Args:
        shape (Tuple[int, int]): the shape of the stimulus array in simulation units (not cm)
        direction (str): Direction of the wave as a string. Can be either:
                         'left', 'right', 'up', or 'down'
        angle (str): Incidence angle of the wave in degrees.
        coverage (float): percentage of the field that the wave will cover.
                        It must be between 0 and 1
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
    stim = linear(shape, direction, coverage, modulus, protocol)
    field = rotate(
        stim.field, angle=angle, mode="nearest", prefilter=False, reshape=False
    )
    return Stimulus(protocol, field)
