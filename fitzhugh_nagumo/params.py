import jax.numpy as np
from typing import NamedTuple


class Params(NamedTuple):
    a: float = 0.7
    b: float = 0.8
    c: float = 0.08


class State(NamedTuple):
    v: np.ndarray
    w: np.ndarray


class Stimulus(NamedTuple):
    t: int
    field: np.ndarray
