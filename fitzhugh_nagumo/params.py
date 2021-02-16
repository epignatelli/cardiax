import jax.numpy as np
from typing import NamedTuple


class Params(NamedTuple):
    a: float = 0.7
    b: float = 0.8
    c: float = 0.08


class Stimulus(NamedTuple):
    start: int
    duration: int
    period: int
    value: np.ndarray
