import jax
import jax.numpy as jnp

from cardiax import fk, ode
from cardiax.fk import params
from cardiax.ode.conditions import neumann
from cardiax.ode import stimulus
from deepx import generate
import matplotlib.pyplot as plt


def test_stimulate():
    shape = (10, 10)
    stimuli = [
        stimulus.linear(
            shape, stimulus.Direction.NORTH, 0.5, 0.6, stimulus.Protocol(0, 2, 1e9)
        )
    ]

    x = jnp.zeros(shape)
    s = fk.solve.stimulate(1, x, stimuli)

    assert jnp.mean(s[:5]) - 0.6 <= 1e-3
    plt.imshow(s)
    plt.show()


if __name__ == "__main__":
    test_stimulate()