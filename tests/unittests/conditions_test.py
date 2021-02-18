import jax
import jax.numpy as jnp

from cardiax import fk, ode
from cardiax.fk.params import PARAMSET_5
from cardiax.ode.conditions import neumann
from deepx import generate


def test_neumann():
    shape = (32, 32)
    condition = neumann()

    x = jnp.ones(shape)
    a = condition.apply(x)
    b = condition.restore(a)
    assert b.shape == x.shape, "{} != {}".format(b.shape, x.shape)


if __name__ == "__main__":
    test_neumann()