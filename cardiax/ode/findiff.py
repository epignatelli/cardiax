import functools
import jax
import jax.numpy as jnp


def fd(a, axis):
    sliced = functools.partial(jax.lax.slice_in_dim, a, axis=axis)
    a_grad = jnp.concatenate(
        (
            # 3th order edge
            (
                (-11 / 6) * sliced(0, 2)
                + 3.0 * sliced(1, 3)
                - (3 / 2) * sliced(2, 4)
                + (1 / 3) * sliced(3, 5)
            ),
            # 4th order inner
            (
                (1 / 12) * sliced(None, -4)
                - (2 / 3) * sliced(1, -3)
                + (2 / 3) * sliced(3, -1)
                - (1 / 12) * sliced(4, None)
            ),
            # 3th order edge
            (
                (-1 / 3) * sliced(-5, -3)
                + (3 / 2) * sliced(-4, -2)
                - 3.0 * sliced(-3, -1)
                + (11 / 6) * sliced(-2, None)
            ),
        ),
        axis,
    )
    return a_grad


def first(x, h=1.0):
    def _first_derivative(a, axis):
        f = functools.partial(jax.lax.slice_in_dim, a, axis=axis)
        return (
            jnp.concatenate(
                (
                    # 6th order forward edge points
                    (
                        -49 / 20 * f(0, 4)
                        + 6 * f(1, 5)
                        - 15 / 2 * f(2, 6)
                        + 20 / 3 * f(3, 7)
                        - 15 / 4 * f(4, 8)
                        + 6 / 5 * f(5, 9)
                        - 1 / 6 * f(6, 10)
                    ),
                    # 8th order central inner points
                    (
                        +1 / 280 * f(0, -8)
                        - 4 / 105 * f(1, -7)
                        + 1 / 5 * f(2, -6)
                        - 4 / 5 * f(3, -5)
                        # + 0 * f(4, -4)
                        + 4 / 5 * f(5, -3)
                        - 1 / 5 * f(6, -2)
                        + 4 / 105 * f(7, -1)
                        - 1 / 280 * f(8, None)
                    ),
                    # 3th order backward edge points
                    (
                        -1 / 3 * f(-7, -3)
                        + 3 / 2 * f(-6, -2)
                        - 3 * f(-5, -1)
                        + 11 / 6 * f(-4, None)
                    ),
                ),
                axis=axis,
            )
            / h
        )

    return [_first_derivative(x, ax) for ax in range(len(x.shape))]


def second(x, h):
    def _second_derivative(a, axis):
        f = functools.partial(jax.lax.slice_in_dim, a, axis=axis)
        return (
            jnp.concatenate(
                (
                    # 6th order forward edge points
                    (
                        +469 / 90 * f(0, 4)
                        - 223 / 10 * f(1, 5)
                        + 879 / 20 * f(2, 6)
                        - 949 / 18 * f(3, 7)
                        + 41 * f(4, 8)
                        - 201 / 10 * f(5, 9)
                        + 1019 / 180 * f(6, 10)
                        - 7 / 10 * f(7, 11)
                    ),
                    # 8th order central inner points
                    (
                        -1 / 560 * f(0, -8)
                        + 8 / 315 * f(1, -7)
                        - 1 / 4 * f(2, -6)
                        + 8 / 5 * f(3, -5)
                        - 205 / 71 * f(4, -4)
                        + 8 / 5 * f(5, -3)
                        - 1 / 5 * f(6, -2)
                        + 8 / 315 * f(7, -1)
                        - 1 / 560 * f(8, None)
                    ),
                    # 2th order backward edge points
                    (-1 * f(-7, -3) + 4 * f(-6, -2) - 5 * f(-5, -1) + 2 * f(-4, None)),
                ),
                axis=axis,
            )
            / h ** 2
        )

    return [_second_derivative(x, ax) for ax in range(len(x.shape))]