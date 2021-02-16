import jax


@jax.jit
def euler(step_fn, x, dt):
    def apply(x):
        return jax.numpy.add(x, step_fn(x) * dt)

    return jax.tree_multimap(apply, x, dt)


@jax.jit
def rk(step_fn, x, t, *args):
    return jax.experimental.ode.odeint(step_fn, x, t)
