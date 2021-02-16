import jax


@jax.jit
def euler(f, x0, dt):
    gradients = f(x0)
    x0_flatten, tree = jax.tree_util.tree_flatten(x0)
    grad_flatten, tree = jax.tree_util.tree_flatten(gradients)
    x1 = jax.vmap(jax.numpy.add)(x0_flatten, grad_flatten) * dt
    return jax.tree_util.tree_unflatten(tree, x1)


@jax.jit
def runge_kutta(f, x0, t):
    return jax.experimental.ode.odeint(f, x0, t)
