import os
import jax
from jax.lib import xla_bridge
import deepx
import wandb
from jax.experimental.optimizers import (
    adam,
    pack_optimizer_state,
    unpack_optimizer_state,
)
from helx.optimise import Optimiser


def test_trainstate_save():
    wandb.init(mode="disabled")
    jax.config.update("jax_platform_name", "cpu")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    n_devices = jax.local_device_count()
    print(n_devices)

    rng = jax.random.PRNGKey(0)
    hparams = (2, 1, 4)  #  simulate hparams
    model = deepx.resnet.ResNet(hparams[0], hparams[1], hparams[2])
    input_shape = (4, 2, 4, 32, 32)
    output_shape = (4, 5, 3, 32, 32)
    _, params = model.init(rng, input_shape)
    x = jax.random.normal(
        rng, (n_devices,) + (input_shape[0] // n_devices,) + input_shape[1:]
    )
    y = jax.random.normal(
        rng, (n_devices,) + (output_shape[0] // n_devices,) + output_shape[1:]
    )
    optimiser = Optimiser(*adam(0.001))
    opt_state = optimiser.init(params)

    deepx.optimise.btt_step(model, optimiser, 5, 0, opt_state, x, y)
    state = deepx.optimise.TrainState(rng, 23, opt_state, hparams)
    state.save("state_ndev_2.pickle")


def test_trainstate_load():
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    n_devices = jax.local_device_count()
    print(n_devices)

    state = deepx.optimise.TrainState.load("state_ndev_2.pickle")
    hparams = state.hparams
    rng = jax.random.PRNGKey(0)
    model = deepx.resnet.ResNet(hparams[0], hparams[1], hparams[2])
    input_shape = (4, 2, 4, 32, 32)
    output_shape = (4, 5, 3, 32, 32)

    # _, params = model.init(rng, input_shape)
    x = jax.random.normal(
        rng, (n_devices,) + (input_shape[0] // n_devices,) + input_shape[1:]
    )
    y = jax.random.normal(
        rng, (n_devices,) + (output_shape[0] // n_devices,) + output_shape[1:]
    )
    optimiser = Optimiser(*adam(0.001))
    opt_state = state.opt_state
    deepx.optimise.btt_step(model, optimiser, output_shape[2], 0, opt_state, x, y)


if __name__ == "__main__":
    test_trainstate_save()
    test_trainstate_load()
