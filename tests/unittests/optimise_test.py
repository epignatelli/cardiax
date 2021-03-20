import os
import jax
import deepx
import wandb


def test_trainstate():
    wandb.init(mode="disabled")

    jax.config.update("jax_platform_name", "cpu")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    n_devices = jax.local_device_count()
    print(n_devices)

    rng = jax.random.PRNGKey(0)
    model = deepx.resnet.ResNet(4, 3, 2)
    input_shape = (4, 2, 4, 32, 32)
    _, params = model.init(rng, input_shape)
    hparams = (2, 3, 4)  #  simulate hparams
    x = jax.random.normal(
        rng, (n_devices,) + (input_shape[0] // n_devices,) + input_shape[1:]
    )
    jax.pmap(model.apply)(params, x)
    state = deepx.optimise.TrainState(rng, 23, params, hparams)
    serialised_state = state.serialise()

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    n_devices = jax.local_device_count()
    print(n_devices)

    train_state = deepx.optimise.TrainState.deserialise(serialised_state)
    x = jax.random.normal(
        rng, (n_devices,) + (input_shape[0] // n_devices,) + input_shape[1:]
    )
    params = train_state.params
    jax.pmap(model.apply)(params, x)


if __name__ == "__main__":
    test_trainstate()