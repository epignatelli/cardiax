import logging
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import wandb
from absl import app, flags
from helx.types import Optimiser
from jax.experimental import optimizers

from deepx import resnet, optimise
from deepx.dataset import Dataset

#  program
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_integer("log_frequency", 5, "")
flags.DEFINE_boolean("debug", False, "")

#  model args
flags.DEFINE_integer("hidden_channels", 8, "")
flags.DEFINE_integer("in_channels", 4, "")
flags.DEFINE_integer("depth", 5, "")

#  optimisation args
flags.DEFINE_float("lr", 0.0001, "")
flags.DEFINE_integer("batch_size", 4, "")
flags.DEFINE_float("lamb", 0.05, "")
flags.DEFINE_integer("evaluation_steps", 20, "")
flags.DEFINE_integer("epochs", 100, "")
flags.DEFINE_integer("train_maxsteps", 100, "")
flags.DEFINE_integer("val_maxsteps", 10, "")
flags.DEFINE_bool("tbtt", True, "")
flags.DEFINE_float("increase_at", 0.01, "")
flags.DEFINE_float("teacher_forcing_prob", 0.0, "")
flags.DEFINE_string("from_checkpoint", "", "")

#  input data arguments
flags.DEFINE_string("root", "/home/epignatelli/repos/cardiax/data/", "")
flags.DEFINE_string("paramset", "paramset5", "")
flags.DEFINE_list("size", [256, 256], "")
flags.DEFINE_integer("frames_in", 2, "")
flags.DEFINE_integer("frames_out", 1, "")
flags.DEFINE_integer("step", 1, "")
flags.DEFINE_integer("refeed", 5, "")
flags.DEFINE_boolean("preload", False, "")

FLAGS = flags.FLAGS


def main(argv):
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if FLAGS.debug else logging.INFO
    )
    #  hparms
    logging.info("Parsing hyperparamers and initialising logger...")
    hparams = resnet.HParams.from_flags(FLAGS)
    refeed = hparams.refeed
    epochs = hparams.epochs
    train_maxsteps = hparams.train_maxsteps if not hparams.debug else 1
    val_maxsteps = hparams.val_maxsteps if not hparams.debug else 1
    log_frequency = hparams.log_frequency
    lamb = hparams.lamb
    tbtt = hparams.tbtt
    wandb.init(project="deepx")

    #  log
    logging.info("Logging hyperparameters...")
    list(
        map(
            lambda i: setattr(wandb.config, hparams._fields[i], hparams[i]),
            list(range(len(hparams))),
        )
    )
    n_sequence_out = hparams.refeed * hparams.frames_out

    #  datasets
    input_shape = (
        hparams.batch_size,
        hparams.frames_in,
        hparams.in_channels,
        *hparams.size,
    )
    logging.debug("Input shape is : {}".format(input_shape))
    logging.info("Creating datasets...")
    make_dataset = lambda subdir: Dataset(
        folder=os.path.join(hparams.root, subdir),
        frames_in=hparams.frames_in,
        frames_out=n_sequence_out,
        step=hparams.step,
        batch_size=hparams.batch_size,
    )
    train_set = make_dataset("train")
    val_set = make_dataset("val")

    #  init
    logging.info("Initialising model...")
    rng = jax.random.PRNGKey(hparams.seed)
    rng_val = jax.random.PRNGKey(hparams.seed)
    model = resnet.ResNet(
        hidden_channels=hparams.hidden_channels,
        out_channels=hparams.frames_out,
        depth=hparams.depth,
    )
    logging.info("Initialising optimisers...")
    _, params = model.init(rng, input_shape)
    if hparams.from_checkpoint not in ("", None):
        with open(hparams.from_checkpoint, "rb") as f:
            params = pickle.load(f)
    optimiser = Optimiser(*optimizers.adam(hparams.lr))
    optimiser_state = optimiser.init(params)
    with open("{}/{}.pickle".format(wandb.run.dir, "start"), "wb") as f:
        pickle.dump(params, f)
    logging.info("Starting training...")
    global_step = 0

    update = optimise.tbtt_step if hparams.tbtt else optimise.btt_step
    for i in range(epochs):
        #  train
        train_loss_epoch = 0.0
        for j in range(train_maxsteps):
            k = (train_maxsteps * i) + j
            global_step += 1
            rng, _ = jax.random.split(rng)
            xs, ys = train_set.sample(rng)
            j_train, ys_hat, optimiser_state = update(
                model, optimiser, refeed, k, optimiser_state, xs, ys
            )
            train_loss_epoch += j_train
            optimise.log_train(
                i,
                epochs,
                k,
                train_maxsteps,
                j_train,
                xs,
                ys_hat,
                ys,
                log_frequency,
                global_step,
            )

        #  validate
        params = optimiser.params(optimiser_state)
        _rng_val = rng_val
        for j in range(val_maxsteps):
            k = (val_maxsteps * i) + j
            global_step += 1
            _rng_val, _ = jax.random.split(_rng_val)
            xs, ys = val_set.sample(_rng_val)
            j_val, ys_hat = optimise.evaluate(model, refeed, params, xs, ys)
            optimise.log_val(
                i,
                epochs,
                k,
                val_maxsteps,
                j_val,
                xs,
                ys_hat,
                ys,
                log_frequency,
                global_step,
            )

        #  schedule
        train_loss_epoch /= train_maxsteps
        if (i != 0) and (train_loss_epoch <= hparams.increase_at) and (refeed < 20):
            refeed += 1
            train_set.increase_frames()
            val_set.increase_frames()
        wandb.log({"refeed": refeed, "epoch": i}, step=global_step)

        #  checkpoint
        with open("{}/{}.pickle".format(wandb.run.dir, i), "wb") as f:
            pickle.dump(params, f)

    return optimiser_state


if __name__ == "__main__":
    app.run(main)
