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
flags.DEFINE_float("lr", 0.001, "")
flags.DEFINE_float("grad_norm", 1.0, "")
flags.DEFINE_bool("normalise", False, "")
flags.DEFINE_integer("batch_size", 4, "")
flags.DEFINE_float("lamb", 0.05, "")
flags.DEFINE_integer("evaluation_steps", 20, "")
flags.DEFINE_integer("epochs", 100, "")
flags.DEFINE_integer("train_maxsteps", 100, "")
flags.DEFINE_integer("val_maxsteps", 10, "")
flags.DEFINE_bool("tbtt", False, "")
flags.DEFINE_float("increase_at", 0.01, "")
flags.DEFINE_float("teacher_forcing_prob", 0.0, "")
flags.DEFINE_string("from_checkpoint", "", "")

#  input data arguments
flags.DEFINE_string("root", "/rds/general/user/sg6513/ephemeral/data/", "")
flags.DEFINE_string("paramset", "paramset5", "")
flags.DEFINE_list("size", [256, 256], "")
flags.DEFINE_integer("frames_in", 2, "")
flags.DEFINE_integer("frames_out", 1, "")
flags.DEFINE_integer("step", 1, "")
flags.DEFINE_integer("refeed", 5, "")
flags.DEFINE_integer("test_refeed", 20, "")
flags.DEFINE_boolean("preload", False, "")

FLAGS = flags.FLAGS


def main(argv):
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if FLAGS.debug else logging.INFO
    )
    #  hparms
    logging.info("Parsing hyperparamers and initialising logger...")
    hparams = resnet.HParams.from_flags(FLAGS)
    train_maxsteps = hparams.train_maxsteps if not hparams.debug else 1
    val_maxsteps = hparams.val_maxsteps if not hparams.debug else 1
    log_frequency = hparams.log_frequency
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
    make_dataset = lambda subdir, n: Dataset(
        folder=os.path.join(hparams.root, subdir),
        frames_in=hparams.frames_in,
        frames_out=n,
        step=hparams.step,
        batch_size=hparams.batch_size,
    )
    train_set = make_dataset("train", n_sequence_out)
    val_set = make_dataset("val", n_sequence_out)
    test_set = make_dataset("val", hparams.test_refeed)

    #  init model
    logging.info("Initialising model...")
    rng = jax.random.PRNGKey(hparams.seed)
    rng_val = jax.random.PRNGKey(hparams.seed)
    model = resnet.ResNet(
        hidden_channels=hparams.hidden_channels,
        out_channels=hparams.frames_out,
        depth=hparams.depth,
    )

    #  init optimiser
    logging.info("Initialising optimisers...")
    _, params = model.init(rng, input_shape)
    optimiser = Optimiser(*optimizers.adam(hparams.lr))
    if hparams.from_checkpoint not in ("", None):
        train_state = optimise.TrainState.load(hparams.from_checkpoint)
    else:
        train_state = optimise.TrainState(rng, 0, params, hparams)
    opt_state = optimiser.init(params)

    #  training
    logging.info("Starting training...")
    global_step = 0
    update = optimise.tbtt_step if hparams.tbtt else optimise.btt_step
    for i in range(hparams.epochs):
        #  train
        train_loss_epoch = 0.0
        for j in range(train_maxsteps):
            k = (train_maxsteps * i) + j
            global_step += 1
            rng, _ = jax.random.split(rng)
            batch = train_set.sample(rng)
            xs, ys = optimise.preprocess(batch) if hparams.normalise else batch
            j_train, ys_hat, opt_state = update(
                model, optimiser, hparams.refeed, k, opt_state, xs, ys
            )
            j_train = j_train[0]  #  remove device axis - loss is returned synchronised
            train_state = optimise.TrainState(rng, global_step, opt_state, hparams)
            train_loss_epoch += j_train
            optimise.log_train(
                i,
                hparams.epochs,
                k,
                train_maxsteps,
                j_train,
                xs,
                ys_hat,
                ys,
                log_frequency,
                global_step,
                train_state,
            )

        #  validate
        params = optimiser.params(opt_state)
        _rng_val = rng_val
        for j in range(val_maxsteps):
            k = (val_maxsteps * i) + j
            global_step += 1
            _rng_val, _ = jax.random.split(_rng_val)
            batch = val_set.sample(_rng_val)
            xs, ys = optimise.preprocess(batch) if hparams.normalise else batch
            j_val, ys_hat = optimise.evaluate(model, hparams.refeed, params, xs, ys)
            j_val = j_val[0]  #  remove device axis - loss is returned synchronised
            optimise.log_val(
                i,
                hparams.epochs,
                k,
                val_maxsteps,
                j_val,
                xs,
                ys_hat,
                ys,
                log_frequency,
                global_step,
            )

            #  test
            batch = test_set.sample(_rng_val)
            xs, ys = optimise.preprocess(batch) if hparams.normalise else batch
            j_test, ys_hat = optimise.evaluate(
                model, hparams.test_refeed, params, xs, ys
            )
            j_test = j_test[0]  #  remove device axis - loss is returned synchronised
            optimise.log_test(
                i,
                hparams.epochs,
                k,
                val_maxsteps,
                j_test,
                xs,
                ys_hat,
                ys,
                log_frequency,
                global_step,
            )

        #  schedule
        train_loss_epoch /= train_maxsteps
        if (
            (i != 0)
            and (train_loss_epoch <= hparams.increase_at)
            and (hparams.refeed < 20)
        ):
            hparams = hparams._replace(refeed=hparams.refeed + 1)
            train_set.increase_frames()
            val_set.increase_frames()
        wandb.log({"refeed": hparams.refeed, "epoch": i}, step=global_step)

        #  checkpoint
        with open("{}/{}.pickle".format(wandb.run.dir, i), "wb") as f:
            pickle.dump(params, f)

    return opt_state


if __name__ == "__main__":
    app.run(main)
