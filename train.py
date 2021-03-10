import logging
import os
import pickle
import sys
from functools import partial

import jax
import jax.numpy as jnp
import wandb
from absl import app, flags
from helx.types import Optimiser
from jax.experimental import optimizers

import cardiax
from deepx import resnet
from deepx.dataset import Dataset, Paramset5Dataset

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
flags.DEFINE_integer("batch_size", 4, "")
flags.DEFINE_integer("evaluation_steps", 20, "")
flags.DEFINE_integer("epochs", 100, "")
flags.DEFINE_float("increase_at", 0.0, "")
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
    #  unroll hparams
    hparams = resnet.HParams(
        seed=FLAGS.seed,
        log_frequency=FLAGS.log_frequency,
        debug=FLAGS.debug,
        hidden_channels=FLAGS.hidden_channels,
        in_channels=FLAGS.in_channels,
        depth=FLAGS.depth,
        lr=FLAGS.lr,
        batch_size=FLAGS.batch_size,
        evaluation_steps=FLAGS.evaluation_steps,
        epochs=FLAGS.epochs,
        increase_at=FLAGS.increase_at,
        teacher_forcing_prob=FLAGS.teacher_forcing_prob,
        from_checkpoint=FLAGS.from_checkpoint,
        root=FLAGS.root,
        paramset=FLAGS.paramset,
        size=tuple(FLAGS.size),
        frames_in=FLAGS.frames_in,
        frames_out=FLAGS.frames_out,
        step=FLAGS.step,
        refeed=FLAGS.refeed,
        preload=FLAGS.preload,
    )

    # set logging level
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if hparams.debug else logging.INFO
    )
    # save hparams
    list(
        map(
            lambda i: setattr(wandb.config, hparams._fields[i], hparams[i]),
            list(range(len(hparams))),
        )
    )
    n_sequence_out = hparams.refeed * hparams.frames_out

    # get data streams
    input_shape = (
        hparams.batch_size,
        hparams.frames_in,
        4,
        *hparams.size,
    )
    logging.debug("Input shape is : {}".format(input_shape))
    make_dataset = lambda subdir: Dataset(
        folder=os.path.join(hparams.root, subdir),
        frames_in=hparams.frames_in,
        frames_out=n_sequence_out,
        step=hparams.step,
        batch_size=hparams.batch_size,
    )
    train_set = make_dataset("train")
    val_set = make_dataset("val")

    # init model
    rng = jax.random.PRNGKey(hparams.seed)
    network = resnet.ResNet(
        hidden_channels=hparams.hidden_channels,
        out_channels=hparams.frames_out,
        depth=hparams.depth,
    )
    if hparams.from_checkpoint is not None and os.path.exists(hparams.from_checkpoint):
        with open(hparams.from_checkpoint, "rb") as f:
            params = pickle.load(f)
    else:
        _, params = network.init(rng, input_shape)

    # init optimiser
    optimiser = Optimiser(*optimizers.adam(hparams.lr))
    optimiser_state = optimiser.init(params)
    train_iteration = 0
    val_iteration = 0
    refeed = hparams.refeed
    for i in range(hparams.epochs):
        ## TRAINING
        train_loss_epoch = 0.0
        for j in range(train_set.num_batches()):
            # learning
            rng, _ = jax.random.split(rng)
            xs, ys = train_set.sample(rng)
            j_train, ys_hat, optimiser_state = resnet.tbtt_step(
                network,
                optimiser,
                refeed,
                train_iteration,
                optimiser_state,
                xs,
                ys,
            )
            train_loss_epoch += j_train

            # logging
            resnet.log_train(
                j_train, xs, ys_hat, ys, train_iteration, hparams.log_frequency
            )

            # prepare next iteration
            print(
                "Epoch {}/{} - Training step {}/{} - Loss: {:.6f}\t\t\t".format(
                    i, hparams.epochs, j, train_set.num_batches(), j_train
                ),
                end="\r",
            )
            train_iteration = train_iteration + 1
            if j == 10 and hparams.debug:
                break

        train_loss_epoch /= len(train_set)

        ## VALIDATING
        val_loss_epoch = 0.0
        params = optimiser.params(optimiser_state)
        for j in enumerate(val_set.num_batches()):
            # prepare data
            rng, _ = jax.random.split(rng)
            xs, ys = train_set.sample(rng)

            # learning
            j_val, ys_hat = resnet.evaluate(
                network, hparams.evaluation_steps, params, xs, ys
            )
            val_loss_epoch += j_val

            # logging
            resnet.log_val(val_loss_epoch, xs, ys_hat, ys, val_iteration, val_iteration)

            # prepare next iteration
            print(
                "Epoch {}/{} - Evaluation step {}/{} - Loss: {:.6f}\t\t\t".format(
                    i, hparams.epochs, j, val_set.num_batches(), j_val
                ),
                end="\r",
            )
            val_iteration = val_iteration + 1
            if j == 10 and hparams.debug:
                break

        val_loss_epoch /= len(val_set)

        ## SCHEDULED UPDATES
        if (i != 0) and (train_loss_epoch <= hparams.increase_at) and (refeed < 20):
            train_set.increase_frames()
            val_set.increase_frames()
            logging.info(
                "Increasing the amount of output frames to {} \t\t\t".format(refeed)
            )
        wandb.log("refeed", refeed, train_iteration)

    return optimiser_state


if __name__ == "__main__":
    app.run(main)
