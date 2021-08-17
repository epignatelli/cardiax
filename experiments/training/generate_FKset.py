import time
import os

import h5py
import jax
from absl import app, flags, logging
from termcolor import colored

import cardiax
import deepx


flags.DEFINE_string(
    "params",
    "3",
    "Paramset from the Fenton-Cherry 2002 paper. You can choose between [1A, 1B, 1C, 1D, 1E, 2, 3, 4A, 4B, 4C, , 5, 6, 7, 8, 9, 10]",
)
flags.DEFINE_list(
    "shape",
    [1200, 1200],
    "Shape of the field to simulate into. Sizes are in computational units. A computational unit is 1/100 of a cm",
)
flags.DEFINE_integer(
    "length",
    1000,
    "The length of the simulation in milliseconds",
)
flags.DEFINE_integer(
    "step",
    1,
    "Simulation states will be saved with a span of <step> measured in milliseconds",
)
flags.DEFINE_integer(
    "n_stimuli",
    3,
    "Number of random stimuli in the simulation. Stimuli occur each 400ms",
)
flags.DEFINE_string(
    "filepath",
    "$EPHEMERAL/data/verify_{}.hdf5",
    "Python forbattable string as filepath to save the simulation to. The simulation is saved in hdf5 format.",
)
flags.DEFINE_list(
    "reshape",
    [256, 256],
    "States will be resized according to this parameter",
)
flags.DEFINE_bool(
    "use_memory",
    True,
    "If true - faster option - simulation steps will be kept in memory and dumped to disk only at the end of the simulation.",
)
flags.DEFINE_integer(
    "n_sequences",
    100,
    "Number of random sequences to generate",
)
flags.DEFINE_integer(
    "start_seed",
    0,
    "Seed that start the sequence of random seeds",
)
flags.DEFINE_bool(
    "plot_while",
    False,
    "If true python will plot all the intermediate steps while runnning the simulation",
)
flags.DEFINE_bool(
    "export_videos",
    False,
    "If true sequences are also exported as mp4 videos",
)


FLAGS = flags.FLAGS


def main(argv):
    paramset = getattr(cardiax.params, "PARAMSET_{}".format(FLAGS.params))
    step = FLAGS.step
    stop = FLAGS.length
    start_seed = FLAGS.start_seed

    def make_hdf5(seed):
        rng = jax.random.PRNGKey(seed)
        return deepx.generate.random_sequence(
            rng=rng,
            params=paramset,
            filepath=FLAGS.filepath.format(seed),
            shape=tuple(FLAGS.shape),
            n_stimuli=FLAGS.n_stimuli,
            stop=stop,
            step=step,
            reshape=tuple(FLAGS.reshape),
            use_memory=FLAGS.use_memory,
            plot_while=FLAGS.plot_while,
        )

    def hdf5_to_mp4(filepath, fps=60):
        with h5py.File(filepath, "r") as f:
            sequence = f["states"]
            diffusivity = f["diffusivity"][:]
            states = [cardiax.solve.State(*x) for x in sequence]
            video = cardiax.plot.animate_state(states, diffusivity, figsize=(20, 5))
            video_filepath = os.path.splitext(filepath)[0] + ".mp4"
            video.save(video_filepath, writer="ffmpeg", fps=fps)
        return video

    n_sequences = FLAGS.n_sequences
    for i in range(n_sequences):
        seed = start_seed + i
        print(
            colored(
                "Generating sequence with random seed {}/{}".format(
                    seed, start_seed + n_sequences
                ),
                "red",
            )
        )
        start = time.time()
        make_hdf5(seed)
        logging.info(
            "Simulation {}/{} completed in {}s".format(
                seed, start_seed + n_sequences, time.time() - start
            )
        )

        if FLAGS.export_videos:
            start = time.time()
            filepath = FLAGS.filepath
            hdf5_to_mp4(filepath.format(seed), fps=60 / step)
            logging.info(
                "Conversion {}/{} completed in {}".format(
                    seed, start_seed + n_sequences, time.time() - start
                )
            )


if __name__ == "__main__":
    app.run(main)
