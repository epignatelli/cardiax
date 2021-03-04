from absl import app
from absl import flags
import os
import h5py
import jax
import cardiax
import deepx
from termcolor import colored
import argparse


flags.DEFINE_string("cuda_visible_devices", "1", "")
flags.DEFINE_string("params", "3", "")
flags.DEFINE_list("shape", [1200, 1200], "")
flags.DEFINE_integer("n_stimuli", 3, "")
flags.DEFINE_string("filepath", "data/verify_{}.hdf5", "")
flags.DEFINE_list("reshape", [256, 256], "")
flags.DEFINE_integer("save_interval_ms", 1, "")
flags.DEFINE_bool("use_memory", True, "")
flags.DEFINE_integer("n_sequences", 10, "")
flags.DEFINE_bool("plot_while", False, "")
flags.DEFINE_bool("export_videos", False, "")
FLAGS = flags.FLAGS


def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    paramset = getattr(cardiax.params, "PARAMSET_{}".format(FLAGS.params))

    def make_hdf5(seed):
        rng = jax.random.PRNGKey(seed)
        return deepx.generate.random_sequence(
            rng,
            paramset,
            FLAGS.filepath.format(seed),
            tuple(FLAGS.shape),
            FLAGS.n_stimuli,
            reshape=tuple(FLAGS.reshape),
            use_memory=FLAGS.use_memory,
            plot_while=FLAGS.plot_while,
        )

    def hdf5_to_mp4(filepath, fps=24):
        with h5py.File(filepath, "r") as f:
            sequence = f["states"]
            states = [cardiax.solve.State(*x) for x in sequence]
            video = cardiax.plot.animate_state(states, figsize=(20, 5))
            video_filepath = os.path.splitext(filepath)[0] + ".mp4"
            video.save(video_filepath, writer="ffmpeg", fps=fps)
        return video

    n_sequences = FLAGS.n_sequences
    for i in range(n_sequences):
        print(
            colored(
                "Generating sequence with random seed {}/{}".format(i, n_sequences - 1),
                "red",
            )
        )
        make_hdf5(i)

        if FLAGS.export_videos:
            filepath = FLAGS.filepath
            hdf5_to_mp4(filepath.format(i))


if __name__ == "__main__":
    app.run(main)