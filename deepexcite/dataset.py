from pytorch_lightning import seed_everything
seed_everything(2)
import os
import random
import re
import numpy as np
import h5py
import torch
import fenton_karma as fk
import os
from os import path
import struct
import array
import jax.numpy as jnp
import gzip
import urllib


class ConcatSequence():
    def __init__(self,
                 root,
                 frames_in=5,
                 frames_out=10,
                 step=1,
                 keys=None,
                 transform=None,
                 preload=False,
                 clean_from_stimuli=False):
        # public:
        self.root = root

        # private:
        self._frames_in = frames_in
        self._frames_out = frames_out
        self._step = step

        if keys is None:
            keys = ".*"

        filenames = [os.path.join(root, name) for name in sorted(os.listdir(root)) if re.search(keys, name)]
        if len(filenames) == 0:
            raise FileNotFoundError("No datasets found that match regex {} in {}".format(keys, root))

        self.datasets = [HDF5Sequence(filename,
                                      frames_in,
                                      frames_out,
                                      step,
                                      transform,
                                      preload,
                                      clean_from_stimuli) for filename in filenames]
        return

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        dataset = random.choice(self.datasets)
        sample = dataset[idx]
        return sample

    @property
    def frames_in(self):
        return self._frames_in

    @frames_in.setter
    def frames_int(self, value):
        self._frames_in = value
        for i in range(len(self.datasets)):
            self.datasets[i].frames_in = value

    @property
    def frames_out(self):
        return self._frames_out

    @frames_out.setter
    def frames_out(self, value):
        self._frames_out = value
        for i in range(len(self.datasets)):
            self.datasets[i].frames_out = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value
        for i in range(len(self.datasets)):
            self.datasets[i].step = value


class HDF5Sequence():
    def __init__(self,
                 filename,
                 frames_in=1,
                 frames_out=0,
                 step=1,
                 transform=None,
                 preload=False,
                 clean_from_stimuli=False):
        # public:
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.step = step
        self.transform = transform
        self.filename = filename
        self.preload = preload
        self.states = None
        self.clean_from_stimuli = clean_from_stimuli

        # private:
        self._is_open = False
        self._tentatives = 0
        return

    def __getitem__(self, idx):
        if not self._is_open:
            self.open()
        if isinstance(idx, slice):
            idx = slice(idx.start, idx.start + (self.frames_in + self.frames_out) * self.step, self.step)
        else:
            idx = slice(idx, idx + (self.frames_in + self.frames_out) * self.step, self.step)

        if self.clean_from_stimuli and self.is_stimulated_within(idx.start, idx.stop):
            if self._tentatives < 30:
                self._tentatives += 1
                return self[idx.stop]
            else:
                print("Spent {} tentatives to acquire data without stimulus, but failed. The sample may contain stimuli.")

        states = np.array(self.states[idx])

        if self.transform is not None:
            states = self.transform(states)

        self._tentatives = 0
        return states

    def __len__(self):
        return 5000 - (self.frames_in + 20) * self.step  # hacky, this sucks - TODO(epignatelli)

    def open(self):
        file = h5py.File(self.filename, "r")
        self.states = file["states"]
        self.stimuli = fk.io.load_stimuli(file)
        if self.preload:
            self.states = self.states[:]
        self._is_open = True
        return

    def is_stimulated_within(self, start, end):
        for stimulus in self.stimuli:
            stim_start = stimulus["start"]
            stim_end = stim_start + stimulus["duration"]
            if start <= stim_start <= end or start <= stim_end <= end:
                return True
        return False

    def stimulus_at_t(self, t):
        stimulated = np.zeros(self.states.shape)
        for stimulus in self.stimuli:
            active = t >= stimulus["start"]
            active &= ((stimulus["start"] - t + 1) % stimulus["period"]) < stimulus["duration"]
            stimulated = torch.where(stimulus["field"] * (active) > 0, stimulus["field"], stimulated)
        return stimulated


def MnistDataset():
    _DATA = "/tmp/jax_example_data/"
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def _download(url, filename):
        """Download a url to a file in the JAX data temp directory."""
        if not path.exists(_DATA):
            os.makedirs(_DATA)
        out_file = path.join(_DATA, filename)
        if not path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)
            print("downloaded {} to {}".format(url, _DATA))

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(num_data, rows, cols)

    for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels