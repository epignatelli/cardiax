import logging
import os
import random
import re
from typing import Callable

import h5py
import numpy as onp
import jax
import cardiax


class DataStream:
    def __init__(self, dataset, batch_size, collate_fn, seed=None):
        # public:
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.seed = seed
        self.n_batches, _ = divmod(len(self.dataset), self.batch_size)
        self.indices = onp.array(range(self.n_batches * self.batch_size))

        # shuffling
        if seed is not None:
            rng = jax.random.PRNGKey(self.seed)
            self.indices = jax.random.permutation(rng, self.indices)
        # batching
        self.batch_indices = iter(
            onp.split(self.indices, self.n_batches)
        )  # List[onp.ndarray]

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self

    def __next__(self):
        indices = next(self.batch_indices)
        batch = [self.dataset[i] for i in indices]
        return self.collate_fn(batch)

    @property
    def frames_out(self):
        return self.dataset.frames_out

    @frames_out.setter
    def frames_out(self, value):
        self.dataset.frames_out = value
        self.reset()

    def increase_frames(self):
        self.frames_out += 1
        return True

    def reset(self):
        self.batch_indices = iter(onp.split(self.indices, self.n_batches))


class ConcatSequence:
    def __init__(
        self,
        root,
        frames_in=2,
        frames_out=5,
        step=5,
        keys=None,
        transform=None,
        preload=False,
        perc=1.0,
    ):
        # public:
        self.root = root

        # private:
        self._frames_in = frames_in
        self._frames_out = frames_out
        self._step = step

        if keys is None:
            keys = ".*"

        filenames = [
            os.path.join(root, name)
            for name in sorted(os.listdir(root))
            if re.search(keys, name)
        ]
        if len(filenames) == 0:
            raise FileNotFoundError(
                "No datasets found that match regex {} in {}".format(keys, root)
            )

        self.datasets = [
            HDF5Sequence(
                filename=filename,
                frames_in=frames_in,
                frames_out=frames_out,
                step=step,
                transform=transform,
                preload=preload,
                perc=perc,
            )
            for filename in filenames
        ]
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


class HDF5Sequence:
    def __init__(
        self,
        filename,
        frames_in=1,
        frames_out=0,
        step=1,
        transform=None,
        preload=False,
        perc=1.0,
    ):
        # public:
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.step = step
        self.transform = transform
        self.filename = filename
        self.preload = preload
        # private:
        self._is_open = False
        self._perc = perc

        self.states = None
        self.stimuli = None
        self.open()
        return

    def __getitem__(self, idx):
        # from idx of start to slice to take the sequence
        if isinstance(idx, slice):
            idx = slice(
                idx.start,
                idx.start + (self.frames_in + self.frames_out) * self.step,
                self.step,
            )
        elif isinstance(idx, onp.ndarray):
            idx = tuple(
                [
                    slice(
                        i, i + (self.frames_in + self.frames_out) * self.step, self.step
                    )
                    for i in idx
                ]
            )
        else:
            idx = slice(
                idx, idx + (self.frames_in + self.frames_out) * self.step, self.step
            )

        states = self.states[idx, :3]  # take only the first three channels

        if self.transform is not None:
            states = self.transform(states)

        return states

    def __len__(self):
        return int(
            (5000 - (self.frames_in + 20) * self.step) * self._perc
        )  # hacky, this sucks - TODO(epignatelli)

    def open(self):
        file = h5py.File(self.filename, "r")
        self.states = file["states"]
        self.stimuli = cardiax.io.load_stimuli(file)
        if self.preload:
            logging.debug("Preloading dataset {}".format(self.filename))
            if not isinstance(self.preload, Callable):
                self.preload = onp.array
            self.states = self.preload(self.states[:])
        self._is_open = True
        return


def imresize(image, size, method):
    shape = (*image.shape[:-2], *size)
    return jax.image.resize(image, shape, method)
