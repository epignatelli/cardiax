import logging
import os
import re

import h5py
import jax
import numpy as onp


class Dataset:
    def __init__(
        self,
        folder,
        frames_in,
        frames_out,
        step,
        batch_size=1,
        re_key=".*",
    ):
        #  public:
        self.folder = folder
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.step = step
        self.batch_size = batch_size
        self.re_key = re_key
        self.files = [h5py.File(filepath, "r") for filepath in self._filepaths()]
        self.n_devices = jax.local_device_count()

        #  private:
        self._n_sequences = len(self.files)
        self._sequence_len = len(self.files[0]["states"])
        self.indices = []
        self._reset_indices()

    def __len__(self):
        return self._n_sequences * self._sequence_len

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def num_batches(self):
        return len(self) // self.batch_size

    def split_for_devices(self, arr):
        return arr.reshape(
            self.n_devices, arr.shape[0] // self.n_devices, *arr.shape[1:]
        )

    def sample(self, rng):
        def _sample(i, t):
            sequence = self.files[i]
            states = onp.array(
                sequence["states"][
                    t : t + (self.frames_in + self.frames_out) * self.step : self.step
                ]
            )
            diffusivity = onp.array(sequence["diffusivity"])
            return states, diffusivity

        def collate(ss, ds):
            xs, ys = onp.split(ss, (self.frames_in,), axis=1)
            dd = onp.tile(ds[:, None, None], (1, self.frames_in, 1, 1, 1))
            xs = onp.concatenate([xs, dd], axis=-3)  # channel axis
            xs = self.split_for_devices(xs)
            ys = self.split_for_devices(ys)
            return xs, ys

        sample_idx = lambda rng, maxval: jax.random.randint(
            rng, (self.batch_size,), minval=0, maxval=maxval
        )
        rng_1, rng_2 = jax.random.split(rng, 2)
        ids = sample_idx(rng_1, self._n_sequences)
        starts = sample_idx(
            rng_2,
            self._sequence_len - (self.frames_in + self.frames_out) * self.step,
        )
        batch, diffusivities = [], []
        for i in range(self.batch_size):
            b, d = _sample(ids[i], starts[i])
            batch.append(b)
            diffusivities.append(d)
        xs, ys = collate(onp.stack(batch), onp.stack(diffusivities))
        return xs, ys

    def increase_frames(self, n=1):
        self.frames_out += n
        logging.info(
            "Increasing the amount of output frames to {} \t\t\t".format(
                self.frames_out
            )
        )
        self._reset_indices()

    def _reset_indices(self):
        ts = onp.arange(0, len(self.files[0]["states"]))
        idx = onp.stack([ts + len(ts) * i for i in range(len(self.files))])
        stop = (self.frames_in + self.frames_out) * self.step
        self.indices = idx[:, :-stop]

    def _filepaths(self):
        filepaths = [
            os.path.join(self.folder, name)
            for name in sorted(os.listdir(self.folder))
            if re.search(self.re_key, name)
        ]
        if len(filepaths) == 0:
            raise FileNotFoundError(
                "No datasets found that match regex {} in {}".format(
                    self.re_key, self.folder
                )
            )
        return filepaths


class Paramset5Dataset(Dataset):
    def __init__(self, folder, frames_in, frames_out, step, batch_size=1):
        super().__init__(
            folder,
            frames_in,
            frames_out,
            step,
            batch_size,
            re_key=".*paramset5.*/i",
        )
