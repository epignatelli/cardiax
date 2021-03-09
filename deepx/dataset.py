import re
import os
import h5py
import logging
import numpy as onp
import jax
import jax.numpy as jnp


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
        self.files = [h5py.File(filepath, "r") for filepath in self.filepaths()]

        #  private:
        self._n_sequences = len(self.files)
        self._sequence_len = len(self.files[0]["states"])
        self._reset_indices()

    def __len__(self):
        return self._n_sequences * self._sequence_len

    def __getitem__(self, idx):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __del__(self):
        for file in self.files:
            file.close()

    def num_batches(self):
        return len(self) // self.batch_size

    def sample(self, rng):
        def _sample(i, t):
            sequence = self.files[i]
            batch = jnp.array(
                sequence["states"][
                    t : t + (self.frames_in + self.frames_out) : self.step
                ]
            )
            xs, ys = batch.split((self.frames_in, self.frames_out))
            diffusivity = jnp.array((sequence["diffusivity"],) * self.frames_in)
            xs = jnp.stack(
                [xs, diffusivity], axis=1
            )  # stacking D as a channel (v, w, u, D)
            return xs, ys

        rng_1, rng_2 = jax.random.split(rng, 2)
        ids = jax.random.randint(rng_1, (self.batch_size,), maxval=self._n_sequences)
        starts = jax.random.randint(
            rng_2, (self.batch_size,), maxval=self._sequence_len
        )
        # ids = self._rng.randint(self._n_sequences, size=(self.batch_size))
        # starts = self._rng.randint(self._sequence_len, size=(self.batch_size))
        return jax.vmap(_sample)(
            jax.random.split(rng_2, self.batch_size), list(zip(ids, starts))
        )

    def increase_frames(self):
        self.frame_out += 1
        self._reset_indices()

    def _reset_indices(self):
        ts = onp.arange(0, len(self.files[0]["states"]))
        idx = onp.stack([ts + len(ts) * i for i in range(len(self.files))])
        stop = (self.frames_in + self.frame_out) * self.step
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
    def __init__(self, folder, frames_in, frames_out, step, batch_size=1, seed=0):
        super().__init__(
            folder,
            frames_in,
            frames_out,
            step,
            batch_size,
            re_key=".*paramset5.*/i",
            seed=seed,
        )
