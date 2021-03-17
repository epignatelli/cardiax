import re
import os
import h5py
import logging
import numpy as onp
import jax
import jax.numpy as jnp
import asyncio


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

    def sample(self, rng):
        def _sample(i, t):
            sequence = self.files[i]
            states = jnp.array(
                sequence["states"][
                    t : t + (self.frames_in + self.frames_out) : self.step
                ]
            )
            diffusivity = jnp.array(sequence["diffusivity"])
            return states, diffusivity

        def collate(ss, ds):
            xs, ys = ss.split((self.frames_in,), axis=1)
            dd = jnp.tile(ds[:, None, None], (1, self.frames_in, 1, 1, 1))
            xs = jnp.concatenate([xs, dd], axis=-3)  # channel axis
            return xs, ys

        sample_idx = lambda rng, maxval: jax.random.randint(
            rng, (self.batch_size,), minval=0, maxval=maxval
        )
        rng_1, rng_2 = jax.random.split(rng, 2)
        ids = sample_idx(rng_1, self._n_sequences)
        starts = sample_idx(
            rng_2,
            self._sequence_len - (self.frames_in + self.frames_out) * self.step - 1,
        )

        batch, diffusivities = [], []
        for i in range(self.batch_size):
            b, d = _sample(ids[i], starts[i])
            batch.append(b)
            diffusivities.append(d)

        return collate(jnp.stack(batch), jnp.stack(diffusivities))

    def increase_frames(self):
        self.frames_out += 1
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
