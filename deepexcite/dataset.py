import os 
import h5py
import torch
import fk


class FkDataset():
    def __init__(self, root, n_frames_in=5, n_frames_out=10, step=1, keys=None):
        self.root = root
        self.n_frames_in = n_frames_in
        self.n_frames_out = n_frames_out
        self.n_frames = n_frames_in + n_frames_out
        self.step = step
        
        filenames = [os.path.join(root, name) for name in sorted(os.listdir(root))]
        if keys is not None:
            filenames = [name for name in filenames 
                 if os.path.basename(name) in keys ]
        self.datasets = [Simulation(filename) for filename in filenames]
        self.cumulative_sizes = self.cumsum(self.datasets)
        
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = torch.utils.data.dataset.bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        start = sample_idx
        end = start + self.n_frames_in + self.n_frames_out
        if end > len(self):
            return None
        return self.datasets[dataset_idx][start:end:self.step]

    def close(self):
        for dataset in self.datasets:
            dataset.states.file.close()
    
class Simulation():
    def __init__(self, filename):
        self.filename = filename
        file = h5py.File(filename, "r")
        self.states = file["states"]
        self.stimuli = fk.io.load_stimuli(file)
        for i in range(len(self.stimuli)):
            self.stimuli[i]["field"] = torch.as_tensor(self.stimuli[i]["field"])
        self.shape = self.states.shape[-2:]
    
    def __getitem__(self, idx):
        states = self.states[idx]
        unstimulated = torch.zeros(self.states.shape[-2:])
        stimuli = torch.stack([self.stimulus_at_t(t) for t in range(idx.start, idx.stop, idx.step)])
        return states, stimuli
    
    def __len__(self):
        return len(self.states)
    
    def stimulus_at_t(self, t):
        stimulated = torch.zeros(self.shape)
        for stimulus in self.stimuli:
            active = t >= stimulus["start"]
            active &= ((stimulus["start"] - t + 1) % stimulus["period"]) < stimulus["duration"]
            stimulated = torch.where(stimulus["field"] * (active) > 0, stimulus["field"], stimulated)
        return stimulated