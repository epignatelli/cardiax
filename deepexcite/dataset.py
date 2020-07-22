import os 
import numpy as np
import h5py
import torch
import fk
import random


class FkDataset():
    def __init__(self, root, n_frames_in=5, n_frames_out=10, step=1,
                 keys=None, transform=None, squeeze=False):
        self.root = root
        self.n_frames_in = n_frames_in
        self.n_frames_out = n_frames_out
        self.n_frames = n_frames_in + n_frames_out
        self.step = step
        self.squeeze = squeeze
        
        filenames = [os.path.join(root, name) for name in sorted(os.listdir(root)) if name.endswith("hdf5")]
        if keys is not None:
            filenames = [name for name in filenames 
                 if os.path.basename(name) in keys ]
        self.datasets = [Simulation(filename, n_frames_in, n_frames_out, step, transform, squeeze) for filename in filenames]
        

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        dataset = random.choice(self.datasets)
        sample = dataset[idx]

        return sample

    def close(self):
        for dataset in self.datasets:
            dataset.states.file.close()
    
class Simulation():
    def __init__(self, filename, in_frames=1, out_frames=0, step=1, transform=None, squeeze=True):
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.step = step
        self.squeeze = squeeze
        self.transform = transform
        
        self.filename = filename
        self.open = False

    
    def __getitem__(self, idx):
        if not self.open:
            self.open_dataset()
            
        if isinstance(idx, slice):
            idx = slice(idx.start, idx.start + (self.in_frames + self.out_frames) * self.step, self.step)
            states = np.array(self.states[idx])
        else:
            states = np.array(self.states[idx: idx + (self.in_frames + self.out_frames) * self.step: self.step])
        
        if self.transform is not None:
            states = self.transform(states)
            
        if self.squeeze:
            states = states.squeeze()
        
        return states
    
    def __len__(self):
        if not self.open:
            self.open_dataset()
        return len(self.states) - (self.in_frames + self.out_frames) * self.step
    
    def open_dataset(self):
        self.open = True
        
        file = h5py.File(self.filename, "r") 
        self.states = file["states_256"]
        
        self.stimuli = fk.io.load_stimuli(file)
        for i in range(len(self.stimuli)):
            self.stimuli[i]["field"] = np.array(self.stimuli[i]["field"])
        self.shape = self.states.shape[-2:]
    
    def stimulus_at_t(self, t):
        stimulated = np.zeros(self.shape)
        for stimulus in self.stimuli:
            active = t >= stimulus["start"]
            active &= ((stimulus["start"] - t + 1) % stimulus["period"]) < stimulus["duration"]
            stimulated = torch.where(stimulus["field"] * (active) > 0, stimulus["field"], stimulated)
        return stimulated