import os 
import numpy as np
import h5py
import torch
import fk
import random


class FkDataset():
    def __init__(self, root,
                 frames_in=5,
                 frames_out=10,
                 step=1,
                 keys=None,
                 transform=None,
                 squeeze=False,
                 preload=False):
        self.root = root
        self.squeeze = squeeze

        filenames = [os.path.join(root, name) for name in sorted(os.listdir(root)) if name.endswith("hdf5")]
        if keys is not None:
            filenames = [name for name in filenames 
                 if os.path.basename(name) in keys ]
        self.datasets = [Simulation(filename, frames_in, frames_out, step, transform, squeeze, preload) for filename in filenames]
                        
        # private
        self._frames_in = frames_in
        self._frames_out = frames_out
        self._step = step
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
                
    def close(self):
        for dataset in self.datasets:
            dataset.states.file.close()
    
class Simulation():
    def __init__(self, filename, frames_in=1, frames_out=0, step=1, transform=None, squeeze=True, preload=False):
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.step = step
        self.squeeze = squeeze
        self.transform = transform
        self.preload = preload
        self.filename = filename
        self.is_open = False
        return
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = slice(idx.start, idx.start + (self.frames_in + self.frames_out) * self.step, self.step)
            states = np.array(self.states[idx])
        else:
            states = np.array(self.states[idx: idx + (self.frames_in + self.frames_out) * self.step: self.step])
        
        if self.transform is not None:
            states = self.transform(states)
            
        if self.squeeze:
            states = states.squeeze()
        
        return states
    
    def __len__(self):
        # TODO(epignatelli): 2000 stands for len(self.states) and is hardcoded. Replace it with correct file shape
        return len(self.states) - (self.frames_in + self.frames_out) * self.step   
    
    @property
    def states(self):
        if not self.is_open:
            self.open()
        return self._states
    
    def open(self):
        file = h5py.File(self.filename, "r") 
        self._states = file["states_256"]
        if self.preload:
            self._states = self._states[:]
        self.is_open = True
        return
    
    def stimulus_at_t(self, t):
        stimulated = np.zeros(self.shape)
        for stimulus in self.stimuli:
            active = t >= stimulus["start"]
            active &= ((stimulus["start"] - t + 1) % stimulus["period"]) < stimulus["duration"]
            stimulated = torch.where(stimulus["field"] * (active) > 0, stimulus["field"], stimulated)
        return stimulated