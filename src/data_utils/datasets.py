import pypianoroll
import torch
import torch.utils.data
import numpy as np
import os

class MultiTrackPianoRollDataset(torch.utils.data.Dataset):
    """Piano Roll dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the pianorolls.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.piano_rolls = []
        self._populate_filenames()
        
                    
        
        self.transform = transform
        
    def _populate_filenames(self):
        for root, _, files in os.walk(self.root_dir, topdown=False):
            for name in files:
                if len(name) > 3 and name[-4:] == '.npz':
                    self.piano_rolls.append(os.path.join(root,name))
                    
    def __len__(self):
        return len(self.piano_rolls)

    def __getitem__(self, idx):
        
        sample = self.piano_rolls[idx]
        sample = pypianoroll.Multitrack(sample)
        
        if self.transform:
            sample = self.transform(sample)

        return sample