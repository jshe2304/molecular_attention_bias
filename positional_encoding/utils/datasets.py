import numpy as np
import torch
from torch.utils.data import Dataset

class NPYDataset(Dataset):
    def __init__(self, atoms_file, y_file):
        self.atoms = np.load(atoms_file, allow_pickle=True)
        self.y = torch.tensor(np.load(y_file, allow_pickle=True))
        self.n_properties = self.y.shape[1]

    def __len__(self):
        return len(self.atoms)
    
    def __getitem__(self, idx):
        return self.atoms[idx], self.y[idx]
