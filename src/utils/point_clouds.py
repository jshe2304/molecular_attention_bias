import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from rdkit.Chem import GetPeriodicTable

ptable = GetPeriodicTable()

def get_features(symbol):
    return [
        ptable.GetAtomicNumber(symbol), 
        ptable.GetAtomicWeight(symbol), 
        ptable.GetNOuterElecs(symbol), 
        ptable.GetDefaultValence(symbol), 
        ptable.GetRvdw(symbol), 
        ptable.GetRcovalent(symbol), 
    ]

def featurize_atoms(atoms_arr, device='cpu'):
    '''
    Featurize a list of atoms corresponding to a molecule.

    Args:
        atoms: List[str]

    Returns:
        List[List[float]]
    '''
    
    features_list = [
        torch.tensor([get_features(atom) for atom in atoms], device=device)
        for atoms in atoms_arr
    ]
    features_tensor = pad_sequence(features_list, batch_first=True, padding_value=0)

    padding_list = [
        torch.zeros(len(atoms), dtype=torch.bool, device=device)
        for atoms in atoms_arr
    ]
    padding_tensor = pad_sequence(padding_list, batch_first=True, padding_value=True)

    return features_tensor, padding_tensor

def collate_point_clouds(batch):
    '''
    Collate a DataLoader outputs corresponding to point clouds.

    Args:
        batch: List(Tuple(str), Array(B, 3), Array(B))

    Returns:
        Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    '''

    atoms_batch, coordinates_batch, y_batch = zip(*batch)

    features, padding = featurize_atoms(atoms_batch)
    coordinates = torch.stack(coordinates_batch)[:, :features.shape[1], :]
    y = torch.stack(y_batch)

    return features, padding, coordinates, y
