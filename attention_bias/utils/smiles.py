import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import GetPeriodicTable

ptable = GetPeriodicTable()

def get_features(atom):
    symbol = atom.GetSymbol()
    return [
        ptable.GetAtomicNumber(symbol), 
        ptable.GetAtomicWeight(symbol), 
        ptable.GetNOuterElecs(symbol), 
        ptable.GetDefaultValence(symbol), 
        ptable.GetRvdw(symbol), 
        ptable.GetRcovalent(symbol), 
    ]

'''
def get_features(atom):
    return [
        atom.GetAtomicNum(), 
        atom.GetFormalCharge(), 
        atom.GetTotalNumHs(), 
        atom.GetNumRadicalElectrons(), 
    ]
'''

def featurize_smiles(smiles, device='cpu'):

    features_list = []
    padding_list = []
    
    for smile in smiles:
        mol = MolFromSmiles(smile)

        # Featurize Nodes
        features = torch.tensor([
            get_features(atom) for atom in mol.GetAtoms()
            if atom.GetSymbol() != "H"
        ])
        features_list.append(features)

        # Construct padding indicator
        padding = torch.zeros(features.shape[0], dtype=torch.bool)
        padding_list.append(padding)

    # Max no. of tokens

    x = pad_sequence(
        features_list, 
        batch_first=True, padding_value=0, padding_side='right'
    )

    paddings = pad_sequence(
        padding_list, 
        batch_first=True, padding_value=True, padding_side='right'
    )

    return x, paddings

def collate_smiles(batch):
    '''
    Collate a DataLoader outputs corresponding to point clouds.

    Args:
        batch: List(Tuple(str), Array(B, 3), Array(B))

    Returns:
        Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    '''

    smiles_batch, conformers_batch, y_batch = zip(*batch)

    features, padding = featurize_smiles(smiles_batch)
    conformers = torch.stack(conformers_batch)[:, :features.shape[1], :]
    y = torch.stack(y_batch)

    return features, padding, conformers, y
