import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from rdkit.Chem import MolFromSmiles

from rdkit import Chem
import torch

from rdkit import Chem
import torch

token_indices = {c: i for i, c in enumerate('HCNOF', start=1)}

def get_mol_tokens(mol):
    return torch.tensor([
        token_indices[atom.GetSymbol()] for atom in mol.GetAtoms()
    ], dtype=torch.int)

def get_mol_adj_list(mol):
    '''
    Generate adjacency list without self-loops
    '''
    
    adj_list = torch.tensor([
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    ], dtype=torch.int)

    if adj_list.size(0) == 0:
        return torch.empty((0, 2), dtype=torch.int)
    
    return torch.concat([
        adj_list, adj_list.flip(dims=(1, ))
    ])

def tokenize_smiles(smiles, device='cpu'):
    '''
    Generate padded tokens and adjacency matrices for SMILES
    '''

    tokens = []
    adj_lists = []
    
    for smile in smiles:
        mol = Chem.AddHs(MolFromSmiles(smile))
        Chem.Kekulize(mol, clearAromaticFlags=True)

        tokens.append(get_mol_tokens(mol))
        adj_lists.append(get_mol_adj_list(mol))

    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    padding = (padded_tokens == 0)

    n_tokens = padding.shape[-1]
    padded_adj_mats = torch.stack([
        torch.sparse_coo_tensor(
            adj_list.T, 
            torch.ones(adj_list.shape[0], dtype=torch.bool),
            (n_tokens, n_tokens)
        ).to_dense()
        for adj_list in adj_lists
    ])

    return padded_tokens, padding, padded_adj_mats

def collate_smiles(batch):
    '''
    Collate SMILES data
    '''

    smiles_batch, y_batch = zip(*batch)
    
    tokens, padding, adj_mats = tokenize_smiles(smiles_batch)
    y = torch.stack(y_batch)

    return tokens, padding, adj_mats, y
