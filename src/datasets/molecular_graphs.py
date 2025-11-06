import os

import numpy as np

import torch
from torch.utils.data import Dataset, default_collate
from torch.nn.utils.rnn import pad_sequence

from rdkit.Chem import MolFromSmiles
from rdkit import Chem

class MolecularGraphDataset(Dataset):
    def __init__(self, data_dir, target_labels):
        """
        Initialize a molecular graph dataset.
        
        Args:
            data_dir: Path to a directory containing the data.
            target_labels: Which properties to include.
        """

        # Make data paths

        atomic_numbers_file = os.path.join(data_dir, 'atomic_numbers.npy')
        smiles_file = os.path.join(data_dir, 'smiles.npy')
        y_file = os.path.join(data_dir, 'y.npy')
        y_labels_file = os.path.join(data_dir, 'y_labels.npy')
        y_mean_file = os.path.join(data_dir, 'y_mean.npy')
        y_std_file = os.path.join(data_dir, 'y_std.npy')

        # Load data

        print(target_labels)

        atomic_numbers = np.load(atomic_numbers_file)
        self.smiles_arr = np.load(smiles_file, allow_pickle=True)
        self.tokens, self.adj_matrices = self._tokenize_smiles(self.smiles_arr, atomic_numbers)
        self.padding = (self.tokens == 0)
        self.y_labels = np.load(y_labels_file, allow_pickle=True)
        self.y = torch.tensor(np.load(y_file), dtype=torch.float32) # (n_samples, n_properties)
        self.y_mean = torch.tensor(np.load(y_mean_file), dtype=torch.float32).unsqueeze(0) # (1, n_properties)
        self.y_std = torch.tensor(np.load(y_std_file), dtype=torch.float32).unsqueeze(0) # (1, n_properties)

        # Select desired properties

        label_mask = np.in1d(self.y_labels, target_labels)
        label_indices_np = label_mask.nonzero()[0]
        self.y_labels = self.y_labels[label_indices_np]
        label_indices = torch.as_tensor(label_indices_np, dtype=torch.long)
        self.y = self.y.index_select(1, label_indices)
        self.y_mean = self.y_mean.index_select(1, label_indices)
        self.y_std = self.y_std.index_select(1, label_indices)
        
        # Normalize targets

        self.y = (self.y - self.y_mean) / self.y_std

        # Check that data is consistent

        assert self.tokens.shape[0] == self.adj_matrices.shape[0] == self.y.shape[0] # Number of samples
        assert self.tokens.shape[1] == self.adj_matrices.shape[1] == self.adj_matrices.shape[2] # Number of atoms
        assert self.y.shape[1] == self.y_mean.shape[1] == self.y_std.shape[1] # Number of properties
    
    @staticmethod
    def _tokenize_mol(mol, token_indices):
        """
        Tokenize a molecule.
        """
        return torch.tensor([
            token_indices[atom.GetAtomicNum()] for atom in mol.GetAtoms()
        ], dtype=torch.int)
    
    @staticmethod
    def _create_adj_list(mol):
        """
        Create an adjacency list for a molecule.
        """
        adj_list = torch.tensor([
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
        ], dtype=torch.int)

        if adj_list.size(0) == 0:
            return torch.empty((0, 2), dtype=torch.int)
        
        return torch.concat([
            adj_list, adj_list.flip(dims=(1, ))
        ])

    def _tokenize_smiles(self, smiles_arr, atomic_numbers):
        """
        Tokenize a list of SMILES strings.
        """

        unique_atomic_numbers = np.sort(np.unique(atomic_numbers))
        token_indices = {c: i for i, c in enumerate(unique_atomic_numbers)}

        # Tokenize and create adjacency lists

        tokens, adj_lists = [], []
        for smile in smiles_arr:
            mol = Chem.AddHs(MolFromSmiles(smile))
            Chem.Kekulize(mol, clearAromaticFlags=True)

            tokens.append(self._tokenize_mol(mol, token_indices))
            adj_lists.append(self._create_adj_list(mol))

        # Pad tokens

        padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)

        # Convert adjacency lists to sparse padded matrices

        n_tokens = padded_tokens.shape[-1]
        adj_matrices = torch.stack([
            torch.sparse_coo_tensor(
                adj_list.T, 
                torch.ones(adj_list.shape[0], dtype=torch.bool),
                (n_tokens, n_tokens)
            ).to_dense()
            for adj_list in adj_lists
        ])

        return padded_tokens, adj_matrices

    def __len__(self):
        return len(self.smiles_arr)

    def __getitem__(self, idx):
        return self.tokens[idx], self.padding[idx], self.adj_matrices[idx], self.y[idx]

    def unnormalize(self, y):
        assert y.shape[1] == self.y_mean.shape[1] == self.y_std.shape[1]
        return y * self.y_std + self.y_mean

    @staticmethod
    def collate(batch):
        tokens, padding, adj_matrices, y = default_collate(batch)
        L = (~padding).sum(dim=1).max().item() # Batch max sequence length
        return tokens[:, :L], padding[:, :L], adj_matrices[:, :L, :L], y

if __name__ == '__main__':
    
    import toml

    config_file = '/home/jshe/molecular_attention_bias/src/config/homo_lumo_U/GraphAttentionTransformer/masked_sdpa.toml'
    config = toml.load(config_file)

    data_dir = os.path.join(config['dataset_config']['data_dir'], 'train')
    target_labels = config['dataset_config']['target_labels']
    dataset = MolecularGraphDataset(data_dir, target_labels)

    print('____Shapes____')
    print('Number of samples:', len(dataset))
    print('Tokens:', dataset.tokens.shape)
    print('Padding:', dataset.padding.shape)
    print('Adjacency matrices:', dataset.adj_matrices.shape)
    print('y:', dataset.y.shape)
    print('y_mean:', dataset.y_mean.shape)
    print('y_std:', dataset.y_std.shape)
    print('y_labels:', dataset.y_labels)

    print('\n____Normalization____')
    print('Mean: ', dataset.y_mean)
    print('Std: ', dataset.y_std)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for tokens, padding, adj_matrices, y in dataloader:
        print('\n____Batch Shapes____')
        print('Batch tokens:', tokens.shape)
        print('Batch padding:', padding.shape)
        print('Batch adjacency matrices:', adj_matrices.shape)
        print('Batch y:', y.shape)

        print('\n____Batch Statistics____')
        print('Batch mean: ', y.mean(axis=0))
        print('Batch std: ', y.std(axis=0))
        break