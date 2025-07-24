import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from rdkit.Chem import MolFromSmiles
from rdkit import Chem

class MolecularGraphDataset(Dataset):
    def __init__(
            self, 
            smiles_file, y_file, 
            y_labels_file, y_mean_file, y_std_file, 
            target_labels,
            atom_symbols='HCNOF', 
        ):
        """
        Initialize a molecular graph dataset.
        
        Args:
            smiles_file: Path to a numpy array of SMILES strings.
            y_file: Path to a numpy array of properties.
            y_labels_file: Path to a numpy array of property labels.
            y_mean_file: Path to a numpy array of mean values of properties. 
            y_std_file: Path to a numpy array of standard deviation values of properties.
            target_labels: Which properties to include.
            atom_symbols: String of atom symbols to use for tokenization.
        """

        self.token_indices = {c: i for i, c in enumerate(atom_symbols, start=1)}

        # Load data

        self.smiles_arr = np.load(smiles_file, allow_pickle=True)
        self.tokens, self.adj_matrices = self._tokenize_smiles(self.smiles_arr, self.token_indices)
        self.padding = (self.tokens == 0)
        self.y = torch.tensor(
            np.load(y_file, allow_pickle=True), 
            dtype=torch.float32
        )

        # Select desired properties

        all_y_labels = np.load(y_labels_file, allow_pickle=True)
        label_indices = np.in1d(all_y_labels, target_labels).nonzero()[0]
        self.y_labels = all_y_labels[label_indices]
        self.y = self.y[:, label_indices]

        # Load target normalization statistics

        self.y_mean = torch.tensor(
            np.load(y_mean_file, allow_pickle=True), 
            dtype=torch.float32
        )[label_indices].unsqueeze(0)
        self.y_std = torch.tensor(
            np.load(y_std_file, allow_pickle=True), 
            dtype=torch.float32
        )[label_indices].unsqueeze(0)

        assert self.tokens.shape[0] == self.adj_matrices.shape[0] == self.y.shape[0] # Number of samples
        assert self.tokens.shape[1] == self.adj_matrices.shape[1] == self.adj_matrices.shape[2] # Number of atoms
        assert self.y.shape[1] == self.y_mean.shape[1] == self.y_std.shape[1] # Number of properties
    
    @staticmethod
    def _tokenize_mol(mol, token_indices):
        """
        Tokenize a molecule.
        """
        return torch.tensor([
            token_indices[atom.GetSymbol()] for atom in mol.GetAtoms()
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

    def _tokenize_smiles(self, smiles_arr, token_indices):
        """
        Tokenize a list of SMILES strings.
        """

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
        return y * self.y_std + self.y_mean

if __name__ == '__main__':
    
    dataset = MolecularGraphDataset(
        smiles_file = "/scratch/midway3/jshe/data/qm9/scaffolded/train/smiles.npy",
        y_file = "/scratch/midway3/jshe/data/qm9/scaffolded/train/y.npy",
        y_mean_file = "/scratch/midway3/jshe/data/qm9/transformed/y_mean.npy",
        y_std_file = "/scratch/midway3/jshe/data/qm9/transformed/y_std.npy",
        y_labels_file = "/scratch/midway3/jshe/data/qm9/transformed/y_labels.npy",
        target_labels = ['homo', 'lumo', 'U0', 'U', 'H', 'G'],
    )

    print('Number of samples:', len(dataset))
    print('Tokens:', dataset.tokens.shape)
    print('Padding:', dataset.padding.shape)
    print('Adjacency matrices:', dataset.adj_matrices.shape)
    print('y:', dataset.y.shape)
    print('y_mean:', dataset.y_mean.shape)
    print('y_std:', dataset.y_std.shape)
    print('y_labels:', dataset.y_labels)
    print()

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for tokens, padding, adj_mats, y in dataloader:
        print('Batch tokens shape:', tokens.shape)
        print('Batch padding shape:', padding.shape)
        print('Batch adjacency matrices shape:', adj_mats.shape)
        print('Batch targets shape:', y.shape)
        break