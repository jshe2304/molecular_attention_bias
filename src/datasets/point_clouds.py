import os

import numpy as np

import torch
from torch.utils.data import Dataset, default_collate
from torch.nn.utils.rnn import pad_sequence

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, target_labels, **kwargs):
        """
        Initialize a point cloud dataset.
        
        Args:
            data_dir: Path to a directory containing the data.
            target_labels: Which properties to include.
        """

        # Make data paths

        atoms_file = os.path.join(data_dir, 'atomic_numbers.npy')
        coordinates_file = os.path.join(data_dir, 'coordinates.npy')
        y_file = os.path.join(data_dir, 'y.npy')
        y_labels_file = os.path.join(data_dir, 'y_labels.npy')
        y_mean_file = os.path.join(data_dir, 'y_mean.npy')
        y_std_file = os.path.join(data_dir, 'y_std.npy')

        # Load data

        atomic_numbers = np.load(atoms_file, allow_pickle=True)
        self.tokens = self._tokenize_atoms(atomic_numbers)
        self.padding = (self.tokens == 0)
        self.coordinates = torch.tensor(np.load(coordinates_file), dtype=torch.float32)
        self.y_labels = np.load(y_labels_file)
        self.y = torch.tensor(np.load(y_file), dtype=torch.float32) # (n_samples, n_properties)
        self.y_mean = torch.tensor(np.load(y_mean_file), dtype=torch.float32).unsqueeze(0) # (1, n_properties)
        self.y_std = torch.tensor(np.load(y_std_file), dtype=torch.float32).unsqueeze(0) # (1, n_properties)

        # Select desired properties

        label_indices = np.in1d(self.y_labels, target_labels).nonzero()[0].tolist()
        self.y_labels = self.y_labels[label_indices]
        self.y = self.y[:, label_indices]
        self.y_mean = self.y_mean[:, label_indices]
        self.y_std = self.y_std[:, label_indices]

        # Normalize targets
        self.y = (self.y - self.y_mean) / self.y_std

        # Check that data is consistent
        assert self.tokens.shape[0] == self.y.shape[0] == self.coordinates.shape[0] # Number of samples
        assert self.tokens.shape[1] == self.coordinates.shape[1] # Number of atoms
        assert self.y.shape[1] == self.y_mean.shape[1] == self.y_std.shape[1] # Number of properties

    @staticmethod
    def _tokenize_atoms(atomic_numbers):
        """
        Tokenize atom strings given a token index dictionary.
        """

        unique_atomic_numbers = np.sort(np.unique(atomic_numbers))
        token_indices = {c: i for i, c in enumerate(unique_atomic_numbers)}

        tokens = [
            torch.tensor([token_indices[atom] for atom in molecule]).int()
            for molecule in atomic_numbers
        ]
        return pad_sequence(tokens, batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.padding[idx], self.coordinates[idx], self.y[idx]

    def unnormalize(self, y):
        assert y.shape[1] == self.y_mean.shape[1] == self.y_std.shape[1]
        return y * self.y_std + self.y_mean

    @staticmethod
    def collate(batch):
        tokens, padding, coordinates, y = default_collate(batch)
        L = (~padding).sum(dim=1).max().item() # Batch max sequence length
        return tokens[:, :L], padding[:, :L], coordinates[:, :L], y
    
if __name__ == '__main__':

    import toml

    config_file = '/home/jshe/molecular_attention_bias/src/config/train/spice/BiasedAttentionTransformer/power_law.toml'
    config = toml.load(config_file)

    data_dir = os.path.join(config['dataset_config']['data_dir'], 'train')
    target_labels = config['dataset_config']['target_labels']
    dataset = PointCloudDataset(data_dir, target_labels)

    print('____Shapes____')
    print('Number of samples:', len(dataset))
    print('Tokens:', dataset.tokens.shape)
    print('Padding:', dataset.padding.shape)
    print('Coordinates:', dataset.coordinates.shape)
    print('y:', dataset.y.shape)
    print('y_mean:', dataset.y_mean.shape)
    print('y_std:', dataset.y_std.shape)
    print('y_labels:', dataset.y_labels)

    print('\n____Normalization____')
    print('Mean: ', dataset.y_mean)
    print('Std: ', dataset.y_std)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    for tokens, padding, coordinates, y in dataloader:
        print('\n____Batch Shapes____')
        print('Batch tokens:', tokens.shape)
        print('Batch padding:', padding.shape)
        print('Batch coordinates:', coordinates.shape)
        print('Batch y:', y.shape)

        print('\n____Batch Statistics____')
        print('Batch mean: ', y.mean(axis=0))
        print('Batch std: ', y.std(axis=0))
        break