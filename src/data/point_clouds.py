import numpy as np

import torch
from torch.utils.data import Dataset, default_collate
from torch.nn.utils.rnn import pad_sequence

class PointCloudDataset(Dataset):
    def __init__(
            self, 
            atoms_file, coordinates_file, y_file, 
            y_labels_file, y_mean_file, y_std_file, 
            target_labels,
            atom_symbols='HCNOF', 
            *args, **kwargs
        ):
        """
        Initialize a point cloud dataset.
        
        Args:
            atoms_file: Path to a numpy array of atom string sequences.
            coordinates_file: Path to a numpy array of coordinates.
            y_file: Path to a numpy array of properties.
            y_labels_file: Path to a numpy array of property labels.
            y_mean_file: Path to a numpy array of mean values of properties. 
            y_std_file: Path to a numpy array of standard deviation values of properties.
            target_labels: Which properties to include.
            atom_symbols: String of atom symbols to use for tokenization.
        """

        self.token_indices = {c: i for i, c in enumerate(atom_symbols, start=1)}

        # Make data

        self.atoms_arr = np.load(atoms_file, allow_pickle=True)
        self.tokens = self._tokenize_atoms(self.atoms_arr, self.token_indices)
        self.padding = (self.tokens == 0)
        self.coordinates = torch.tensor(
            np.load(coordinates_file, allow_pickle=True), 
            dtype=torch.float32
        )[:, :self.tokens.shape[1]]
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

        assert self.tokens.shape[0] == self.y.shape[0] == self.coordinates.shape[0] # Number of samples
        assert self.tokens.shape[1] == self.coordinates.shape[1] # Number of atoms
        assert self.y.shape[1] == self.y_mean.shape[1] == self.y_std.shape[1] # Number of properties

    @staticmethod
    def _tokenize_atoms(atoms_arr, token_indices):
        """
        Tokenize atom strings given a token index dictionary.
        """
        tokens = [
            torch.tensor([token_indices[atom] for atom in atoms]).int()
            for atoms in atoms_arr
        ]
        return pad_sequence(tokens, batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.atoms_arr)

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

    config_file = '../config/train/power_law_bias.toml'
    config = toml.load(config_file)

    dataset = PointCloudDataset(**config['train_dataset_config'])

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