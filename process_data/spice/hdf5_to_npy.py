import h5py
import json
from tqdm import tqdm
import numpy as np

def save_keys(hdf_file, keys_file):
    with h5py.File(hdf_file, "r") as f:
        keys = list(f.keys())
        with open(keys_file, 'w') as f:
            json.dump(keys, f)

def load_keys(keys_file):
    with open(keys_file, 'r') as f:
        keys = json.load(f)
        return keys

def extract_data(hdf_file, keys):
    all_smiles = []
    all_atomic_numbers = []
    all_conformations = []
    all_formation_energies = []

    with h5py.File(hdf_file, "r") as f:
        for i, key in enumerate(keys):
            try:
                smiles = f[key]['smiles'][0]
                conformations = f[key]['conformations'][0]
                atomic_numbers = f[key]['atomic_numbers'][:]
                formation_energy = f[key]['formation_energy'][0]
            except:
                continue

            all_smiles.append(smiles)
            all_atomic_numbers.append(atomic_numbers)
            all_conformations.append(conformations)
            all_formation_energies.append(formation_energy)

            if i % 1000 == 0:
                print(f'{i} done. ')

    return (
        all_smiles, 
        all_atomic_numbers, 
        all_conformations, 
        all_formation_energies, 
    )

def pad_data(data, max_length, padding_value=np.inf):
    shape = data.shape
    padded_data = np.full((max_length, *(shape[1:])), padding_value)
    padded_data[:len(data)] = data
    return padded_data

if __name__ == '__main__':

    outdir = '/scratch/midway3/jshe/data/spice/raw/'
    keys_file = '/scratch/midway3/jshe/data/spice/hdf/keys.json'
    hdf_file = '/scratch/midway3/jshe/data/spice/hdf/spice-2.0.1.hdf5'
    max_length = 110 # max number of atoms in a molecule

    keys = load_keys(keys_file)

    # Compile lists of data

    (
        all_smiles, 
        all_atomic_numbers, 
        all_conformations, 
        all_formation_energies, 
    ) = extract_data(hdf_file, keys)

    # Pad variable length values

    all_atomic_numbers = [
        pad_data(atomic_numbers, max_length, padding_value=0) for atomic_numbers in all_atomic_numbers
    ]
    all_conformations = [
        pad_data(conformations, max_length) for conformations in all_conformations
    ]

    # Convert to numpy arrays

    all_smiles = np.array(all_smiles).astype(str)
    all_atomic_numbers = np.array(all_atomic_numbers).astype(int)
    all_conformations = np.array(all_conformations)
    all_formation_energies = np.array(all_formation_energies)[:, np.newaxis]

    # Save data

    np.save(outdir + 'smiles.npy', all_smiles)
    np.save(outdir + 'atomic_numbers.npy', all_atomic_numbers)
    np.save(outdir + 'coordinates.npy', all_conformations)
    np.save(outdir + 'y.npy', all_formation_energies)

    y_labels = np.array(['U'])
    np.save(outdir + 'y_labels.npy', y_labels)
