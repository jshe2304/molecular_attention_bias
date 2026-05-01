import numpy as np

from rdkit.Chem import GetPeriodicTable

import os

xyzdir = '/scratch/midway3/jshe/data/qm9/xyz/'
outdir = '/scratch/midway3/jshe/data/qm9/raw/'

pse = GetPeriodicTable()

bad_samples = set([
    21725, 87037, 117523, 128113, 129053, 
    129152, 129158, 130535, 6620, 59818, 129053
])

# Loop through XYZ files

all_coordinates = []
all_atoms = []
all_smiles = []
all_properties = []

for i, fname in enumerate(os.listdir(xyzdir)):
    if i % 10000 == 0: print(f'Processed {i} files')

    # Read XYZ file

    with open(os.path.join(xyzdir, fname)) as f: 
        lines = f.readlines()

    # Read number of atoms

    n_atoms = int(lines[0])

    # Read properties and SMILES

    tag, mol_id, *properties = lines[1].split()
    if int(mol_id) in bad_samples: continue
    smiles, _ = lines[-2].split()

    # Read out coordinates, partial charges, and atoms from XYZ file

    coordinates = np.full((29, 3), np.inf)
    atoms = np.zeros(29)
    for j in range(2, n_atoms + 2):
        atom, *(coordinates[j-2]), _ = lines[j].replace('*^', 'e').split()
        atoms[j-2] = pse.GetAtomicNumber(atom)

    # Append to lists

    all_smiles.append(smiles)
    all_properties.append(properties)
    all_coordinates.append(coordinates)
    all_atoms.append(atoms)

# Save data as .npy files

np.save(outdir + 'smiles.npy', np.array(all_smiles))
np.save(outdir + 'y.npy', np.array(all_properties).astype(float))
np.save(outdir + 'coordinates.npy', np.array(all_coordinates).astype(float))
np.save(outdir + 'atomic_numbers.npy', np.array(all_atoms).astype(int))

y_labels = np.array(['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'])
np.save(outdir + 'y_labels.npy', y_labels)