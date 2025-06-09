import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

import os

outdir = '/scratch/midway3/jshe/data/qm9/raw/'
xyzdir = '/scratch/midway3/jshe/data/qm9/xyz/'

bad_samples = set([
    21725, 87037, 117523, 128113, 129053, 
    129152, 129158, 130535, 6620, 59818, 129053
])

# Loop through XYZ files

all_fnames = []
all_coordinates = []
all_conformers = []
all_partial_charges = []
all_atoms = []
all_smiles = []
all_relaxed_smiles = []
all_properties = []

for i, fname in enumerate(os.listdir(xyzdir)):

    # Read XYZ file

    with open(os.path.join(xyzdir, fname)) as f: 
        lines = f.readlines()

    # Read number of atoms

    n_atoms = int(lines[0])

    # Read properties and SMILES

    tag, mol_id, *properties = lines[1].split()
    if int(mol_id) in bad_samples: continue
    smile, relaxed_smile = lines[-2].split()

    # Create conformer

    mol = Chem.MolFromSmiles(relaxed_smile)
    if mol is None: continue
    mol = Chem.AddHs(mol)
    embed = AllChem.EmbedMolecule(mol, randomSeed=16)
    if embed != 0: continue
    AllChem.UFFOptimizeMolecule(mol)

    # Read out conformer coordinates

    conformers = np.full((9, 3), np.inf)
    j = 0
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H': continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        conformers[j] = pos.x, pos.y, pos.z
        j += 1

    # Read out coordinates, partial charges, and atoms from XYZ file

    coordinates = np.full((29, 3), np.inf)
    partial_charges = np.zeros(29)
    atoms = ''
    for j in range(2, n_atoms + 2):
        atom, *(coordinates[j-2]), partial_charges[j-2] = lines[j].replace('*^', 'e').split()
        atoms += atom

    # Append to lists

    all_fnames.append(fname)
    all_smiles.append(smile)
    all_relaxed_smiles.append(relaxed_smile)
    all_properties.append(properties)
    all_coordinates.append(coordinates)
    all_conformers.append(conformers)
    all_partial_charges.append(partial_charges)
    all_atoms.append(atoms)

# Save data as .npy files

np.save(outdir + 'fnames.npy', np.array(all_fnames))
np.save(outdir + 'smiles.npy', np.array(all_smiles))
np.save(outdir + 'relaxed_smiles.npy', np.array(all_relaxed_smiles))
np.save(outdir + 'properties.npy', np.array(all_properties).astype(float))
np.save(outdir + 'coordinates.npy', np.array(all_coordinates).astype(float))
np.save(outdir + 'conformers.npy', np.array(all_conformers).astype(float))
np.save(outdir + 'partial_charges.npy', np.array(all_partial_charges).astype(float))
np.save(outdir + 'atoms.npy', np.array(all_atoms))
