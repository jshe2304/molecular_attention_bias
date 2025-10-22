import os

from .point_clouds import PointCloudDataset
from .molecular_graphs import MolecularGraphDataset

def get_dataset(dataset_type, data_dir, split, **kwargs):
    data_dir = os.path.join(data_dir, split)
    if dataset_type == "PointCloudDataset":
        return PointCloudDataset(data_dir, **kwargs)
    if dataset_type == "MolecularGraphDataset": 
        return MolecularGraphDataset(data_dir, **kwargs)