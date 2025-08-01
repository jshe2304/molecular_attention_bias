from .point_clouds import PointCloudDataset
from .molecular_graphs import MolecularGraphDataset

def get_dataset(dataset_type: str, *args, **kwargs):
    if dataset_type == "PointCloudDataset":
        return PointCloudDataset(*args, **kwargs)
    if dataset_type == "MolecularGraphDataset": 
        return MolecularGraphDataset(*args, **kwargs)