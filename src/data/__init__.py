from .point_clouds import PointCloudDataset
from .molecular_graphs import MolecularGraphDataset

name_to_dataset = {
    "PointCloudDataset": PointCloudDataset,
    "MolecularGraphDataset": MolecularGraphDataset,
}