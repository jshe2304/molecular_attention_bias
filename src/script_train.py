"""
Transformer training script.

To run, pass in a path to a TOML config file as an argument. 
The TOML should contain the following sections:
- model: The model to train
- training: The training parameters
- train_dataset: The training dataset
- validation_dataset: The validation dataset
"""

import sys
import toml

import torch

import models
import data
from trainers.train import train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # Initialize model

    Model = models.name_to_model[config['model_name']]
    model = Model(**config['model']).to(device)

    # Initialize datasets

    Dataset = data.name_to_dataset[config['dataset_name']]
    train_dataset = Dataset(**config['train_dataset'])
    validation_dataset = Dataset(**config['validation_dataset'])

    # Train model

    train(
        model, device, 
        train_dataset, validation_dataset, 
        **config['training'], 
    )
