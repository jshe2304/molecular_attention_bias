"""
Transformer training script.

To run, pass in a path to a TOML config file as an argument. 
The TOML should contain the following sections:
- model_config: The model to train
- train_config: The training parameters
- train_dataset_config: The training dataset
- val_dataset_config: The validation dataset
"""

import os
import sys
import toml
from datetime import datetime

import torch

import models
import datasets
from trainers.train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _make_run_ids(model_type, E, H, D, **kwargs):
    '''
    Create run identifiers for logging. 
    '''

    # Group ID
    group_id = model_type + '/'
    if 'radial_function_type' in kwargs:
        if type(kwargs['radial_function_type']) == list:
            group_id += kwargs['identifier']
        else:
            group_id += kwargs['radial_function_type']
        group_id += str(kwargs.get('p', ''))
        group_id += str(kwargs.get('n_kernels', ''))
    elif 'positional_encoding_type' in kwargs:
        group_id += kwargs['positional_encoding_type']
        group_id += str(kwargs.get('k', ''))
    elif 'graph_attention_operator_type' in kwargs:
        group_id += kwargs['graph_attention_operator_type']
    group_id += f'/E{E}H{H}D{D}'

    # Run ID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    slurm_id = str(slurm_id if slurm_id is not None else 0)
    run_id = timestamp + slurm_id

    return group_id, run_id

def main(config):

    # Expand config

    output_dir = config['output_dir']
    model_config = config['model_config']
    dataset_config = config['dataset_config']
    train_config = config['train_config']
    
    # Make run identifiers

    group_id, run_id = _make_run_ids(**model_config)
    label_id = '_'.join(dataset_config['target_labels'])
    output_dir = os.path.join(output_dir, label_id, group_id, run_id)

    # Initialize model

    model_config['out_features'] = len(dataset_config['target_labels'])
    model = models.get_model(**model_config).to(device)

    # Create datasets

    train_dataset = datasets.get_dataset(**dataset_config, split='train')
    val_dataset = datasets.get_dataset(**dataset_config, split='validation')
    test_dataset = datasets.get_dataset(**dataset_config, split='test')
    
    # Train model

    train(
        model, device, 
        train_dataset, val_dataset, test_dataset, 
        **train_config, 
        output_dir=output_dir, 
    )

if __name__ == '__main__':

    config_path = sys.argv[1]
    #print(config_path)
    config = toml.load(config_path)

    main(config)