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
import wandb

import torch

import models
import data
from trainers.train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _make_run_ids(model_type, radial_function_type, E, H, D, *args, **kwargs):

    # Group ID
    group_id = model_type + '_'
    group_id += radial_function_type + '_'
    group_id += f'E{E}H{H}D{D}'

    # Run ID
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    slurm_id = str(slurm_id if slurm_id is not None else 0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp + slurm_id

    return group_id, run_id

def main(config):

    # Expand config

    output_dir = config['output_dir']
    model_config = config['model_config']
    train_config = config['train_config']
    train_dataset_config = config['train_dataset_config']
    val_dataset_config = config['val_dataset_config']

    # Make run identifiers

    group_id, run_id = _make_run_ids(**model_config)
    output_dir = os.path.join(output_dir, group_id, run_id)

    # Initialize wandb

    logger = wandb.init(
        project="molecular-attention-bias", 
        dir=output_dir, 
        name=run_id, 
        group=group_id, 
        config=config, 
        mode='offline', 
    )

    # Initialize model

    model = models.get_model(**model_config).to(device)
    # logger.watch(model)

    # Initialize datasets

    train_dataset = data.get_dataset(**train_dataset_config)
    val_dataset = data.get_dataset(**val_dataset_config)

    # Train model

    train(
        model, device, 
        train_dataset, val_dataset, 
        **train_config, 
        output_dir=output_dir, 
        logger=logger
    )

    # Clean up

    logger.finish()

if __name__ == '__main__':

    config_path = sys.argv[1]
    config = toml.load(config_path)

    main(config)