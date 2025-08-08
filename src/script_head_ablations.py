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
import copy
import toml

import numpy as np
import torch

import models
import data
from trainers.head_ablations import ablate_heads

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(config):

    # Expand config

    checkpoint_file = config['checkpoint_file']
    output_dir = config['output_dir']
    model_config = config['model_config']
    val_dataset_config = config['dataset_config']

    print(checkpoint_file)

    # Make output directory

    os.makedirs(output_dir, exist_ok=True)

    # Initialize model

    model_config['out_features'] = len(val_dataset_config['target_labels'])
    model = models.get_model(**model_config).to(device)

    # Load model

    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)

    # Initialize datasets

    val_dataset = data.get_dataset(**val_dataset_config)

    # Ablate heads

    exponents, losses = ablate_heads(model, device, val_dataset)

    # Save results

    np.save(os.path.join(output_dir, 'exponents.npy'), np.array(exponents))
    np.save(os.path.join(output_dir, 'head_ablation.npy'), np.array(losses))

    print('Saved.')


if __name__ == '__main__':

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # Run on single checkpoint

    if 'checkpoint_file' in config and 'output_dir' in config:
        main(config)
        exit()

    # Run on all checkpoints in directory

    if 'checkpoint_dir' in config and 'output_dir' not in config:
        
        for root, dirs, files in os.walk(config['checkpoint_dir']):
            if not files: continue

            checkpoint_file = os.path.join(root, 'model.pt')
            output_dir = root

            this_config = copy.deepcopy(config)

            this_config['checkpoint_file'] = checkpoint_file
            this_config['output_dir'] = output_dir

            main(this_config)
