#!/bin/bash

#SBATCH --job-name=graph_transformer
#SBATCH --error=run.err
#SBATCH --output=/dev/null
#SBATCH --account=pi-andrewferguson
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

conda init
conda activate env

python3 script_train.py $1
