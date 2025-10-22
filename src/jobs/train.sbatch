#!/bin/bash

#SBATCH --job-name=train
#SBATCH --error=error.txt
#SBATCH --output=/dev/null
#SBATCH --account=pi-andrewferguson
#SBATCH --time=04:00:00
#SBATCH --partition=andrewferguson-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

cd /home/jshe/molecular_attention_bias/src

python3 script_train.py $1
