#!/bin/bash

#SBATCH --job-name=train
#SBATCH --error=job.err
#SBATCH --output=/dev/null
#SBATCH --account=pi-andrewferguson
#SBATCH --time=02:00:00
#SBATCH --partition=andrewferguson-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

cd /home/jshe/molecular_attention_bias/src

python3 script_train.py $1
