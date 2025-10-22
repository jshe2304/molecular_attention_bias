#!/bin/bash

#SBATCH --job-name=cpu
#SBATCH --error=error.txt
#SBATCH --output=/dev/null
#SBATCH --account=pi-andrewferguson
#SBATCH --time=02:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=2000

cd /home/jshe/molecular_attention_bias/src

python3 "$@"
