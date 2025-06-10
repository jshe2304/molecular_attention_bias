#!/bin/bash

# List of class names to run
BIAS_CLASSES=(
    "FixedPowerLaw1"
    "FixedPowerLaw2"
    "FixedPowerLaw3"
    "FixedPowerLaw4"
    "FixedPowerLaw5"
    "LearnedNegativePowerLaw"
    "GaussianBasis"
)

# Loop through each class and submit a SLURM job
for CLASS_NAME in "${BIAS_CLASSES[@]}"; do
    sbatch run.sbatch "$CLASS_NAME" 128 8 8