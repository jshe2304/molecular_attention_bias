#!/bin/bash

# # SPICE

# sbatch jobs/train_ensemble.sbatch config/spice/U/BiasedAttentionTransformer/mixed_1.toml
# sbatch jobs/train_ensemble.sbatch config/spice/U/BiasedAttentionTransformer/mixed_2.toml
# sbatch jobs/train_ensemble.sbatch config/spice/U/BiasedAttentionTransformer/mixed_3.toml
# sbatch jobs/train_ensemble.sbatch config/spice/U/BiasedAttentionTransformer/mixed_4.toml

# sbatch jobs/train_ensemble.sbatch config/spice/U/GraphAttentionTransformer/masked_sdpa.toml
# sbatch jobs/train_ensemble.sbatch config/spice/U/GraphPETransformer/random_walk_pe.toml

# QM9

sbatch jobs/train_ensemble.sbatch config/qm9/homo_lumo_U_H_G/BiasedAttentionTransformer/mixed_1.toml
sbatch jobs/train_ensemble.sbatch config/qm9/homo_lumo_U_H_G/BiasedAttentionTransformer/mixed_2.toml

sbatch jobs/train_ensemble.sbatch config/qm9/homo_lumo_U_H_G/FixedAttentionTransformer/mixed_1.toml
sbatch jobs/train_ensemble.sbatch config/qm9/homo_lumo_U_H_G/FixedAttentionTransformer/mixed_2.toml

# sbatch jobs/train_ensemble.sbatch config/qm9/homo_lumo_U_H_G/GraphAttentionTransformer/masked_sdpa.toml
# sbatch jobs/train_ensemble.sbatch config/qm9/homo_lumo_U_H_G/GraphPETransformer/random_walk_pe.toml
