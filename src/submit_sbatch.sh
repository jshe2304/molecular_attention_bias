#!/bin/bash

sbatch jobs/train.sbatch config/spice/U/BiasedAttentionTransformer/power_law.toml
sbatch jobs/train.sbatch config/qm9/U0_U_H_G/BiasedAttentionTransformer/power_law.toml

sbatch jobs/train.sbatch config/spice/U/BiasedAttentionTransformer/exp_negative_power_law.toml
sbatch jobs/train.sbatch config/qm9/U0_U_H_G/BiasedAttentionTransformer/exp_negative_power_law.toml

sbatch jobs/train.sbatch config/spice/U/BiasedAttentionTransformer/zeros.toml
sbatch jobs/train.sbatch config/qm9/U0_U_H_G/BiasedAttentionTransformer/zeros.toml

sbatch jobs/train.sbatch config/spice/U/BiasedAttentionTransformer/fixed_power_law_1.toml
sbatch jobs/train.sbatch config/qm9/U0_U_H_G/BiasedAttentionTransformer/fixed_power_law_1.toml

sbatch jobs/train.sbatch config/spice/U/BiasedAttentionTransformer/fixed_power_law_2.toml
sbatch jobs/train.sbatch config/qm9/U0_U_H_G/BiasedAttentionTransformer/fixed_power_law_2.toml

sbatch jobs/train.sbatch config/spice/U/BiasedAttentionTransformer/gaussian_kernel.toml
sbatch jobs/train.sbatch config/qm9/U0_U_H_G/BiasedAttentionTransformer/gaussian_kernel.toml

sbatch jobs/train.sbatch config/spice/U/BiasedAttentionTransformer/mixed_1.toml
sbatch jobs/train.sbatch config/qm9/U0_U_H_G/BiasedAttentionTransformer/mixed_1.toml