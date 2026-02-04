#!/bin/bash -l
#SBATCH --job-name=obama-dpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output train_dpo.out
#SBATCH --error train_dpo.err
#SBATCH --gpus=1
module load Miniforge3
conda activate obama-digital-twin


python train_dpo.py