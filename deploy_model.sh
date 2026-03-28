#!/bin/bash -l
#SBATCH --job-name=obama-deploy
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=deploy.out
#SBATCH --error=deploy.err
#SBATCH --gpus=1

# Load necessary modules
module load Miniforge3

# Activate your conda environment
conda activate obama-digital-twin

# Run the deployment script
# The --host=0.0.0.0 makes the Gradio app accessible from outside the node
python deploy.py --simple
