#!/usr/bin/env bash

#SBATCH --job-name=polnota
#SBATCH --partition=ais-gpu 
#SBATCH --gpus=1 
#SBATCH --nodes=1              # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB 
#SBATCH --time=24:00:00 
#SBATCH --output=/trinity/home/alina.smolina/sbatch_logs/%A@%x_%a.out 
#SBATCH --error=/trinity/home/alina.smolina/sbatch_logs/%A@%x_%a.err

source activate conda
source activate geotorch

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python3 DL-pipeline/notebooks/sakhalin_polnota_train.py
