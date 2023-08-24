#!/bin/bash
#SBATCH --job-name=ViT_1M_3layer 
#SBATCH --output=ViT_1M_3layer.out 
#SBATCH --error=ViT_1M_3layer.err 
#SBATCH --time=35:59:59 
#SBATCH --account=pi-moyer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=rtx6000
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8


##module load cuda
srun python AICCA_prototype.py 

