#!/bin/bash
#SBATCH --job-name=AICCA_prototype_no_annulus 
#SBATCH --output=AICCA_prototype_no_annulus_06-08-15:49:37.out 
#SBATCH --error=AICCA_prototype_no_annulus_06-08-15:49:37.err 
#SBATCH --time=35:59:59 
#SBATCH --account=pi-moyer
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:2


srun python AICCA_prototype_no_annulus.py 

