#!/bin/bash
#SBATCH --job-name=AICCA_fit_wide 
#SBATCH --output=AICCA_fit_wide.out
#SBATCH --error=AICCA_fit_wide.err 
#SBATCH --time=23:59:00
#SBATCH --account=pi-moyer
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1


conda activate please_work

python AICCA_prototype_wide.py

