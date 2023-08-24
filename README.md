# AICCA_ViT
Revisiting the AICCA cloud classification network using the Vision Transformer (ViT)

This project is in it's beginning stages. I've organized this repository similarily to the directory on midway3 that I'm using to train/test new ViT architectures.


A quick overview of how this directory is organized
---------------------------------------------------
CURRENT MODEL: The current model can be found in the "AICCA_prototype.py" script. Various other scripts such as "AICCA_prototype_run[x].py" are used for hyperparemeter testing, and can be ignored for now.

TO RUN: I currently run the model using the "run_matrix_midway3.py" script, which modifies the sbatch submission script ("run_on_midway3_gpu.sbatch") and the pytorch model script ("AICCA_prototype.py"). To submit a job simply type "python run_matrix_midway3.py" in the project directory. On submission, "run_matrix_midway3.py" will generate new ""run_on_midway3_gpu_run[x].sbatch" and "AICCA_prototype_run[x].py" scripts to be used during training. Need to organize this better, but it works for now...

TO MODIFY HYPERPARAMETERS: Hyperparameters can be modified either in the "AICCA_prototype.py" script directly or in the "run_matrix_midway3.py" submission script. It is better to modifiy hyperparameters in "run_matrix_midway3.py," to avoid things being overwritten / forgotten when testing a lot of models.

DATA: The current scripts assume the data folder is present in the project directory, and specifies the path to the data in line 16 of "run_matrix_midway3.py": 'DATASET_PATH = "cloudimages1M"'. To use a different path just modify line 16.

PREVIOUS RUNS: previous runs are stored in the "training_runs" directory



Other things to keep in mind
----------------------------
CONDA ENVIRONMENT: I put the conda environment.yml file in the project directory under the name "AICCA_pytorch.yml"
DEPENDENCIES: Pytorch, Pytorch Lighting, cuda, numpy, 
INTERNAL DEPENDENCIES: I use a helper function to load the AICCA patches (which are .npy files) into the datalaoders. The helper function is titled "import_numpy.py" and is in the main project directory

