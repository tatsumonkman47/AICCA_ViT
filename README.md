# AICCA_ViT
Revisiting the AICCA cloud classification network using the Vision Transformer (ViT).

This project is an modification of the network used to generate [AI-Driven Cloud Classification Atlas](https://www.mdpi.com/2072-4292/14/22/5690), an unsupervised cloud classification system based off of Takuya Kurihana's PhD work.

AICCA ViT is in it's early stages, and this repository is mostly for me and Takuya to prototype. I've organized this repository similarily to the directory on midway3 that I'm using to train/test new ViT architectures. We haven't added unsupervised clustering yet.


## A quick overview of how this directory is organized
---------------------------------------------------
#### Current Model
The current model can be found in the "AICCA_prototype.py" script. Various other scripts such as "AICCA_prototype_run[x].py" are used for hyperparemeter testing, and can be ignored for now.

#### To Run
I currently run the model using the "run_matrix_midway3.py" script, which modifies the sbatch submission script ("run_on_midway3_gpu.sbatch") and the pytorch model script ("AICCA_prototype.py"). To submit a job simply type "python run_matrix_midway3.py" in the project directory. On submission, "run_matrix_midway3.py" will generate new ""run_on_midway3_gpu_run[x].sbatch" and "AICCA_prototype_run[x].py" scripts to be used during training. Need to organize this better, but it works for now...

#### To Modify Hyperparameters
Hyperparameters can be modified either in the "AICCA_prototype.py" script directly or in the "run_matrix_midway3.py" submission script. It is better to modifiy hyperparameters in "run_matrix_midway3.py," to avoid things being overwritten / forgotten when testing a lot of models.

#### DATA
The current scripts assume the data folder is present in the project directory, and specifies the path to the data in line 16 of "run_matrix_midway3.py": 'DATASET_PATH = "cloudimages1M"'. To use a different path just modify line 16 in "run_matrix_midway3.py".

#### Models
Previous training runs are stored in the "training_runs" directory. I've tried to make the names descriptive of the setups we tested...



# Other things to keep in mind
----------------------------
#### Conda Environment
I put the conda environment.yml file in the project directory under the name "AICCA_pytorch.yml"
#### Dependencies
Pytorch, Pytorch Lighting, cuda, numpy, 
#### Internal Dependencies
I use a helper function to load the AICCA patches (which are .npy files) into the datalaoders. The helper function is titled "import_numpy.py" and is located in the main project directory.

