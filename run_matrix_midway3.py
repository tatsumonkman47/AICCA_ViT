import subprocess
import os
import glob
from datetime import datetime 

##################################
#### SPECIFY MODEL PYTHON FILE ###
##################################
# The strings listed in "models" are the names of the python scripts
# that contain the ViT models. I don't add the ".py" extension because I use 
# the model name for some other labeling sometimes
models = [
          "AICCA_prototype",
          "AICCA_prototype",
         ]

current_time = datetime.now().strftime("%d-%m-%H:%M:%S")
partition = "on_midway3_gpu"
quick_run = False
run_12hrs = False
use_latest = True
use_version = False

##################################
###### SPECIFY DATASET PATH ## ###
##################################
# Specify the path to the dataset folder, here it's just
# in the cwd.
DATASET_PATH = "cloudimages1M"


##################################
###### SPECIFY HYPERPARAMETERS ###
##################################
# A couple of hyper parameters you can adjust right from this
# script (see where they are used below. If you want to change other
# ones you can either edit the model.py scripts directly or change this
# script to edit the model.py files during initalization. Each specified
# hyper paramter list should be the same length as the "models" list above,
# and each entry corresponds to the model listed in the same index in
# "models" 

lr = [1e-4,
      1e-4,
     ]

num_layers = [2,
              3,
              ]

patch_size = [16,
              16,
              ]

num_patches = [64,
              64,
              ]

lr_scheduler_flag = ["MultistepLR",
                     "MultistepLR",
                    ]
##################################
##### SPECIFY LOG DIRECTORIES  ###
##################################


LOG_DIR = [f"training_runs/ViT_1M_{num_layers[0]}layers_{num_patches[0]}patches",
           f"training_runs/ViT_1M_{num_layers[1]}layers_{num_patches[1]}patches",
           ]
print(LOG_DIR)

####################################################
##### EDIT AND SAVE NEW SCRIPT FILES FOR A RUN  ####
####################################################
# Here we edit both the submit.sbatch script for submiting a training run
# and the model.py script to be used. This will generate a new submit.sbatch
# and model.py file to use for the specific run. the new scripts are named 
# "xxxx_run[i].sbatch" and "xxx_run[i].py", where [i] specifies the the 
# index of the entry in "models" 
for i, model in enumerate(models):
    # Read in and modify sbatch file
    with open(f"./run_{partition}.sbatch", "r") as data_file:
        list_of_lines = data_file.readlines()
        for j, line in enumerate(list_of_lines):
            if "job-name" in line:
                list_of_lines[j] = f"#SBATCH --job-name={LOG_DIR[i]} \n" 
            if "--output" in line:
                list_of_lines[j] = f"#SBATCH --output={LOG_DIR[i]}.out \n" 
            if "--error" in line:
                list_of_lines[j] = f"#SBATCH --error={LOG_DIR[i]}.err \n" 
            if "--time" in line:
                if quick_run:
                    list_of_lines[j] = f"#SBATCH --time=0:10:00 \n" 
                if run_12hrs:
                    list_of_lines[j] = f"#SBATCH --time=12:00:00 \n" 
                else:
                    list_of_lines[j] = f"#SBATCH --time=35:59:59 \n" 
            if "python" in line:
                list_of_lines[j] = f"srun python {model}_run{i}.py \n" 
    # Save modified sbatch file
    with open(f"./run_{partition}_run{i}.sbatch", "w") as data_file:
        data_file.writelines(list_of_lines)
        
    # Read in and modify model file
    with open(f"./{model}.py", "r") as model_file:
        if os.path.isdir(LOG_DIR[i]):
            try:
                versions = [int(directory.split("_")[1]) for directory in os.listdir(f"{LOG_DIR[i]}/lightning_logs") if "version" in directory]
            except:
                versions = None
        else:
            versions = None
        model_list_of_lines = model_file.readlines()
        print(f"versions: {versions}")
        for j, line in enumerate(model_list_of_lines):
            if "LOG_DIR = " in line:
                print(f"Updated LOG_DIR to {LOG_DIR[i]}")
                model_list_of_lines[j] = f'LOG_DIR = "{LOG_DIR[i]}" \n'
            if "DATASET_PATH = " in line:
                print(f"Dataset path = {DATASET_PATH}")
                model_list_of_lines[j] = f'DATASET_PATH = "{DATASET_PATH}" \n'                
            # use the latest version if we're continuing a model run
            if "training_checkpoint_path = " in line and use_version:
                print(f"Found checkpoint in {LOG_DIR[i]}, running from version {max(versions)}")
                model_list_of_lines[j] = f'training_checkpoint_path = f"{os.getcwd()}/{LOG_DIR[i]}/lightning_logs/version_{version[i]}/checkpoints/" \n'         
            elif "training_checkpoint_path = " in line and versions and use_latest:
                print(f"Found checkpoint in {LOG_DIR[i]}, running from version {max(versions)}")
                model_list_of_lines[j] = f'training_checkpoint_path = f"{os.getcwd()}/{LOG_DIR[i]}/lightning_logs/version_{max(versions)}/checkpoints/" \n'
            # do this if we're running from scratch
            elif "training_checkpoint_path = " in line:
                print(f"Starting from scratch writing to {LOG_DIR[i]}")
                model_list_of_lines[j] = f"training_checkpoint_path = f'{os.getcwd()}/{LOG_DIR[i]}' \n"
            if "LEARNING RATE FLAG" in line:
                print(f"Using learning rate = {lr[i]}")
                model_list_of_lines[j] = f"\t\t\t\t lr={lr[i]}, #LEARNING RATE FLAG \n"
            if "LR SCHEDULER FLAG" in line:
                print(f"Using lr scheduler = {lr_scheduler_flag[i]}")
                model_list_of_lines[j] = f"\t\t\t\t lr_scheduler_flag='{lr_scheduler_flag[i]}', # LR SCHEDULER FLAG \n"
            if "'num_layers':" in line:
                print(f"Using num_layers = {num_layers[i]}")
                model_list_of_lines[j] = f"\t\t\t\t'num_layers':{num_layers[i]}, # number of layers in attention block \n"
            if "'patch_size':" in line:
                print(f"Using patch_size = {patch_size[i]}")
                model_list_of_lines[j] = f"\t\t\t\t'patch_size':{patch_size[i]}, # patch size \n"
            if "'num_patches':" in line:
                print(f"Using num_patches = {num_patches[i]}")
                model_list_of_lines[j] = f"\t\t\t\t'num_patches':{num_patches[i]}, # num patches \n"
    # Save modified model.py file
    with open(f"./{model}.py", "w") as model_file:
        model_file.writelines(model_list_of_lines)
        
    with open(f"./{model}_run{i}.py", "w") as model_file:
        model_file.writelines(model_list_of_lines)
        
    subprocess.call(f"sbatch run_{partition}_run{i}.sbatch", shell=True)
