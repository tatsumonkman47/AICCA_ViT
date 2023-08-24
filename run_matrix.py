import subprocess
import os
import glob
from datetime import datetime 

models = [
	  "AICCA_prototype_no_annulus",
      #"AICCA_prototype_no_annulus",
         ]

current_time = datetime.now().strftime("%d-%m-%H:%M:%S")
partition = "on_gpu"
quick_run = False
run_12hrs = False
use_latest = True
use_version = False
version = ["version_29437262","version_29437262"]

#different_logdir_path = True
#different_logdir_paths = []

DATASET_PATH = "cloudimages1M"
lr = [1e-5,
      #1e-6
     ]
lr_scheduler_flag = ["MultistepLR",
                     #"CosineAnnealingLR",
                    ]


LOG_DIR = [f"TESTAUG16_{model}_{lr_scheduler_flag[i]}_lr{lr[i]}" for i, model in enumerate(models)]
LOG_DIR = ["MultistepLR_1e4"]
LOG_DIR = ["TEST_1e5_from_1e6"]
LOG_DIR = [f"ViT_1Mlr1e-5_no_annulus/"]

print(LOG_DIR)

EXTRA_LABEL_FLAG = ""

for i, model in enumerate(models):
    # Read in and modify sbatch file
    with open(f"./run_{partition}.sbatch", "r") as data_file:
        list_of_lines = data_file.readlines()
        for j, line in enumerate(list_of_lines):
            if "job-name" in line:
                list_of_lines[j] = f"#SBATCH --job-name={model}{lr[i]}{EXTRA_LABEL_FLAG} \n" 
            if "--output" in line:
                list_of_lines[j] = f"#SBATCH --output={model}{lr[i]}{EXTRA_LABEL_FLAG}_{current_time}.out \n" 
            if "--error" in line:
                list_of_lines[j] = f"#SBATCH --error={model}{lr[i]}{EXTRA_LABEL_FLAG}_{current_time}.err \n" 
            if "--time" in line:
                if quick_run:
                    list_of_lines[j] = f"#SBATCH --time=0:10:00 \n" 
                if run_12hrs:
                    list_of_lines[j] = f"#SBATCH --time=12:00:00 \n" 
                else:
                    list_of_lines[j] = f"#SBATCH --time=35:59:59 \n" 
            if "python" in line:
                list_of_lines[j] = f"srun python {model}.py \n" 
    # Save modified sbatch file
    with open(f"./run_{partition}.sbatch", "w") as data_file:
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
                model_list_of_lines[j] = f'training_checkpoint_path = f"{os.getcwd()}/{LOG_DIR[i]}" \n'
            if "LEARNING RATE FLAG" in line:
                print(f"Using learning rate = {lr[i]}")
                model_list_of_lines[j] = f' \t\t\t\t\t lr={lr[i]}, #LEARNING RATE FLAG \n'
            if "LR SCHEDULER FLAG" in line:
                print(f"Using lr scheduler = {lr_scheduler_flag[i]}")
                model_list_of_lines[j] = f' \t\t\t\t\t lr_scheduler_flag="{lr_scheduler_flag[i]}", # LR SCHEDULER FLAG \n'
    # Save modified model.py file
    with open(f"./{model}.py", "w") as model_file:
        model_file.writelines(model_list_of_lines)
    
    subprocess.call(f"sbatch run_{partition}.sbatch", shell=True)
