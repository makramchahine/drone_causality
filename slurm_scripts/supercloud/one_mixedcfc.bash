#!/bin/bash

#SBATCH -o one_mixedcfc.log-%j
#SBATCH -c 40
#SBATCH --gres=gpu:volta:2

source /etc/profile
module load anaconda/2020a
module load cuda/10.2

eval "$(conda shell.bash hook)"
conda activate ramin

cd ~/deepdrone/utils
python hyperparameter_tuning.py mixedcfc_objective /home/gridsan/pdkao/data/devens_12102021_sliced --n_trials 20 --batch_size 128 --storage_name mixedcfc.pkl --save_pkl
