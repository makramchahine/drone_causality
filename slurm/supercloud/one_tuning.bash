#!/bin/bash

#SBATCH -o %x.log-%j
#SBATCH -c 40
#SBATCH --gres=gpu:volta:2
#SBATCH --mail-user=pdkao@mit.edu
#SBATCH --mail-type=FAIL

source /etc/profile
module load anaconda/2020a
module load cuda/10.2

eval "$(conda shell.bash hook)"
export PYTHONNOUSERSITE=1
conda activate ramin

cd ~/drone-causality
python hyperparameter_tuning.py "${SLURM_JOB_NAME}" /home/gridsan/pdkao/data/devens_12102021_sliced --n_trials 20 --batch_size 128 --storage_name "${SLURM_JOB_NAME}".pkl --save_pkl
