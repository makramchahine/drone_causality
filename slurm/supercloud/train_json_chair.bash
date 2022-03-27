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
python train_multiple.py "${SLURM_JOB_NAME}" /home/gridsan/pdkao/data/devens_chair --n_trains 5 --batch_size 128 --storage_name old_db/"${SLURM_JOB_NAME}".json --storage_type json --extra_data_dir /home/gridsan/pdkao/data/synthetic_chair3 --out_prefix chair3
