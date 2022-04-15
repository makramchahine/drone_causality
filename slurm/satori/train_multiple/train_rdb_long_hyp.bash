#!/bin/bash
#SBATCH -o train_%x-%j.out
#SBATCH -e train_%x-%j.err
#SBATCH --mail-user=pdkao@mit.edu
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --qos=sched_level_2

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ramin
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source "${CONDA_ROOT}"/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
cd ~/drone-causality
python train_multiple.py "${SLURM_JOB_NAME}" /nobackup/users/pdkao/data/devens_snowy_fixed --n_trains 1 --batch_size 300 --storage_name sqlite:///old_db/hyperparam_tuning.db --storage_type rdb --timeout 57600 --extra_data_dir /nobackup/users/pdkao/data/synthetic_small4 --study_name hyperparam_tuning_ --out_dir chair4_long_balaned
