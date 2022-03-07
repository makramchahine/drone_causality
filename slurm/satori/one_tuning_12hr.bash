#!/bin/bash
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --mail-user=pdkao@mit.edu
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive

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
python hyperparameter_tuning.py "${SLURM_JOB_NAME}" /nobackup/users/pdkao/data/devens_snowy_fixed --n_trials 20 --timeout 28800 --batch_size 300
