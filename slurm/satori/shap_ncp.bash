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
python visualization_runner.py visualize_each slurm/satori/chair_approach.json /home/yinm/rosetta_drone/rnn_control/src/models/chair3_all/train/params.json --vis_type shap --vis_model_type ncp
