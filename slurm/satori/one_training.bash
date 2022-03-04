#!/bin/bash
#SBATCH -J one_training
#SBATCH -o one_training%j.out
#SBATCH -e one_training%j.err
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
PYTHON_VIRTUAL_ENVIRONMENT=test8
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
cd ~/drone-causality
./tf_data_training.py --model lstm --rnn_sizes 128 --data_dir /nobackup/users/pdkao/data/devens_snowy_sliced --seq_len 64 --epochs 100 --val_split 0.05 --opt adam --lr .0009 --data_shift 16 --data_stride 1 --batch_size 300