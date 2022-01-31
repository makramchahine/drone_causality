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
conda activate ramin

cd ~/deepdrone/utils
python train_multiple.py ncp_objective /home/gridsan/pdkao/data/devens_12102021_sliced --num_trains 5 --batch_size 128 --storage_name sqlite:///ncp.db
