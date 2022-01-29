#!/bin/bash

sbatch -J gruode_objective one_tuning.bash
sleep 10
sbatch -J hawk_objective one_tuning.bash
sleep 10
sbatch -J mmrnn_objective one_tuning.bash
sleep 10
sbatch -J ctgru_objective one_tuning.bash