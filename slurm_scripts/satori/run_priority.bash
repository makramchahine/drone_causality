#!/bin/bash

sbatch -J ctrnn_objective one_tuning.bash
sleep 10
sbatch -J node_objective one_tuning.bash
sleep 10
sbatch -J mmrnn_objective one_tuning.bash
sleep 10
sbatch -J ctgru_objective one_tuning.bash