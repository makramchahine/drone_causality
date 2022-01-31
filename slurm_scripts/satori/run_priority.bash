#!/bin/bash

sbatch -J gruode_objective one_tuning.bash
sbatch -J hawk_objective one_tuning.bash
sbatch -J mmrnn_objective one_tuning.bash
sbatch -J ctgru_objective one_tuning.bash