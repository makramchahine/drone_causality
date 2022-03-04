#!/bin/bash

sbatch -J vanilla_objective one_tuning_noqueue.bash
sbatch -J bidirect_objective one_tuning_noqueue.bash