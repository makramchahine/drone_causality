#!/bin/bash

sbatch -J vanilla_objective one_tuning_noqueue.bash
sleep 10
sbatch -J bidirect_objective one_tuning_noqueue.bash