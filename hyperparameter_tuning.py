#!/usr/bin/env python
import argparse
import functools
import logging
import os
import sys
from typing import Sequence

import joblib
# add directory up to path to get main naming script
from optuna.pruners import MedianPruner

from utils.objective_functions import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def optimize_hyperparameters(obj_fn: Callable, data_dir: str, study_name: str, n_trials: int, batch_size: int,
                             timeout: float = None, storage_name: str = "sqlite:///hyperparam_tuning.db",
                             save_pkl: bool = False, train_kwargs: Optional[Dict[str, Any]] = None):
    """
    Runner script that runs hyperparameter tuning for a given model

    @param obj_fn: Optuna objective function. Takes trial and returns objective value
    @param data_dir:
    @param study_name:
    @param n_trials:
    @param batch_size:
    @param timeout:
    @param storage_name:
    @param save_pkl:
    @return:
    @return:
    :param train_kwargs:
    """
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name_network = f"{study_name}_{obj_fn.__name__}"
    study_params = {
        "study_name": study_name_network,
        "load_if_exists": True,
        "direction": "minimize",
        "pruner": MedianPruner(n_warmup_steps=10, n_min_trials=3),
    }

    if save_pkl:
        path_relative = os.path.join(SCRIPT_DIR, storage_name)
        if os.path.exists(path_relative):
            study = joblib.load(path_relative)
        else:
            print(f"No existing study found at path {path_relative}. Creating a new one")
            study = optuna.create_study(**study_params)
    else:
        study = optuna.create_study(storage=storage_name, **study_params)

    if train_kwargs is None:
        train_kwargs = {}

    study.optimize(functools.partial(objective_fn, data_dir=data_dir, batch_size=batch_size, **train_kwargs),
                   n_trials=n_trials, timeout=timeout)

    if save_pkl:
        joblib.dump(study, storage_name)
    return study


def parse_unknown_args(unknown: Sequence[str]) -> Dict[str, str]:
    # TODO: support non-string arguments
    assert len(unknown) % 2 == 0, "all arguments need to be paired"
    arg_dict = {}
    for i in range(0, len(unknown), 2):
        arg_name = unknown[i]
        arg_val = unknown[i + 1]
        assert arg_name[0] == "-", "All argument names should be prefixed with a dash"
        assert arg_val[0] != "-", f"Received {arg_val} as argument value, but it has a leading dash"
        arg_dict[arg_name.replace("-", "")] = arg_val

    return arg_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter bayesian optimization on deepdrone data')
    parser.add_argument("objective_fn", type=str, help="Name of objective function in this file to run")
    parser.add_argument("data_dir", type=str, help="Folder path of dataset")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Number of seconds after which new trials won't be created")
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for training")
    parser.add_argument("--storage_name", type=str, default="sqlite:///hyperparam_tuning.db",
                        help="Filepath for pkl, or URL for SQL RDB")
    parser.add_argument("--save_pkl", action='store_true',
                        help="Whether to save study in file (otherwise uses SQL RDB")
    args, unknown_args = parser.parse_known_args()
    training_args_dict = parse_unknown_args(unknown_args)
    objective_fn = locals()[args.objective_fn]
    optimize_hyperparameters(objective_fn, args.data_dir, "hyperparam_tuning", storage_name=args.storage_name,
                             batch_size=args.batch_size, n_trials=args.n_trials, timeout=args.timeout,
                             save_pkl=args.save_pkl, train_kwargs=training_args_dict)
