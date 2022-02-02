#!/usr/bin/env python
import copy
import json
import os

# import files from hyperparameter_tuning.py one dir up
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from hyperparameter_tuning import *

# reset script dir since hyperparameter_tuning.py sets it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_multiple(obj_fn: Callable, data_dir: str, study_name: str, n_trains: int, batch_size: int,
                   storage_name: str = "sqlite:///hyperparam_tuning.db", save_pkl: bool = False, out_prefix: str = ""):
    """
    Runs obj_fn with the best parameters from study_name in storage_time
    """
    # create output directory
    out_dir = os.path.join(SCRIPT_DIR, out_prefix)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name_network = f"{study_name}_{obj_fn.__name__}"

    if save_pkl:
        path_relative = os.path.join(SCRIPT_DIR, storage_name)
        if os.path.exists(path_relative):
            study = joblib.load(path_relative)
        else:
            raise ValueError(f"No study found at {path_relative}")
    else:
        study = optuna.create_study(storage=storage_name, study_name=study_name_network, load_if_exists=True)

    obj_filled = functools.partial(objective_fn, data_dir=data_dir, batch_size=batch_size)

    # account for previous trains in n_trains counting
    num_prev_trains = 0
    for file in os.listdir(out_dir):
        if file.endswith(".json"):
            num_prev_trains += 1
    print(f"Discovered {num_prev_trains} existing training runs in {out_dir}")

    while num_prev_trains < n_trains:
        trial_instance = copy.deepcopy(study.best_trial)
        obj_value = obj_filled(trial_instance)
        out_filename = os.path.join(out_dir, f"{study_name_network}_{i}_train_results.json")
        run_results = copy.deepcopy(trial_instance.user_attrs)
        run_results["obj_value"] = obj_value
        with open(out_filename, "w") as outfile:
            json.dump(run_results, outfile)
        num_prev_trains += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter bayesian optimization on deepdrone data')
    parser.add_argument("objective_fn", type=str, help="Name of objective function in this file to run")
    parser.add_argument("data_dir", type=str, help="Folder path of dataset")
    parser.add_argument("--n_trains", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for training")
    parser.add_argument("--storage_name", type=str, default="sqlite:///hyperparam_tuning.db",
                        help="Filepath for pkl, or URL for SQL RDB")
    parser.add_argument("--save_pkl", action='store_true',
                        help="Whether to save study in file (otherwise uses SQL RDB")
    args = parser.parse_args()

    objective_fn = locals()[args.objective_fn]
    train_multiple(objective_fn, args.data_dir, "hyperparam_tuning", storage_name=args.storage_name,
                   batch_size=args.batch_size, n_trains=args.n_trains, save_pkl=args.save_pkl)
