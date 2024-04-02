#!/usr/bin/env python
import json
# import files from hyperparameter_tuning.py one dir up
import os.path
from enum import Enum
from pathlib import Path
from typing import Union

from optuna.trial import FixedTrial

from hyperparameter_tuning import *
# noinspection PyUnresolvedReferences
from utils.model_utils import get_readable_name
from utils.objective_functions import *

# reset script dir since hyperparameter_tuning.py sets it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class StorageType(Enum):
    RDB = 0
    PKL = 1
    JSON = 2


def should_stop(start_time: float, timeout: Optional[float]):
    return timeout is not None and time.time() - start_time > timeout


def get_prev_trains(out_dir: str, study_name_network: str):
    num_prev_trains = 0
    for file in os.listdir(out_dir):
        if file.endswith(".json") and study_name_network in file:
            num_prev_trains += 1
    return num_prev_trains


def find_hotstart_checkpoint(search_type: str, checkpoint_dir: str, require_equality: bool = True):
    """
    Of all the models in params.json in checkpoint dir, finds the candidate model that has the same model type as
    candidate type based on the model params and returns its absolute checkpoint path

    :param require_equality: if false, doensn't check for strict equality and just sees if search_type is IN the candidate
    :param search_type: string returned by get_readable_name that specifies which class of model to look for
    :param checkpoint_dir: base dir for models (should be 2 levels up from train/params.json)
    :return:
    """
    hotstart_dir = os.path.join(checkpoint_dir, "train")
    hotstart_json = os.path.join(hotstart_dir, "params.json")
    with open(hotstart_json, "r") as f:
        hotstart_params = json.load(f)

    candidate_checkpoint = None
    for checkpoint_path, cand_params in hotstart_params.items():
        cand_type = get_readable_name(cand_params)
        satisfies = cand_type == search_type if require_equality else search_type in cand_type
        if satisfies:
            assert candidate_checkpoint is None, f"Only one checkpoint should match the parameter type, but {candidate_checkpoint} also matches"
            candidate_checkpoint = checkpoint_path

    assert candidate_checkpoint is not None, "No matching candidate checkpoint found"
    return os.path.abspath(os.path.join(hotstart_dir, candidate_checkpoint))


def train_multiple(obj_fn: Callable, data_dir: str, study_name: str, n_trains: int, batch_size: int,
                   storage_name: str = "sqlite:///hyperparam_tuning.db",
                   storage_type: Union[str, StorageType] = StorageType.RDB, out_dir: str = "",
                   timeout: Optional[float] = None, train_kwargs: Optional[Dict[str, Any]] = None,
                   hotstart_dir: Optional[str] = None, n_epochs: int = None, decay_rate: float = None, lr: float = None, data_shift: int = None):
    """
    Runs obj_fn with the best parameters from study_name in storage_time
    """
    # convert to enum if in string form
    if isinstance(storage_type, str):
        storage_type = StorageType[storage_type.upper()]

    if train_kwargs is None:
        # default value
        train_kwargs = {}

    start_time = time.time()
    # create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name_network = f"{study_name}{obj_fn.__name__}"
    # TODO: FIX hardcoding CFC
    # study_name_network = study_name_network.replace("lem", "wiredcfc")
    study_name_network = study_name_network.replace("debug_ltc", "wiredcfc")


    path_relative = os.path.join(SCRIPT_DIR, storage_name)
    if storage_type == StorageType.PKL:
        if os.path.exists(path_relative):
            study = joblib.load(path_relative)
            best_trial = study.best_trial
        else:
            raise ValueError(f"No study found at {path_relative}")
    elif storage_type == StorageType.RDB:
        study = optuna.create_study(storage=storage_name, study_name=study_name_network, load_if_exists=True)
        best_trial = study.best_trial
    elif storage_type == StorageType.JSON:
        with open(path_relative, "r") as f:
            params = json.load(f)
            best_trial = FixedTrial(params)
    else:
        raise ValueError(f"Unsupported storage type {storage_type}")

    # find hotstart checkpoint
    if hotstart_dir is not None:
        try:
            model_type = get_readable_name(best_trial.user_attrs["model_params"])
            hotstart_checkpoint = find_hotstart_checkpoint(model_type, hotstart_dir)
        except KeyError:
            # loaded from json with no model params. Use objective function name instead
            # note: because this only works with "in", likely won't work if cfc used as both mixedcfc and cfc will match
            model_type = obj_fn.__name__.replace("_objective", "")
            # try ctrnn version first
            try:
                hotstart_checkpoint = find_hotstart_checkpoint(f"ctrnn_{model_type}", hotstart_dir)
            except AssertionError:
                hotstart_checkpoint = find_hotstart_checkpoint(model_type, hotstart_dir)

        print(f"Found hotstart checkpoint {hotstart_checkpoint}")
        train_kwargs["hotstart"] = hotstart_checkpoint

    train_kwargs["save_period"] = 1
    obj_filled = functools.partial(objective_fn, lr=lr, decay_rate=decay_rate, n_epochs=n_epochs, data_dir=data_dir, batch_size=batch_size, **train_kwargs)

    # account for previous trains in n_trains counting
    num_prev_trains = get_prev_trains(out_dir, study_name_network)
    print(f"Discovered {num_prev_trains} existing training runs in {out_dir}")

    while num_prev_trains < n_trains:
        if should_stop(start_time, timeout):
            # separate clause for unique print
            print(f"Time limit reached. Script has been running for {time.time() - start_time} seconds.")
            break
        # open file early to indicate to other processes training in progress
        out_filename = os.path.join(out_dir, f"{study_name_network}_{num_prev_trains}_train_results.json")
        with open(out_filename, "w") as outfile:
            print(f"Starting train {num_prev_trains}")
            trial_instance = copy.deepcopy(best_trial)
            obj_value = obj_filled(trial_instance)
            run_results = copy.deepcopy(trial_instance.user_attrs)
            run_results["obj_value"] = obj_value
            run_results["train_kwargs"] = train_kwargs
            json.dump(run_results, outfile)
        num_prev_trains = get_prev_trains(out_dir, study_name_network)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter bayesian optimization on deepdrone data')
    parser.add_argument("objective_fn", type=str, help="Name of objective function in this file to run")
    parser.add_argument("data_dir", type=str, help="Folder path of dataset")
    parser.add_argument("--n_epochs", type=int, default=None, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--decay_rate", type=float, default=None, help="Learning rate decay rate")
    parser.add_argument("--data_shift", type=int, default=None, help="Number of frames to shift data by")
    parser.add_argument("--n_trains", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size for training")
    parser.add_argument("--storage_name", type=str, default="sqlite:///hyperparam_tuning.db",
                        help="Filepath for pkl, or URL for SQL RDB")
    parser.add_argument("--storage_type", type=str, default="rdb",
                        help="how to load study/params, either 'rdb', 'pkl', or 'json'")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Time in seconds such that if the script has been running for longer than this, no trains"
                             "are started")
    parser.add_argument("--out_dir", type=str, default="", help="Directory to store logs into")
    parser.add_argument("--study_name", type=str, default="", help="Prefix added to study name before objective")
    parser.add_argument("--hotstart_dir", type=str, default=None, help="Directory to look for hotstart checkpoint"
                                                                       "with same type as trained model")
    args, unknown_args = parser.parse_known_args()
    training_args_dict = parse_unknown_args(unknown_args)
    objective_fn = locals()[args.objective_fn]
    train_multiple(objective_fn, args.data_dir, args.study_name, n_trains=args.n_trains, batch_size=args.batch_size,
                   storage_name=args.storage_name, storage_type=args.storage_type, timeout=args.timeout,
                   train_kwargs=training_args_dict, out_dir=args.out_dir, hotstart_dir=args.hotstart_dir,
                   n_epochs=args.n_epochs,decay_rate=args.decay_rate, lr=args.lr, data_shift=args.data_shift)