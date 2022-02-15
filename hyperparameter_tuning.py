#!/usr/bin/env python
import argparse
import functools
import logging
import os
import sys
import time
import warnings
from typing import Callable, Optional, Dict

import joblib
import numpy as np
import optuna
from optuna import Trial
from optuna.integration import TFKerasPruningCallback
# add directory up to path to get main naming script
from optuna.pruners import MedianPruner

from tf_data_training import train_model
from utils.model_utils import NCPParams, LSTMParams, CTRNNParams, TCNParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class KerasPruningCallbackFunction(TFKerasPruningCallback):
    """
    Convenience class that allows pruning based on any function of the logs, instead of just looking at 1
    log metric
    """

    def __init__(self, trial: optuna.trial.Trial, get_objective: Callable) -> None:
        super().__init__(trial, "")
        self.get_objective = get_objective

    # copied from optuna/integration/keras.py
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        logs = logs or {}
        current_score = self.get_objective(logs)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self._monitor)
            )
            warnings.warn(message)
            return
        # logging a nan obj leads to crash
        if np.isnan(current_score):
            message = f"Trial was pruned at epoch {epoch} because objective value was NaN"
            raise optuna.TrialPruned(message)

        self._trial.report(float(current_score), step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def calculate_objective(trial: Trial, history):
    """
    Calculates objective value from history of losses and also logs train_loss and val_loss separately

    @param trial: optuna trial
    @param history: Tensorflow history object returned by trainer
    @return: objective value
    """
    losses = np.array([[epoch_train_loss, epoch_val_loss] for epoch_train_loss, epoch_val_loss in
                       zip(history.history["loss"], history.history["val_loss"])])
    loss_sums = losses.sum(axis=1)
    best_epoch = np.argmin(loss_sums)
    trial.set_user_attr("sum_train_loss", losses[best_epoch, 0])
    trial.set_user_attr("sum_val_loss", losses[best_epoch, 1])
    trial.set_user_attr("best_sum_epoch", int(best_epoch))

    # calculate best train and val epochs
    best_train = np.argmin(losses[:, 0])
    trial.set_user_attr("best_train_epoch", int(best_train))
    trial.set_user_attr("best_train_loss", losses[best_train, 0])
    best_val = np.argmin(losses[:, 1])
    trial.set_user_attr("best_val_epoch", int(best_val))
    trial.set_user_attr("best_val_loss", losses[best_val, 1])

    trial.set_user_attr("trial_time", time.time())

    objective = loss_sums[best_epoch]
    return objective


# args to train_model that are shared between all objective function types
COMMON_TRAIN_PARAMS = {
    "epochs": 100,
    "val_split": 0.05,
    "opt": "adam",
    "data_shift": 16,
    "data_stride": 1,
    "cached_data_dir": "cached_data"
}

COMMON_MODEL_PARAMS = {
    "seq_len": 64,
    "single_step": False,
    "no_norm_layer": False,
}


# optuna objetive functions
def ncp_objective(trial: Trial, data_dir: str, batch_size: int):
    # get trial params from bayesian optimization
    seeds_to_try = list(range(22221, 22230)) + [55555]
    ncp_seed = trial.suggest_categorical("ncp_seed", seeds_to_try)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    def sum_val_train_loss(logs):
        return logs["loss"] + logs["val_loss"]

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = NCPParams(seed=ncp_seed, **COMMON_MODEL_PARAMS)
    # note rnn_size not needed for ncp
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **COMMON_TRAIN_PARAMS)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def cfc_objective_base(trial: Trial, ct_network_type: str, data_dir: str, batch_size: int):
    # get trial params from bayesian optimization
    # note: could also do backbone_dr but probably not as important
    forget_bias = trial.suggest_float("forget_bias", low=.8, high=3.2)
    backbone_units = trial.suggest_int("bakcbone_units", low=64, high=256)
    backbone_layers = trial.suggest_int("backbone_layers", low=1, high=3)
    weight_decay = trial.suggest_float("weight_decay", low=1e-8, high=1e-4, log=True)
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    def sum_val_train_loss(logs):
        return logs["loss"] + logs["val_loss"]

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    cfc_config = {
        "clipnorm": 1,
        "backbone_activation": "silu",
        "backbone_dr": 0.1,
        "forget_bias": forget_bias,
        "backbone_units": backbone_units,
        "backbone_layers": backbone_layers,
        "weight_decay": weight_decay
    }

    model_params = CTRNNParams(rnn_sizes=[rnn_size], ct_network_type=ct_network_type, config=cfc_config,
                               **COMMON_MODEL_PARAMS)
    # note rnn_size not needed for ncp
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **COMMON_TRAIN_PARAMS)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


# define in global scope for command line calls
def cfc_objective(trial: Trial, data_dir: str, batch_size: int):
    return cfc_objective_base(trial, ct_network_type="cfc", data_dir=data_dir, batch_size=batch_size, )


def mixedcfc_objective(trial: Trial, data_dir: str, batch_size: int):
    return cfc_objective_base(trial, ct_network_type="mixedcfc", data_dir=data_dir, batch_size=batch_size, )


def ctrnn_objective_base(trial: Trial, data_dir: str, batch_size: int, ct_network_type: str):
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    def sum_val_train_loss(logs):
        return logs["loss"] + logs["val_loss"]

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = CTRNNParams(rnn_sizes=[rnn_size], ct_network_type=ct_network_type, **COMMON_MODEL_PARAMS)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **COMMON_TRAIN_PARAMS)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


# lots of convenience functions that call ctrnn_objective_base but have diff function names so
# they are saved in different otptuna trials
def ctrnn_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="ctrnn")


def node_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="node")


def mmrnn_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="mmrnn")


def ctgru_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="ctgru")


def vanilla_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="vanilla")


def bidirect_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="bidirect")


def grud_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="grud")


def phased_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="phased")


def gruode_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="gruode")


def hawk_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="hawk")


def ltc_objective(trial: Trial, data_dir: str, batch_size: int):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="ltc")


def lstm_objective(trial: Trial, data_dir: str, batch_size: int):
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)
    # use same dropout for dropout and recurrent_dropout to avoid too many vars
    dropout = trial.suggest_float("dropout", low=0.0, high=0.3)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    def sum_val_train_loss(logs):
        return logs["loss"] + logs["val_loss"]

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = LSTMParams(rnn_sizes=[rnn_size], dropout=dropout, recurrent_dropout=dropout,
                              rnn_stateful=False, **COMMON_MODEL_PARAMS)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **COMMON_TRAIN_PARAMS)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def tcn_objective(trial: Trial, data_dir: str, batch_size: int):
    KERNEL_DILATION_CONFIGS = [
        {"kernel_size": 2, "dilations": [1, 2, 4, 8, 16]},
        {"kernel_size": 3, "dilations": [1, 2, 4, 8]},
        {"kernel_size": 5, "dilations": [1, 2, 4]},
    ]
    nb_filters = trial.suggest_int("nb_filters", low=64, high=256)
    config_number = trial.suggest_categorical("kernel_dilation_config", range(3))
    kernel_dilation_config = KERNEL_DILATION_CONFIGS[config_number]
    kernel_size = kernel_dilation_config["kernel_size"]
    dilations = kernel_dilation_config["dilations"]
    dropout = trial.suggest_float("dropout", low=0.0, high=0.3)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    def sum_val_train_loss(logs):
        return logs["loss"] + logs["val_loss"]

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = TCNParams(nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations, dropout=dropout,
                             **COMMON_MODEL_PARAMS)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **COMMON_TRAIN_PARAMS)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def wiredcfccell_objective(trial: Trial, data_dir: str, batch_size: int):
    """
    Even though wiredcfc is a type of ctrnn, it takes an additional parameter, wiring seed, so it gets a custom
    objective function
    """
    seeds_to_try = list(range(22221, 22228)) + [55555]
    wiredcfc_seed = trial.suggest_categorical("wiredcfc_seed", seeds_to_try)
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    def sum_val_train_loss(logs):
        return logs["loss"] + logs["val_loss"]

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = CTRNNParams(rnn_sizes=[rnn_size], ct_network_type="wiredcfccell", wiredcfc_seed=wiredcfc_seed,
                               **COMMON_MODEL_PARAMS)
    # note rnn_size not needed for ncp
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **COMMON_TRAIN_PARAMS)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def optimize_hyperparameters(obj_fn: Callable, data_dir: str, study_name: str, n_trials: int, batch_size: int,
                             timeout: float = None, storage_name: str = "sqlite:///hyperparam_tuning.db",
                             save_pkl: bool = False):
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

    study.optimize(functools.partial(objective_fn, data_dir=data_dir, batch_size=batch_size), n_trials=n_trials,
                   timeout=timeout)

    if save_pkl:
        joblib.dump(study, storage_name)
    return study


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
    args = parser.parse_args()

    objective_fn = locals()[args.objective_fn]
    optimize_hyperparameters(objective_fn, args.data_dir, "hyperparam_tuning", storage_name=args.storage_name,
                             batch_size=args.batch_size, n_trials=args.n_trials, timeout=args.timeout,
                             save_pkl=args.save_pkl)
