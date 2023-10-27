import copy
import time
import warnings
from typing import Dict, Any, Callable, Optional, Tuple

import numpy as np
import optuna
from optuna import Trial
from optuna.integration import TFKerasPruningCallback
from tensorflow.python.keras.callbacks import History

from tf_data_training import train_model
from utils.model_utils import NCPParams, CTRNNParams, LSTMParams, TCNParams

# args to train_model that are shared between all objective function types
COMMON_TRAIN_PARAMS = {
    "epochs": 100,
    "val_split": 0.05,
    "opt": "adam",
    "data_shift": 16,
    "data_stride": 1,
    "cached_data_dir": "cached_data",
    "save_period": 20,
}
COMMON_MODEL_PARAMS = {
    "seq_len": 64,
    "single_step": False,
    "no_norm_layer": False,
    "augmentation_params": {
        "noise": 0.05,
        "sequence_params": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
        }
    },
}


# optuna objetive functions
def ncp_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    # get trial params from bayesian optimization
    seeds_to_try = list(range(22221, 22230)) + [55555]
    ncp_seed = trial.suggest_categorical("ncp_seed", seeds_to_try)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = NCPParams(seed=ncp_seed, **COMMON_MODEL_PARAMS)
    merged_kwargs = copy.deepcopy(COMMON_TRAIN_PARAMS)
    merged_kwargs.update(**train_kwargs)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **merged_kwargs)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def cfc_objective_base(trial: Trial, ct_network_type: str, data_dir: str, batch_size: int,
                       **train_kwargs: Dict[str, Any]):
    # get trial params from bayesian optimization
    # note: could also do backbone_dr but probably not as important
    forget_bias = trial.suggest_float("forget_bias", low=.8, high=3.2)
    backbone_units = trial.suggest_int("bakcbone_units", low=64, high=256)
    backbone_layers = trial.suggest_int("backbone_layers", low=1, high=3)
    weight_decay = trial.suggest_float("weight_decay", low=1e-8, high=1e-4, log=True)
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

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
    merged_kwargs = copy.deepcopy(COMMON_TRAIN_PARAMS)
    merged_kwargs.update(**train_kwargs)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **merged_kwargs)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def cfc_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return cfc_objective_base(trial, ct_network_type="cfc", data_dir=data_dir, batch_size=batch_size, **train_kwargs)


def mixedcfc_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return cfc_objective_base(trial, ct_network_type="mixedcfc", data_dir=data_dir, batch_size=batch_size,
                              **train_kwargs)


def ctrnn_objective_base(trial: Trial, data_dir: str, batch_size: int, ct_network_type: str,
                         **train_kwargs: Dict[str, Any]):
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = CTRNNParams(rnn_sizes=[rnn_size], ct_network_type=ct_network_type, **COMMON_MODEL_PARAMS)
    merged_kwargs = copy.deepcopy(COMMON_TRAIN_PARAMS)
    merged_kwargs.update(**train_kwargs)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **merged_kwargs)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


# lots of convenience functions that call ctrnn_objective_base but have diff function names so
# they are saved in different optuna objective fxns
def ctrnn_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="ctrnn",
                                **train_kwargs)


def node_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="node",
                                **train_kwargs)


def mmrnn_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="mmrnn",
                                **train_kwargs)


def ctgru_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="ctgru",
                                **train_kwargs)


def vanilla_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="vanilla",
                                **train_kwargs)


def bidirect_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="bidirect",
                                **train_kwargs)


def grud_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="grud",
                                **train_kwargs)


def phased_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="phased",
                                **train_kwargs)


def gruode_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="gruode",
                                **train_kwargs)


def hawk_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="hawk",
                                **train_kwargs)


def ltc_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    return ctrnn_objective_base(trial=trial, data_dir=data_dir, batch_size=batch_size, ct_network_type="ltc",
                                **train_kwargs)


def lstm_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)
    # use same dropout for dropout and recurrent_dropout to avoid too many vars
    dropout = trial.suggest_float("dropout", low=0.0, high=0.3)

    lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    decay_rate = trial.suggest_float("decay_rate", 0.85, 1)

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = LSTMParams(rnn_sizes=[rnn_size], dropout=dropout, recurrent_dropout=dropout,
                              rnn_stateful=False, **COMMON_MODEL_PARAMS)
    merged_kwargs = copy.deepcopy(COMMON_TRAIN_PARAMS)
    merged_kwargs.update(**train_kwargs)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **merged_kwargs)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def tcn_objective(trial: Trial, data_dir: str, batch_size: int, **train_kwargs: Dict[str, Any]):
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

    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    model_params = TCNParams(nb_filters=nb_filters, kernel_size=kernel_size, dilations=dilations, dropout=dropout,
                             **COMMON_MODEL_PARAMS)
    merged_kwargs = copy.deepcopy(COMMON_TRAIN_PARAMS)
    merged_kwargs.update(**train_kwargs)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **merged_kwargs)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


def wiredcfccell_objective(trial: Trial, data_dir: str, batch_size: int, n_epochs: float = None, lr: float = None, decay_rate: float = None, **train_kwargs: Dict[str, Any]):
    """
    Even though wiredcfc is a type of ctrnn, it takes an additional parameter, wiring seed, so it gets a custom
    objective function
    """
    seeds_to_try = list(range(22221, 22228)) + [55555]
    wiredcfc_seed = trial.suggest_categorical("wiredcfc_seed", seeds_to_try)
    rnn_size = trial.suggest_int("rnn_size", low=64, high=256)

    if lr is None:
        lr = trial.suggest_float("lr", low=1e-5, high=1e-2, log=True)
    if decay_rate is None:
        decay_rate = trial.suggest_float("decay_rate", 0.85, 1)
    print(f"decay_rate: {decay_rate}")
    print(f"lr: {lr}")
    prune_callback = [KerasPruningCallbackFunction(trial, sum_val_train_loss)]

    if n_epochs is not None:
        COMMON_TRAIN_PARAMS["epochs"] = n_epochs
    model_params = CTRNNParams(rnn_sizes=[rnn_size], ct_network_type="wiredcfccell", wiredcfc_seed=wiredcfc_seed,
                               **COMMON_MODEL_PARAMS)
    merged_kwargs = copy.deepcopy(COMMON_TRAIN_PARAMS)
    merged_kwargs.update(**train_kwargs)
    history = train_model(lr=lr, decay_rate=decay_rate, callbacks=prune_callback,
                          model_params=model_params, data_dir=data_dir, batch_size=batch_size, **merged_kwargs)
    trial.set_user_attr("model_params", repr(model_params))

    return calculate_objective(trial, history)


# classes and functions that are called in objective functions. Can't be in hyperparameter_tuning.py or else circular
# import

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


def calculate_objective(trial: Trial, result: Tuple[History, str]):
    """
    Calculates objective value from history of losses and also logs train_loss and val_loss separately

    @param trial: optuna trial
    @param result: Tensorflow history object returned by trainer
    @return: objective value
    """
    history, time_str = result
    trial.set_user_attr("checkpoint_time_str", time_str)

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
    trial.set_user_attr("loss", history.history["loss"])
    trial.set_user_attr("val_loss", history.history["val_loss"])

    objective = loss_sums[best_epoch]
    return objective


def sum_val_train_loss(logs):
    return logs["loss"] + logs["val_loss"]
