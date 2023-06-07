import copy
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Optional, List, Iterable, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras.models import Functional

import sys
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, ".."))

from keras_models import IMAGE_SHAPE, DEFAULT_CFC_CONFIG, generate_ncp_model, generate_ctrnn_model, generate_lstm_model, \
    generate_tcn_model, DEFAULT_NCP_SEED


# helper classes that contain all the parameters in the generate_*_model functions
@dataclass
class ModelParams:
    # dataclasses can't have non-default follow default
    seq_len: int = field(default=False, init=True)
    image_shape: Tuple[int, int, int] = IMAGE_SHAPE
    augmentation_params: Optional[Dict] = None
    batch_size: Optional[int] = None
    single_step: bool = False
    no_norm_layer: bool = False


@dataclass
class NCPParams(ModelParams):
    seed: int = DEFAULT_NCP_SEED


@dataclass
class LSTMParams(ModelParams):
    rnn_sizes: List[int] = field(default=False, init=True)
    dropout: float = 0.1
    recurrent_dropout: float = 0.1
    rnn_stateful: bool = False


@dataclass
class CTRNNParams(ModelParams):
    rnn_sizes: List[int] = field(default=False, init=True)
    ct_network_type: str = 'ctrnn'
    config: Dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_CFC_CONFIG))
    rnn_stateful: bool = False
    wiredcfc_seed: int = DEFAULT_NCP_SEED


@dataclass
class TCNParams(ModelParams):
    nb_filters: int = field(default=False, init=True)
    kernel_size: int = field(default=False, init=True)
    dilations: Iterable[int] = field(default=False, init=True)
    dropout: float = 0.1


def get_skeleton(params: ModelParams):
    """
    Returns a new model with randomized weights according to the parameters in params
    """
    if isinstance(params, NCPParams) or "NCPParams" in params.__class__.__name__:
        model_skeleton = generate_ncp_model(**asdict(params))
    elif isinstance(params, CTRNNParams) or "CTRNNParams" in params.__class__.__name__:
        model_skeleton = generate_ctrnn_model(**asdict(params))
    elif isinstance(params, LSTMParams) or "LSTMParams" in params.__class__.__name__:
        model_skeleton = generate_lstm_model(**asdict(params))
    elif isinstance(params, TCNParams) or "TCNParams" in params.__class__.__name__:
        model_skeleton = generate_tcn_model(**asdict(params))
    else:
        raise ValueError(f"Could not parse param type {params.__class__}")
    return model_skeleton


def load_model_from_weights(params: ModelParams, checkpoint_path: str, load_name_ok: bool = False):
    """
    Convenience function that loads weights from checkpoint_path into model_skeleton
    """
    model_skeleton = get_skeleton(params)
    if load_name_ok:
        try:
            model_skeleton.load_weights(checkpoint_path)
        except ValueError:
            # different number of weights from file and model. Assume normalization layer in model but not file
            # rename conv layers starting at 5
            print("Model had incorrect number of layers. Attempting to load from layer names")
            conv_index = 5
            dense_index = 1
            for layer in model_skeleton.layers:
                if isinstance(layer, Conv2D):
                    layer._name = f"conv2d_{conv_index}"
                    conv_index += 1
                elif isinstance(layer, Dense):
                    layer._name = f"dense_{dense_index}"
                    dense_index += 1
            model_skeleton.load_weights(checkpoint_path, by_name=True)
    else:
        model_skeleton.load_weights(checkpoint_path)

    return model_skeleton


def load_model_no_params(checkpoint_path: str, single_step: bool):
    """
    Convenience function that calls load_model_from weights as above but tries to infer reasonable default params if not
    known
    """
    if 'ncp' in checkpoint_path:
        params = NCPParams(seq_len=64, single_step=single_step)
    elif 'mixedcfc' in checkpoint_path:
        params = CTRNNParams(seq_len=64, rnn_sizes=[128], ct_network_type="mixedcfc", single_step=single_step)
    elif 'lstm' in checkpoint_path:
        params = LSTMParams(seq_len=64, rnn_sizes=[128], single_step=single_step)
    elif "tcn" in checkpoint_path:
        params = TCNParams(seq_len=64, nb_filters=128, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32])
    else:
        raise ValueError(f"Unable to infer model name from path {checkpoint_path}")

    return load_model_from_weights(params, checkpoint_path)


COMPATIBILITY_REPLACEMENTS = {
    "do_normalization=False,": "",
    "do_normalization=True,": "",
    "do_augmentation=False,": "",
    "do_augmentation=True,": "",
    "data=None,": "",
    "time_distributed=True,": "",
    "rnn_stateful=False,": "",
}


def eval_model_params(param_str: str):
    """
    Converts a string-serialized model params instance into its respective ModelParams subclass
    :param param_str:
    :return:
    """
    try:
        params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams] = eval(param_str)
    except TypeError:
        # TODO: fix in a way that doesn't involve manual dict objects/string mod (dataclass takes other args?)
        print("Could not parse param string into object. Trying to replace deprecated options")
        for bad_str, replace_str in COMPATIBILITY_REPLACEMENTS.items():
            param_str = param_str.replace(bad_str, replace_str)
        params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams] = eval(param_str)

    return params


def get_readable_name(params: Union[ModelParams, str]):
    """
    Extracts the model name from the class of params
    """
    if isinstance(params, str):
        params = eval_model_params(params)
    class_name = str(params.__class__.__name__)
    name = class_name.replace("Params", "").lower()
    if isinstance(params, CTRNNParams):
        # include ctrnn type in name
        name = f"{name}_{params.ct_network_type}"
    return name


def generate_hidden_list(model: Functional, return_numpy: bool = True):
    """
    Generates a list of tensors that are used as the hidden state for the argument model when it is used in single-step
    mode. The batch dimension (0th dimension) is assumed to be 1 and any other dimensions (seq len dimensions) are
    assumed to be 0

    :param return_numpy: Whether to return output as numpy array. If false, returns as keras tensor
    :param model: Single step functional model to infer hidden states for
    :return: list of hidden states with 0 as value
    """
    constructor = np.zeros if return_numpy else tf.zeros
    hiddens = []
    if len(model.input_shape)==1:
        lool = model.input_shape[0][1:]
    else:
        lool = model.input_shape[2:]

    for input_shape in lool:  # ignore 1st output, as is this control output
        hidden = []
        for i, shape in enumerate(input_shape):
            if shape is None:
                if i == 0:  # batch dim
                    hidden.append(1)
                    continue
                elif i == 1:  # seq len dim
                    hidden.append(0)
                    continue
                else:
                    print("Unable to infer hidden state shape. Leaving as none")
            hidden.append(shape)
        hiddens.append(constructor(hidden))
    return hiddens


def get_params_from_json(params_path: str, checkpoint_path: str):
    with open(params_path, "r") as f:
        data = json.loads(f.read())
        model_params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams] = eval(
            data[os.path.basename(checkpoint_path)])
        return model_params
