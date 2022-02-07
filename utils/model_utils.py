import copy
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Optional, List, Iterable

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras.models import Functional

from keras_models import IMAGE_SHAPE, DEFAULT_CFC_CONFIG, generate_ncp_model, generate_ctrnn_model, generate_lstm_model, \
    generate_tcn_model, DEFAULT_NCP_SEED


# helper classes that contain all the parameters in the generate_*_model functions
@dataclass
class ModelParams:
    # dataclasses can't have non-default follow default
    seq_len: int = field(default=False, init=True)
    image_shape: Tuple[int, int, int] = IMAGE_SHAPE
    augmentation_params: Dict = None
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
    if isinstance(params, NCPParams):
        model_skeleton = generate_ncp_model(**asdict(params))
    elif isinstance(params, CTRNNParams):
        model_skeleton = generate_ctrnn_model(**asdict(params))
    elif isinstance(params, LSTMParams):
        model_skeleton = generate_lstm_model(**asdict(params))
    elif isinstance(params, TCNParams):
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
        except TypeError:
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


def get_readable_name(params: ModelParams):
    """
    Extracts the model name from the class of params
    """
    class_name = str(params.__class__.__name__)
    return class_name.replace("Params", "").lower()


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
    for input_shape in model.input_shape[1:]:  # ignore 1st output, as is this control output
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
