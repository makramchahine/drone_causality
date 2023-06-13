import json
import os
from pathlib import Path
from typing import Optional, Callable, Sequence, Union, Dict, Any, Iterable

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tensorflow import Tensor
from tensorflow.python.keras.models import Functional
from tqdm import tqdm
from tensorflow.keras.layers import LSTMCell, InputLayer
from kerasncp.tf import LTCCell, WiredCfcCell

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras_models import IMAGE_SHAPE, IMAGE_SHAPE_CV
from utils.data_utils import image_dir_generator
from utils.model_utils import generate_hidden_list, NCPParams, LSTMParams, CTRNNParams, TCNParams


def run_signal(sig_model: Functional, data: Union[str, Iterable],
                      output_path: Optional[str] = None,
                      reverse_channels: bool = True,
                      sig_kwargs: Optional[Dict[str, Any]] = None) -> Sequence[ndarray]:
    """
    Runner script that loads images, runs networks, and saves signals at different layers
    """
    if sig_kwargs is None:
        sig_kwargs = {}

    if isinstance(data, str):
        data = image_dir_generator(data, IMAGE_SHAPE, reverse_channels)

    if output_path is not None:
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    control_hiddens = generate_hidden_list(sig_model, True)

    hiddens = []
    outputs = []
    for i, img in tqdm(enumerate(data)):
        out = sig_model.predict([[img,[0,1]], *control_hiddens])
        vel_cmd = out[0]
        control_hiddens = out[1:]
        # Y avait pas grand chose a faire mais fais le
        hidtemp =[]
        for j in range(len(control_hiddens)):
            hidtemp.extend(control_hiddens[j][0])
        hiddens.append(hidtemp)
        outputs.append(vel_cmd[0])

    d = {'output': outputs, 'hidden': hiddens}
    df = pd.DataFrame(data=d)
    ceesve = "arqam.csv"
    df.to_csv(ceesve, index=False)

    return


def parse_params_json(params_path: str, set_single_step: bool = True):
    with open(params_path, "r") as f:
        params_data = json.loads(f.read())

    for local_path, params_str in params_data.items():
        model_params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams, None] = eval(params_str)
        if set_single_step:
            model_params.single_step = True
        model_path = os.path.join(os.path.dirname(params_path), local_path)
        yield local_path, model_path, model_params

def get_decision_model(model: Functional):
    # cleave off only convolutional head

    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]

    act_model_inputs = model.inputs[0]  # don't want to take in hidden state, just image
    activation_model = keras.models.Model(inputs=act_model_inputs,
                                          outputs=[layer.output for layer in conv_layers])
    return activation_model