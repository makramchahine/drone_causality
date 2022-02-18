import argparse
import json
import os
from os import listdir
from os.path import isfile
from pathlib import Path
from typing import Optional, Union

from tensorflow import keras
from tensorflow.python.keras.layers import Normalization, Rescaling

from utils.model_utils import NCPParams, LSTMParams, CTRNNParams, TCNParams, load_model_from_weights


def remove_norm_rescaling_layers(checkpoint_path: str, params_path: str, dest_path: Optional[str] = None):
    if dest_path is None:
        dir_path, filename = os.path.split(checkpoint_path)
        dest_path = os.path.join(dir_path, "headless", filename)

    # create dest_path if it doesn't exist
    Path(os.path.dirname(dest_path)).mkdir(parents=False, exist_ok=True)

    # get model params and load model
    with open(params_path, "r") as f:
        data = json.loads(f.read())
        model_params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams] = eval(data[os.path.basename(checkpoint_path)])
        model_params.single_step = False

    # even though will be used in single-step mode, easier to recreate in sequential mode and weights are the same
    # either way
    model = load_model_from_weights(model_params, checkpoint_path)

    # remove rescaling, normalization layers
    input_layer = model.input
    x = input_layer

    to_drop = [0,1]
    for i, layer in enumerate(model.layers[1:]):
        if i not in to_drop:
            x = layer(x)

    trimmed_model = keras.Model(inputs=input_layer, outputs=x)
    trimmed_model.save_weights(dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("params_path", type=str)
    parser.add_argument("--dest_path", type=str, default=None)
    args = parser.parse_args()
    for child in listdir(args.checkpoint_dir):
        # filter only model weights
        if isfile(os.path.join(args.checkpoint_dir, child)) and "hdf5" in child:
            remove_norm_rescaling_layers(os.path.join(args.checkpoint_dir, child), args.params_path, args.dest_path)
