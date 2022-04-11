# Created by Patrick Kao at 4/4/22
import argparse
import os
from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import ndarray
from tqdm import tqdm

from analysis.vis_utils import parse_params_json
from keras_models import IMAGE_SHAPE
from utils.data_utils import image_dir_generator
from utils.model_utils import ModelParams, load_model_from_weights, generate_hidden_list, get_readable_name, TCNParams


def calculate_lipschitz_constant(model_path: str, model_params: ModelParams, sequence_path: str) -> Sequence[ndarray]:
    model = load_model_from_weights(model_params, model_path)
    hiddens = generate_hidden_list(model=model, return_numpy=True)
    all_hiddens = []  # list of list of arrays with shape num_timesteps x num_hiddens x hidden_dim
    for i, img in tqdm(enumerate(image_dir_generator(sequence_path, IMAGE_SHAPE))):
        img_batched_tensor = tf.expand_dims(img, axis=0)
        all_hiddens.append(hiddens)
        out = model.predict([img_batched_tensor, *hiddens])
        hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim

    # crete list with same shape as hidden vectors where contents are lipschitz values of each dimension
    lip = [np.zeros_like(h) for h in all_hiddens[0]]
    for i in range(len(all_hiddens) - 1):
        current_hiddens = all_hiddens[i]
        next_hiddens = all_hiddens[i + 1]
        diff = [np.abs(n - c) for n, c in zip(next_hiddens, current_hiddens)]
        lip = [np.maximum(l, d) for l, d in zip(lip, diff)]
    return lip


def graph_lipschitz_constant(lip_hiddens: Sequence[ndarray], model_name: str = "", display_result: bool = False,
                             save_path: Optional[str] = None):
    # concat all hidden dims into 1d
    flattened = [lip.flatten() for lip in lip_hiddens]
    all_hiddens = np.concatenate(flattened, axis=0)

    lip_sorted = np.sort(all_hiddens)
    plt.plot(lip_sorted)
    plt.xlabel("Node Rank")
    plt.ylabel("Lipschitz Constant of Hidden State Nodes")

    if model_name:
        plt.title(model_name)

    # note that save needs to happen before plt.show() b/c plt.show() clears figures
    if save_path is not None:
        plt.savefig(save_path)

    if display_result:
        plt.show()


def params_lipschitz_constant(params_path: str, sequence_path: str, display_result: bool = False,
                              save_dir: Optional[str] = None) -> Sequence[float]:
    model_lipschitz = {}
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for local_path, model_path, model_params in parse_params_json(params_path):
        if isinstance(model_params, TCNParams):
            print("Skipping TCN")
            continue

        model_name = get_readable_name(model_params)
        lip = calculate_lipschitz_constant(model_path, model_params, sequence_path)
        model_lipschitz[model_name] = lip
        if display_result or save_dir is not None:
            save_path = os.path.join(save_dir, model_name) if save_dir is not None else None
            graph_lipschitz_constant(lip_hiddens=lip, model_name=model_name, display_result=display_result,
                                     save_path=save_path)

    return model_lipschitz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path")
    parser.add_argument("sequence_path")
    parser.add_argument("--display_result", action="store_true")
    parser.add_argument("--save_dir", type=str, default="lipschitz_out")
    args = parser.parse_args()
    params_lipschitz_constant(args.params_path, args.sequence_path, args.display_result, args.save_dir)
