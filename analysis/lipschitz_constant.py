# Created by Patrick Kao at 4/4/22
import argparse
import json
import os
from pathlib import Path
from typing import Sequence, Optional, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm import tqdm

from keras_models import IMAGE_SHAPE
from utils.data_utils import image_dir_generator
from utils.model_utils import ModelParams, load_model_from_weights, generate_hidden_list, get_readable_name
from utils.vis_utils import parse_params_json


def calculate_lipschitz_constant(model_path: str, model_params: ModelParams, sequence_path: str,
                                 reverse_channels: bool) -> Sequence[ndarray]:
    model = load_model_from_weights(model_params, model_path)
    hiddens = generate_hidden_list(model=model, return_numpy=True)
    all_hiddens = []  # list of list of arrays with shape num_timesteps x num_hiddens x hidden_dim
    for i, img in tqdm(enumerate(image_dir_generator(sequence_path, IMAGE_SHAPE, reverse_channels))):
        all_hiddens.append(hiddens)
        out = model.predict([img, *hiddens])
        hiddens = out[1:]  # list num_hidden long, each el is hidden_dim,

    # flatten batch dim
    all_hiddens = [[np.squeeze(hid, axis=0) for hid in step_hid] for step_hid in all_hiddens]
    # crete list with same shape as hidden vectors where contents are lipschitz values of each dimension
    lip = [np.zeros_like(h) for h in all_hiddens[0]]
    for i in range(len(all_hiddens) - 1):
        current_hiddens = all_hiddens[i]
        next_hiddens = all_hiddens[i + 1]
        diff = [np.abs(n - c) for n, c in zip(next_hiddens, current_hiddens)]
        lip = [np.maximum(l, d) for l, d in zip(lip, diff)]
    return lip


def graph_lipschitz_constant(lip_mean: Dict[str, ndarray], lip_std: Optional[Dict[str, ndarray]],
                             display_result: bool = False,
                             save_path: Optional[str] = None):
    plt.clf()
    plt.xlabel("Node Rank")
    plt.ylabel("Lipschitz Constant of Hidden State Nodes")

    # concat all hidden dims into 1d
    for model_name, hiddens in lip_mean.items():
        # hiddens shape: flattened_hidden_dim
        sort_order = np.argsort(hiddens)

        lip_sorted = hiddens[sort_order]
        lip_x = np.linspace(0, 1, num=len(lip_sorted))
        plt.plot(lip_x, lip_sorted, label=model_name)
        if lip_std is not None:
            std = lip_std[model_name]
            std_sorted = std[sort_order]
            plt.fill_between(lip_x, lip_sorted + std_sorted, lip_sorted - std_sorted, alpha=0.5)

    plt.legend(loc="upper left")
    # note that save needs to happen before plt.show() b/c plt.show() clears figures
    if save_path is not None:
        plt.savefig(save_path)

    if display_result:
        plt.show()


def params_lipschitz_constant(datasets_json: str, params_path: str, display_result: bool = False,
                              save_dir: Optional[str] = None) -> Tuple[Dict, Dict]:
    with open(datasets_json, "r") as f:
        datasets: Dict[str, Tuple[str, bool]] = json.load(f)

    all_mean = {}
    all_std = {}
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for local_path, model_path, model_params in parse_params_json(params_path):
        model_name = get_readable_name(model_params)
        if model_name == "tcn":
            # TCN has no hidden state
            continue

        lips = []
        for dataset_name, (data_path, reverse_channels, csv_path) in datasets.items():
            lip = calculate_lipschitz_constant(model_path, model_params, data_path, reverse_channels=reverse_channels)
            lips.append(lip)

        # lips shape: num_datasets x num_hiddens x hidden dim
        # shape : num_dataset x flattened_hidden_dim
        lips_flat = np.array([np.hstack(dataset) for dataset in lips])
        lip_mean = np.mean(lips_flat, axis=0)
        lip_std = np.std(lips_flat, axis=0)
        model_lip_mean = {model_name: lip_mean}
        model_lip_std = {model_name: lip_std}
        if display_result or save_dir is not None:
            save_path = os.path.join(save_dir, model_name) if save_dir is not None else None
            graph_lipschitz_constant(lip_mean=model_lip_mean, lip_std=model_lip_std, display_result=display_result,
                                     save_path=save_path)

        all_mean.update(model_lip_mean)
        all_std.update(model_lip_std)

    # graph all lipschitz
    if display_result or save_dir is not None:
        graph_lipschitz_constant(lip_mean=all_mean, lip_std=all_std, display_result=display_result,
                                 save_path=os.path.join(save_dir, "all_lipschitz"))

    if save_dir is not None:
        with open(os.path.join(save_dir, "lip_data.json"), "w") as f:
            lip_data = {"means": convert_values_to_list(all_mean), "stds": convert_values_to_list(all_std)}
            json.dump(lip_data, f)

    return all_mean, all_std


def convert_values_to_list(to_convert: Dict[str, ndarray]):
    to_ret = {}
    for key, np_arr in to_convert.items():
        to_ret[key] = list(np_arr)

    return to_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_json")
    parser.add_argument("params_path")
    parser.add_argument("--display_result", action="store_true")
    parser.add_argument("--save_dir", type=str, default="lipschitz_out")
    args = parser.parse_args()
    params_lipschitz_constant(datasets_json=args.datasets_json, params_path=args.params_path,
                              display_result=args.display_result, save_dir=args.save_dir)
