# Created by Patrick Kao at 4/27/22
import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Dict, Sequence

import numpy as np
from skimage.metrics import structural_similarity

from analysis.visual_backprop import get_conv_head, compute_visualbackprop
from keras_models import IMAGE_SHAPE
from utils.data_utils import load_image
from utils.graph_utils import graph_deltas
from utils.model_utils import get_readable_name
from utils.traj_utils import noise_perturbation, switch_top_keys
from utils.vis_utils import parse_params_json


def one_ssim(datasets: Dict, params_path: str, noise_std: float) -> Tuple[Dict, Dict]:
    all_mean = {}
    all_std = {}

    for local_path, model_path, model_params in parse_params_json(params_path):
        model_name = get_readable_name(model_params)
        model = get_conv_head(model_path, model_params)
        ssims = []
        for dataset_name, (img_path, reverse_channels, csv_path) in datasets.items():
            og_img = load_image(img_path, IMAGE_SHAPE, reverse_channels)
            noise_img = noise_perturbation(og_img, noise_std).numpy()
            og_img = og_img.astype(np.float32)  # make both arrs have float32 dtype

            og_back = compute_visualbackprop(og_img, model).numpy()
            noise_back = compute_visualbackprop(noise_img, model).numpy()
            # normalize images between 0 and 1
            im_min = np.min([og_back, noise_back])
            im_max = np.max([og_back, noise_back])
            og_norm = (og_back - im_min) / (im_max - im_min)
            noise_norm = (noise_back - im_min) / (im_max - im_min)

            ssim = structural_similarity(og_norm, noise_norm, data_range=1, channel_axis=-1)
            ssims.append(ssim)

        ssim_mean = float(np.mean(ssims, axis=0))  # cast to float for serialize
        ssim_std = float(np.std(ssims, axis=0))
        model_ssim_mean = {model_name: ssim_mean}
        model_ssim_std = {model_name: ssim_std}
        all_mean.update(model_ssim_mean)
        all_std.update(model_ssim_std)

    return all_mean, all_std


def params_ssid(datasets_json: str, params_path: str, deltas: Sequence[float], display_result: bool = False,
                save_dir: Optional[str] = None):
    with open(datasets_json, "r") as f:
        datasets: Dict[str, Tuple[str, bool]] = json.load(f)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    means = OrderedDict()  # map from delta to dict of model to distances for each model
    stds = OrderedDict()
    sorted_deltas = sorted(deltas)
    for delta in sorted_deltas:
        mean_distance, std_distance = one_ssim(datasets, params_path, delta)
        means[delta] = mean_distance
        stds[delta] = std_distance

    model_means = switch_top_keys(means)
    model_stds = switch_top_keys(stds)

    graph_deltas(model_means, model_stds, sorted_deltas, save_dir=save_dir, display_result=display_result,
                 x_label="Gaussian Noise (pixels)", y_label="SSIM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_json")
    parser.add_argument("params_path")
    parser.add_argument("--display_result", action="store_true")
    parser.add_argument("--save_dir", type=str, default="ssim_out")
    parser.add_argument("--deltas", nargs="+", type=float, default=[])
    args = parser.parse_args()
    params_ssid(datasets_json=args.datasets_json, params_path=args.params_path, deltas=args.deltas,
                display_result=args.display_result, save_dir=args.save_dir)
