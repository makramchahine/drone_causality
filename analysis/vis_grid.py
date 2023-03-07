# Created by Patrick Kao at 4/27/22
import argparse
import copy
import json
import os.path
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple, Sequence

import numpy as np
from matplotlib import pyplot as plt

from analysis.visual_backprop import compute_visualbackprop, get_conv_head
from keras_models import IMAGE_SHAPE
from utils.data_utils import image_dir_generator
from utils.graph_utils import LABEL_MAP, GRAPH_ORDER
from utils.model_utils import get_readable_name
from utils.vis_utils import parse_params_json, run_visualization

DPI = 100


def grid_one_run(dataset_path: str, params_path: str, vis_func: Callable, reverse_channels: bool, save_path: str,
                 skip_models: Sequence[str]):
    all_vis = OrderedDict()

    og_imgs = []
    for img in image_dir_generator(dataset_path, IMAGE_SHAPE, reverse_channels):
        og_imgs.append(np.squeeze(img, axis=0))

    all_vis["Original Images"] = og_imgs
    for local_path, model_path, model_params in parse_params_json(params_path):
        if get_readable_name(model_params) in skip_models:
            continue
        model = get_conv_head(model_path, model_params)
        saliency_imgs = run_visualization(vis_model=model, data=dataset_path, vis_func=vis_func, image_output_path=None,
                                          video_output_path=None, reverse_channels=reverse_channels,
                                          control_source=None,
                                          absolute_norm=True, )

        all_vis[get_readable_name(model_params)] = saliency_imgs

    plt.clf()

    fig, axes = plt.subplots(nrows=len(og_imgs), ncols=len(all_vis.keys()),
                             figsize=(
                                 len(all_vis.keys()) * IMAGE_SHAPE[1] // DPI, len(og_imgs) * IMAGE_SHAPE[0] // DPI),
                             dpi=DPI)

    mod_graph = []
    for graph in GRAPH_ORDER:
        if graph not in skip_models:
            mod_graph.append(graph)
    col_order_map = {name: index+1 for index, name in enumerate(mod_graph)}
    col_order_map["Original Images"] = 0
    for model_name, saliencies in all_vis.items():
        for j, sal_img in enumerate(saliencies):
            axes[j, col_order_map[model_name]].imshow(sal_img)
            axes[j, col_order_map[model_name]].set_xticks([])
            axes[j, col_order_map[model_name]].set_yticks([])

        column_name = LABEL_MAP[model_name] if model_name != "Original Images" else "Original Images"
        axes[len(saliencies) - 1, col_order_map[model_name]].set_xlabel(column_name)

    for i in range(len(all_vis["Original Images"])):
        axes[i, 0].set_ylabel(i + 1, rotation=0)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(save_path, bbox_inches='tight')


def params_grid(datasets_json: str, params_path: str, save_dir: str, skip_models: Sequence[str]):
    with open(datasets_json, "r") as f:
        datasets: Dict[str, Tuple[str, bool]] = json.load(f)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for dataset_name, (data_path, reverse_channels, csv_path) in datasets.items():
        save_path = os.path.join(save_dir, f"{dataset_name}.png")
        grid_one_run(dataset_path=data_path, params_path=params_path, vis_func=compute_visualbackprop,
                     reverse_channels=reverse_channels, save_path=save_path, skip_models=skip_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_json")
    parser.add_argument("params_path")
    parser.add_argument("--save_dir", default="grid_out")
    parser.add_argument("--skip_models", nargs="+", type=str, default=[])
    args = parser.parse_args()
    params_grid(datasets_json=args.datasets_json, params_path=args.params_path, save_dir=args.save_dir,
                skip_models=args.skip_models)
