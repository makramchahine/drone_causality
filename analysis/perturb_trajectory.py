# Created by Patrick Kao at 4/15/22
import argparse
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Tuple, Iterable, Callable, Sequence

import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Functional
from tqdm import tqdm

from analysis.visual_backprop import compute_visualbackprop, get_conv_head_model
from keras_models import IMAGE_SHAPE
from utils.data_utils import image_dir_generator
from utils.graph_utils import graph_deltas, MARKERS
from utils.model_utils import get_readable_name, generate_hidden_list, load_model_from_weights
from utils.traj_utils import *
from utils.vis_utils import parse_params_json, run_visualization

INFERENCE_FREQ = 2


def integrate_positions(input_sequence: Sequence[Sequence[float]]):
    """
    Perform forward euler numerical integration to get list of positions given velocities
    :param input_sequence: See generate_trajectory
    :return:
    """
    pos = np.array([0, 0, 0])
    heading = 0

    to_ret = [pos]
    dt = 1 / INFERENCE_FREQ
    for forward, left, up, ccw, in input_sequence:
        final_heading = heading + ccw * dt
        # for integration, move with average of start and final heading (not real forward euler)
        calc_heading = (heading + final_heading) / 2
        delta_x = forward * math.cos(calc_heading)
        delta_y = left * math.sin(calc_heading)
        pos = pos + np.array([delta_x, delta_y, up]) * dt
        to_ret.append(pos)

    return np.array(to_ret)


TRAJ_LINE_STYLES = [
    "solid",
    "dashed"
]


def display_trajectory(positions: Sequence[ndarray], labels: Optional[Sequence[str]] = None,
                       save_path: Optional[str] = None, display_result: bool = False):
    """
    Integrates velocities into trajectories and plots them onto one graph

    :param positions: List of positions for each agent, where each agent has an ndarray of shape
    num_timestamps x 4, and coordinates are forward, left, up and yaw_conterclockwise
    :param labels: names of each sequence for legend
    :return:
    """
    plt.clf()
    if labels is None:
        labels = [None] * len(positions)
    assert len(positions) == len(labels), "need same number of labels as sequences to plot"
    alpha_points = np.linspace(1, 0.5, positions[0].shape[0])
    size_points = np.linspace(200, 50, positions[0].shape[0])
    for i, (label, pos) in enumerate(zip(labels, positions)):
        sns.lineplot(x=pos[:, 0], y=pos[:, 1], label=label, linestyle=TRAJ_LINE_STYLES[i])
        # also plot markers at each point using scatterplot, make markers diff color/size to rep time
        sns.scatterplot(x=pos[:, 0], y=pos[:, 1], marker=MARKERS[i], alpha=alpha_points, s=size_points)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")

    # remove top and right borders:
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # ticks inward:
    plt.tick_params(axis='both', which='both', direction='in', )

    # tick lengths y-axis:
    plt.tick_params(axis='both', which='both', length=0)
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)

    # make axes square
    plt.axis('equal')

    # grid lines
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.4)
    plt.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.4)

    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')

    if display_result:
        plt.show()


def get_cmd_sequence(model, data_source: Iterable, perturb_frac: float = 1):
    hiddens = generate_hidden_list(model, False)
    vel_cmds = []
    img_data = [i for i in data_source]
    keep_len = round(len(img_data) * perturb_frac)
    trimmed_data = img_data[:keep_len]
    for i, img in tqdm(enumerate(trimmed_data)):
        res = model.predict([img, *hiddens])
        vel_cmd = res[0]
        hiddens = res[1:]
        vel_cmds.append(np.squeeze(vel_cmd))
    # out shape: time x ndarray (4,)
    return vel_cmds


def perturb_one(model: Functional, data_path: str, perturb_fxn: Callable, img_save_path: str, delta: float,
                perturb_frac: float = 1, display_result: bool = False, reverse_channels: bool = False,
                video_save_path: Optional[str] = None):
    def perturb_data():
        for img in image_dir_generator(data_path, IMAGE_SHAPE, reverse_channels):
            yield perturb_fxn(img, delta)

    og_data_source = image_dir_generator(data_path, IMAGE_SHAPE, reverse_channels=reverse_channels)
    og_seq = get_cmd_sequence(model, data_source=og_data_source, perturb_frac=perturb_frac)
    perturb_seq = get_cmd_sequence(model, data_source=perturb_data(), perturb_frac=perturb_frac)
    # og_seq and perturb_seq are lists num_timesteps x 4 long

    positions = [integrate_positions(seq) for seq in [og_seq, perturb_seq]]

    display_trajectory(positions=positions, labels=["Original Sequence", "Perturbed Sequence"],
                       save_path=img_save_path, display_result=display_result)
    if video_save_path:
        vis_model = get_conv_head_model(model)
        video_dir = os.path.dirname(video_save_path)
        video_base = os.path.splitext(os.path.basename(video_save_path))
        og_vid_path = os.path.join(video_dir, f"{video_base[0]}_og{video_base[1]}")
        perturbed_vid_path = os.path.join(video_dir, f"{video_base[0]}_perturbed{video_base[1]}")

        run_visualization(
            vis_model=vis_model,
            data=data_path,
            vis_func=compute_visualbackprop,
            image_output_path=None,
            video_output_path=og_vid_path,
            reverse_channels=reverse_channels,
            control_source=model,
            vis_kwargs={}
        )

        run_visualization(
            vis_model=vis_model,
            data=perturb_data(),
            vis_func=compute_visualbackprop,
            image_output_path=None,
            video_output_path=perturbed_vid_path,
            reverse_channels=reverse_channels,
            control_source=model,
            vis_kwargs={}
        )

    return positions


def perturb_trajectory(datasets_json: str, params_path: str, perturb_fxn: Callable, delta: float,
                       perturb_frac: float = 1, skip_models=Sequence[str], output_prefix: str = ".",
                       display_result: bool = False, save_video: bool = True, distance_fxn: Optional[Callable] = None):
    with open(datasets_json, "r") as f:
        datasets: Dict[str, Tuple[str, bool]] = json.load(f)

    mean_distances = {}
    std_distances = {}
    for local_path, model_path, model_params in parse_params_json(params_path):
        dataset_distances = []
        model_name = get_readable_name(model_params)
        if model_name in skip_models:
            continue
        for dataset_name, (data_path, reverse_channels, csv_path) in datasets.items():
            data_model_id = f"{model_name}_{dataset_name}"
            output_name = os.path.join(output_prefix, data_model_id)
            Path(output_name).mkdir(exist_ok=True, parents=True)
            model = load_model_from_weights(model_params, model_path)
            img_save_path = os.path.join(output_name, f"{data_model_id}_{delta}.png")
            vid_save_path = os.path.join(output_name, f"{data_model_id}_{delta}.mp4")
            positions = perturb_one(model, data_path=data_path, perturb_fxn=perturb_fxn,
                                    img_save_path=img_save_path, delta=delta, perturb_frac=perturb_frac,
                                    display_result=display_result, reverse_channels=reverse_channels,
                                    video_save_path=vid_save_path if save_video else None)
            if distance_fxn is not None:
                traj_distance = distance_fxn(*positions)
                dataset_distances.append(traj_distance)

        if distance_fxn is not None:
            mean_distances[model_name] = float(np.mean(dataset_distances))
            std_distances[model_name] = float(np.std(dataset_distances))

    return mean_distances, std_distances


def compare_distances(datasets_json: str, params_path: str, perturb_fxn: Callable, deltas: Sequence[float],
                      perturb_frac: float = 1, output_prefix: str = ".", display_result: bool = False,
                      save_video: bool = True, distance_fxn: Optional[Callable] = None,
                      skip_models: Optional[Sequence[str]] = None, force_even_x: bool = False):
    """
    Gets the average trajectory distance for each of the dataset jsons
    """
    assert len(deltas), "No deltas passed"
    if skip_models is None:
        skip_models = []

    print(f"Loading datasets from {datasets_json}")
    print(f"Saving to {output_prefix}")

    distances = OrderedDict()  # map from delta to dict of model to distances for each model
    stds = OrderedDict()
    sorted_deltas = sorted(deltas)
    for delta in sorted_deltas:
        mean_distance, std_distance = perturb_trajectory(datasets_json=datasets_json, params_path=params_path,
                                                         perturb_fxn=perturb_fxn, delta=delta,
                                                         perturb_frac=perturb_frac, skip_models=skip_models,
                                                         output_prefix=output_prefix, display_result=display_result,
                                                         save_video=save_video, distance_fxn=distance_fxn)
        distances[delta] = mean_distance
        stds[delta] = std_distance

    # unpack all distances for each model
    model_distances = switch_top_keys(distances)
    model_stds = switch_top_keys(stds)

    graph_deltas(model_distances, model_stds, sorted_deltas, save_dir=output_prefix, display_result=display_result,
                 x_label=f"{perturb_fxn.__name__}", y_label="Distance (m)", force_even_x=force_even_x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_json")
    parser.add_argument("params_path")
    parser.add_argument("perturb_fxn", default="brightness_perturbation")
    parser.add_argument("--output_prefix", default="./perturb_out")
    parser.add_argument("--display_result", action="store_true")
    parser.add_argument("--deltas", nargs="+", type=float, default=[])
    parser.add_argument("--skip_models", nargs="+", type=str, default=[])
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--distance_fxn", default=None)
    parser.add_argument("--perturb_frac", default=1.0, type=float)
    parser.add_argument("--force_even_x", action="store_true")
    args = parser.parse_args()
    perturb_fxn_callable = locals()[args.perturb_fxn.lower()]
    distance_fxn_callable = locals()[args.distance_fxn.lower()] if args.distance_fxn is not None else None
    compare_distances(datasets_json=args.datasets_json, params_path=args.params_path, perturb_fxn=perturb_fxn_callable,
                      output_prefix=args.output_prefix, display_result=args.display_result, save_video=args.save_video,
                      deltas=args.deltas, perturb_frac=args.perturb_frac, distance_fxn=distance_fxn_callable,
                      skip_models=args.skip_models, force_even_x=args.force_even_x)
