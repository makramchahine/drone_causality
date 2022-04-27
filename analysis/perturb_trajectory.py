# Created by Patrick Kao at 4/15/22
import argparse
import json
import math
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional, Dict, Tuple, Iterable, Callable

from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Functional
from tqdm import tqdm

from analysis.vis_utils import parse_params_json, run_visualization
from analysis.visual_backprop import compute_visualbackprop, get_conv_head_model
from keras_models import IMAGE_SHAPE
from utils.data_utils import image_dir_generator
from utils.model_utils import get_readable_name, generate_hidden_list, load_model_from_weights
from utils.traj_utils import *

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
    # TODO: plot height as color
    for label, pos in zip(labels, positions):
        plt.plot(pos[:, 0], pos[:, 1], label=label)

    plt.legend(loc="upper left")
    if save_path is not None:
        plt.savefig(save_path)

    if display_result:
        plt.show()


def get_cmd_sequence(model, data_source: Iterable):
    hiddens = generate_hidden_list(model, False)
    vel_cmds = []
    for i, img in tqdm(enumerate(data_source)):
        res = model.predict([img, *hiddens])
        vel_cmd = res[0]
        hiddens = res[1:]
        vel_cmds.append(np.squeeze(vel_cmd))
    # out shape: time x ndarray (4,)
    return vel_cmds


def perturb_one(model: Functional, data_path: str, perturb_fxn: Callable, img_save_path: str, delta: float,
                display_result: bool = False, reverse_channels: bool = False, video_save_path: Optional[str] = None):
    def perturb_data():
        for img in image_dir_generator(data_path, IMAGE_SHAPE, reverse_channels):
            yield perturb_fxn(img, delta)

    og_seq = get_cmd_sequence(model, data_source=image_dir_generator(data_path, IMAGE_SHAPE,
                                                                     reverse_channels=reverse_channels))
    perturb_seq = get_cmd_sequence(model, data_source=perturb_data())
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
                       output_prefix: str = ".", display_result: bool = False, save_video: bool = True,
                       distance_fxn: Optional[Callable] = None):
    with open(datasets_json, "r") as f:
        datasets: Dict[str, Tuple[str, bool]] = json.load(f)

    mean_distances = {}
    std_distances = {}
    for local_path, model_path, model_params in parse_params_json(params_path):
        dataset_distances = []
        model_name = get_readable_name(model_params)
        for dataset_name, (data_path, reverse_channels, csv_path) in datasets.items():
            data_model_id = f"{model_name}_{dataset_name}"
            output_name = os.path.join(output_prefix, data_model_id)
            Path(output_name).mkdir(exist_ok=True, parents=True)
            model = load_model_from_weights(model_params, model_path)
            img_save_path = os.path.join(output_name, f"{data_model_id}.png")
            vid_save_path = os.path.join(output_name, f"{data_model_id}.mp4")
            positions = perturb_one(model, data_path=data_path, perturb_fxn=perturb_fxn,
                                    img_save_path=img_save_path, delta=delta,
                                    display_result=display_result, reverse_channels=reverse_channels,
                                    video_save_path=vid_save_path if save_video else None)
            if distance_fxn is not None:
                traj_distance = distance_fxn(*positions)
                dataset_distances.append(traj_distance)

        if distance_fxn is not None:
            mean_distances[model_name] = np.mean(dataset_distances)
            std_distances[model_name] = np.std(dataset_distances)

    return mean_distances, std_distances


def compare_distances(datasets_json: str, params_path: str, perturb_fxn: Callable, deltas: Sequence[float],
                      output_prefix: str = ".", display_result: bool = False, save_video: bool = True,
                      distance_fxn: Optional[Callable] = None):
    """
    Gets the average trajectory distance for each of the dataset jsons
    """
    assert len(deltas), "No deltas passed"

    distances = OrderedDict()  # map from delta to dict of model to distances for each model
    stds = OrderedDict()
    sorted_deltas = sorted(deltas)
    for delta in sorted_deltas:
        mean_distance, std_distance = perturb_trajectory(datasets_json, params_path, perturb_fxn, delta, output_prefix,
                                                         display_result,
                                                         save_video,
                                                         distance_fxn)
        distances[delta] = mean_distance
        stds[delta] = std_distance

    # unpack all distances for each model
    model_distances = switch_top_keys(distances)
    model_stds = switch_top_keys(stds)

    plt.clf()
    for model, model_distance in model_distances.items():
        model_distance = np.asarray(model_distance)
        plt.plot(sorted_deltas, model_distance, label=model)
        std = np.asarray(model_stds[model])
        plt.fill_between(sorted_deltas, model_distance - std, model_distance + std, alpha=0.5)

    plt.legend(loc="upper left")
    plt.savefig(os.path.join(output_prefix, "distances.png"))
    with open(os.path.join(output_prefix, "distances.json"), "w") as f:
        distances_stds = {"means": model_distances, "stds": model_stds, "deltas": sorted_deltas}
        json.dump(distances_stds, f)

    if display_result:
        plt.show()


def switch_top_keys(distances):
    to_ret = defaultdict(list)
    for delta, delta_distance in distances.items():
        for model_name, distance in delta_distance.items():
            to_ret[model_name].append(distance)

    return to_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_json")
    parser.add_argument("params_path")
    parser.add_argument("perturb_fxn", default="brightness_perturbation")
    parser.add_argument("--output_prefix", default="./perturb")
    parser.add_argument("--display_result", action="store_true")
    parser.add_argument("--deltas", nargs="+", type=float, default=[])
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--distance_fxn", default=None)
    args = parser.parse_args()
    perturb_fxn_callable = locals()[args.perturb_fxn.lower()]
    distance_fxn_callable = locals()[args.distance_fxn.lower()] if args.distance_fxn is not None else None
    compare_distances(datasets_json=args.datasets_json, params_path=args.params_path, perturb_fxn=perturb_fxn_callable,
                      output_prefix=args.output_prefix, display_result=args.display_result, save_video=args.save_video,
                      deltas=args.deltas, distance_fxn=distance_fxn_callable)
