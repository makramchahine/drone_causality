# Created by Patrick Kao at 4/15/22
import math
import os
from typing import Sequence, Optional, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from analysis.vis_utils import parse_params_json
from keras_models import IMAGE_SHAPE
from utils.data_utils import image_dir_generator
from utils.model_utils import get_readable_name, generate_hidden_list

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


def display_trajectory(input_sequences: Sequence, labels: Optional[Sequence[str]] = None,
                       save_path: Optional[str] = None, display_result: bool = False):
    """
    Integrates velocities into trajectories and plots them onto one graph

    :param input_sequences: List of velocities for each agent, where each agent has a list of shape num_timestamps x
    4, and coordinates are forward, left, up and yaw_conterclockwise (assume radians per second)
    :param labels: names of each sequence for legend
    :return:
    """
    if labels is None:
        labels = [None] * len(input_sequences)
    assert len(input_sequences) == len(labels), "need same number of labels as sequences to plot"
    # TODO: plot height as color
    positions = [integrate_positions(seq) for seq in input_sequences]
    for label, pos in zip(labels, positions):
        plt.plot(pos[:, 0], pos[:, 1], label=label)

    if save_path is not None:
        plt.savefig(save_path)

    if display_result:
        plt.show()

def perturb_one(model, data_path):
    vis_hiddens = generate_hidden_list(model, False)
    for i, img in tqdm(enumerate(image_dir_generator(data_path, IMAGE_SHAPE))):
        pass

def perturb_trajectory(datasets: Dict[str, Tuple[str, bool]], output_prefix: str = ".",
                       params_path: Optional[str] = None, ):
    for local_path, model_path, model_params in parse_params_json(params_path):
        for dataset_name, (data_path, reverse_channels, csv_path) in datasets.items():
            data_model_id = f"{get_readable_name(model_params)}_{dataset_name}"
            output_name = os.path.join(output_prefix, data_model_id)

