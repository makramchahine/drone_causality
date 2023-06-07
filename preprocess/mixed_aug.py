# Created by Patrick Kao at 4/11/22
import os
from pathlib import Path
from typing import Sequence, Optional, Tuple

import pandas as pd
from PIL import Image
from pandas import DataFrame
from simple_pid import PID

from aug_utils import YAW_P, YAW_I, YAW_D, THROT_ROLL_I, THROT_ROLL_P, THROT_ROLL_D, compute_crop_offsets, \
    generate_crop_location, save_processsed_seq
from process_data import process_csv
from utils.data_utils import CSV_COLUMNS


def generate_mixed_sequence(
        img_sequence: Sequence[Image.Image],
        control_df: DataFrame,
        crop_size: Sequence[int],
        start_offset: Sequence[int],
) -> Tuple[Sequence[Image.Image], DataFrame]:
    """
    Generates augmented sequence by taking entire input sequence

    :param img_sequence:
    :param control_df:
    :param target_location:
    :param crop_size:
    :param start_offset:
    :return:
    """
    if len(control_df.columns) > 4:
        control_df = process_csv(control_df)
    yaw_pid = PID(Kp=YAW_P, Ki=YAW_I, Kd=YAW_D)
    throt_pid = PID(Kp=THROT_ROLL_P, Ki=THROT_ROLL_I, Kd=THROT_ROLL_D)
    seq_len = len(img_sequence)

    out_seq = []
    # in order, controls are forward, left, up, and yaw counterclockwise (rad/s)
    control_outputs = pd.DataFrame(columns=CSV_COLUMNS)
    for cur_index in range(seq_len):
        pitch_command = 0
        roll_command = 0

        # augment image
        aug_img = img_sequence[cur_index]
        seq_frac = (seq_len - cur_index) / seq_len  # goes down linearly from 1 to 0 as cur_index increases
        # compute crop linearly based on offset
        x_offset = int(start_offset[0] * seq_frac)
        y_offset = int(start_offset[1] * seq_frac)
        crop_tuple = compute_crop_offsets(frame_center=(aug_img.width // 2, aug_img.height // 2), crop_size=crop_size,
                                          offset=(x_offset, y_offset))
        aug_img = aug_img.crop(crop_tuple)
        yaw_command = yaw_pid(x_offset)
        throt_command = throt_pid(y_offset)

        out_seq.append(aug_img)
        # calculate control signal
        control_outputs.loc[cur_index, "vx"] = control_df.loc[cur_index, "vx"] + pitch_command
        control_outputs.loc[cur_index, "vy"] = control_df.loc[cur_index, "vy"] + roll_command  # always 0 for now
        control_outputs.loc[cur_index, "vz"] = control_df.loc[cur_index, "vz"] + throt_command
        control_outputs.loc[cur_index, "omega_z"] = control_df.loc[cur_index, "omega_z"] + yaw_command

    return out_seq, control_outputs


def augment_image_mixed(run_path: str, target_location: Sequence[int], out_path: str, control_df: DataFrame,
                        min_x_offset: int, max_x_offset: int, min_y_offset: int, max_y_offset: int,
                        frame_size_padding: Optional[int] = None, ) -> bool:
    assert min_x_offset < max_x_offset and min_y_offset < max_y_offset, "min should be less than max"
    run_imgs = sorted(os.listdir(run_path))
    run_imgs = [os.path.join(run_path, img) for img in run_imgs if "png" in img]  # filter and add absolute path
    img_sequence = [Image.open(img) for img in run_imgs]

    first_img = img_sequence[0]
    Path(out_path).mkdir(parents=True, exist_ok=True)

    def valid_fn(target_location: int, crop_size: int, candidate_coord: int, img_dim: int) -> bool:
        return img_dim // 2 - crop_size // 2 + candidate_coord < target_location < img_dim // 2 + crop_size // 2 + candidate_coord

    # generate params
    params = generate_crop_location(target_location=target_location, min_x_offset=min_x_offset,
                                    max_x_offset=max_x_offset, min_y_offset=min_y_offset,
                                    max_y_offset=max_y_offset,
                                    frame_size_padding=frame_size_padding,
                                    img_width=first_img.width, img_height=first_img.height,
                                    valid_fn=valid_fn)
    if params is None:
        print(f"Could not find valid augmentations for run {run_path}")
        return False
    crop_size, start_offset = params

    out_seq, control_inputs = generate_mixed_sequence(img_sequence=img_sequence, control_df=control_df,
                                                      crop_size=crop_size, start_offset=start_offset)

    save_processsed_seq(out_path, out_seq, control_inputs, process_seq=False)
    return True
