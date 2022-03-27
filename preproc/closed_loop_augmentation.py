import argparse
import itertools
import json
import os
from enum import Enum
from pathlib import Path
from typing import Sequence, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from pandas import DataFrame
from simple_pid import PID
from tqdm import tqdm

from preproc.aug_utils import generate_aug_params, compute_crop_offsets, save_processsed_seq, zoom_at
from preproc.process_data import process_csv
from utils.data_utils import CSV_COLUMNS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# PID gains for privileged controller

# full image gains
# YAW_P = 0.01
# YAW_I = 0  # no actual feedback, so I should be 0
# YAW_D = 0
#
# THROT_P = 0.01
# THROT_I = 0
# THROT_D = 0
#
# FORWARD_P = 0.4
# FORWARD_I = 0
# FORWARD_D = 0

# processed (cropped) image gains
YAW_P = 0.01
YAW_I = 0  # no actual feedback, so I should be 0
YAW_D = 0

THROT_ROLL_P = 0.01
THROT_ROLL_I = 0
THROT_ROLL_D = 0

FORWARD_P = 0.3  # these shouldn't change
FORWARD_I = 0
FORWARD_D = 0


class TurnChannel(Enum):
    YAW = "yaw"
    ROLL = "roll"


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


def generate_synthetic_sequence(
        image: Image.Image,
        target_location: Sequence[int],
        crop_size: Sequence[int],
        seq_len: int,
        start_offset: Sequence[int],
        lateral_motion: bool = True,
        max_zoom: Optional[float] = None,
        static_fraction: float = 0,
        turn_channel: TurnChannel = TurnChannel.YAW
) -> Tuple[Sequence[Image.Image], DataFrame]:
    """
    Given augmentation params, generates 1 data sequence of target moving from center + start_offset to the center of
    the frame and zooming in from 1 to max zoom

    :param image: base image to augmnet
    :param target_location: x, y coords of target in image
    :param crop_size: size of sliding crop window
    :param seq_len: number of frames in generated sequence
    :param start_offset: pixel offset of target in 1st image in sequence.
    :param lateral_motion: Whether to do lateral motion
    :param max_zoom: How much to zoom in at
    :param static_fraction: fraction of seq_len additional frames that will be added to end of sequence with 0 control
    :return:
    """
    yaw_pid = PID(Kp=YAW_P, Ki=YAW_I, Kd=YAW_D)
    throt_pid = PID(Kp=THROT_ROLL_P, Ki=THROT_ROLL_I, Kd=THROT_ROLL_D)
    roll_pid = PID(Kp=THROT_ROLL_P, Ki=THROT_ROLL_I, Kd=THROT_ROLL_D)
    pitch_pid = PID(Kp=FORWARD_P, Ki=FORWARD_I, Kd=FORWARD_D)

    out_seq = []
    # in order, controls are forward, left, up, and yaw counterclockwise (rad/s)
    control_outputs = pd.DataFrame(columns=CSV_COLUMNS)
    num_static_frames = int(seq_len * static_fraction)
    for cur_index in range(seq_len + num_static_frames):
        yaw_command = 0
        pitch_command = 0
        throt_command = 0
        roll_command = 0

        # augment image
        aug_img = image
        # goes down linearly from 1 to 0 as cur_index increases, stays at 0 during static frames (can't go negative)
        seq_frac = max(0.0, (seq_len - cur_index) / seq_len)
        if lateral_motion:
            # compute crop linearly based on offset
            x_offset = int(start_offset[0] * seq_frac)
            y_offset = int(start_offset[1] * seq_frac)
            # Coords move target in direction of PIL axes (ex +x moves target right, +y moves target down). Note this
            # moves the crop borders in the opposite direction
            crop_tuple = compute_crop_offsets(frame_center=target_location, crop_size=crop_size,
                                              offset=(x_offset, y_offset))
            aug_img = aug_img.crop(crop_tuple)
            if turn_channel == TurnChannel.YAW:
                yaw_command = yaw_pid(x_offset)
            elif turn_channel == TurnChannel.ROLL:
                roll_command = roll_pid(x_offset)
            else:
                raise ValueError(f"Unsupported turn channel {turn_channel}")
            throt_command = throt_pid(y_offset)

        # zoom image
        if max_zoom is not None:
            zoom_level = 1 + (max_zoom - 1) * (1 - seq_frac)
            aug_x_center = aug_img.width // 2
            aug_y_center = aug_img.height // 2
            aug_img = zoom_at(aug_img, aug_x_center, aug_y_center, zoom_level)
            pitch_command = pitch_pid(zoom_level - max_zoom)

        out_seq.append(aug_img)
        # calculate control signal
        control_outputs.loc[cur_index, "vx"] = pitch_command
        control_outputs.loc[cur_index, "vy"] = roll_command
        control_outputs.loc[cur_index, "vz"] = throt_command
        control_outputs.loc[cur_index, "omega_z"] = yaw_command  # always 0 for now

    return out_seq, control_outputs


def augment_image_synthetic(image_path: str, target_location: Sequence[int], out_path: str,
                            min_x_offset: int, max_x_offset: int, min_y_offset: int, max_y_offset: int,
                            min_seq_len: int, max_seq_len: int, min_static_fraction: float, max_static_fraction: float,
                            frame_size_padding: Optional[int] = None, max_zoom: Optional[float] = None,
                            turn_channel: TurnChannel = TurnChannel.YAW) -> bool:
    """
    Calculates random offsets and good frame size for augmentations. For param meanings, see generate_sequence
    """
    assert min_x_offset < max_x_offset and min_y_offset < max_y_offset, "min should be less than max"

    img = Image.open(image_path)

    def valid_fn(target_location: int, crop_size: int, candidate_coord: int, img_dim: int) -> bool:
        end_lower = target_location - crop_size // 2
        end_upper = target_location + crop_size // 2
        last_frame_ok = end_lower >= 0 and end_upper < img_dim
        start_lower = end_lower - candidate_coord  # minus because offset should shift target, not borders, in +x, +y d
        start_upper = end_upper - candidate_coord
        first_frame_ok = start_lower >= 0 and start_upper < img_dim
        return last_frame_ok and first_frame_ok

    # generate params
    params = generate_aug_params(target_location=target_location, min_x_offset=min_x_offset,
                                 max_x_offset=max_x_offset, min_y_offset=min_y_offset,
                                 max_y_offset=max_y_offset,
                                 frame_size_padding=frame_size_padding,
                                 img_width=img.width, img_height=img.height, valid_fn=valid_fn)
    if params is None:
        print(f"Could not find valid augmentations for image {image_path}")
        return False

    crop_size, start_offset = params

    # generate misc params
    seq_len = np.random.randint(min_seq_len, max_seq_len)
    static_fraction = np.random.uniform(min_static_fraction, max_static_fraction)
    # augment and save result
    out_seq, control_inputs = generate_synthetic_sequence(image=img, target_location=target_location,
                                                          crop_size=crop_size,
                                                          seq_len=seq_len, start_offset=start_offset,
                                                          lateral_motion=True,
                                                          max_zoom=max_zoom, static_fraction=static_fraction,
                                                          turn_channel=turn_channel)

    save_processsed_seq(out_path, out_seq, control_inputs, process_seq=True)
    return True


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
    params = generate_aug_params(target_location=target_location, min_x_offset=min_x_offset,
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


def augment_image_list(aug_fn: Callable, img_data_path: str, out_path: str, num_aug: int, *args,
                       parallel_execution: bool = True, **kwargs):
    """
    Augments all images and target locations found in a json at img_data_path. For param meanings, see generate_sequence
    """
    with open(os.path.join(SCRIPT_DIR, img_data_path), "r") as f:
        img_data = json.load(f)

    def perform_single_aug(i, img_path, target_loc):
        extra_args = {}
        if aug_fn == augment_image_mixed:
            # get csv file for run, might either be unprocessed csv 1 dir up or data_out.csv in image_dir
            try:
                run_num = os.path.basename(os.path.dirname(img_path))
                csv_filename = os.path.join(os.path.dirname(img_path), "..", '%.2f.csv' % float(run_num))
                control_df = pd.read_csv(csv_filename)
            except (FileNotFoundError, ValueError) as e:
                csv_filename = os.path.join(img_path, "data_out.csv")
                control_df = pd.read_csv(csv_filename)

            extra_args["control_df"] = control_df
            extra_args["run_path"] = img_path
        else:
            extra_args["image_path"] = img_path

        res = []
        for j in range(num_aug):
            seq_out = os.path.join(out_path, f"{i}_{j}")
            if os.path.exists(seq_out):
                print(f"Found out path {seq_out}, skipping")
                res.append(False)
                continue
            res.append(aug_fn(target_location=target_loc, out_path=seq_out,
                              *args, **extra_args, **kwargs))
        return res

    # perform augmentation in parallel
    if parallel_execution:
        res = Parallel(n_jobs=6)(
            delayed(perform_single_aug)(i, img_path, target_loc) for i, (img_path, target_loc) in
            tqdm(enumerate(img_data)))
    else:
        res = []
        for i, (img_path, target_loc) in tqdm(enumerate(img_data)):
            res.append(perform_single_aug(i, img_path, target_loc))

    # flatten output list
    res = list(itertools.chain(*res))
    print(f"Fraction succeeded {sum(res) / len(res)}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("aug_fn", type=str)
    parser.add_argument("img_data_path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--num_aug", type=int, default=10)
    args = parser.parse_args()

    if args.aug_fn == "augment_image_synthetic":
        # full images
        # augment_image_list(aug_fn=augment_image_synthetic, img_data_path=args.img_data_path, out_path=args.out_path,
        #                    seq_len=args.seq_len, num_aug=args.num_aug, min_x_offset=90, max_x_offset=180,
        #                    min_y_offset=60,
        #                    max_y_offset=120, )
        augment_image_list(aug_fn=augment_image_synthetic, img_data_path=args.img_data_path, out_path=args.out_path,
                           num_aug=args.num_aug, min_x_offset=20, max_x_offset=70,
                           min_y_offset=10, max_y_offset=40, parallel_execution=True, max_zoom=2,
                           frame_size_padding=50, min_static_fraction=0.2, max_static_fraction=0.35, min_seq_len=120,
                           max_seq_len=250, turn_channel=TurnChannel.YAW)
    elif args.aug_fn == "augment_image_mixed":
        augment_image_list(aug_fn=augment_image_mixed, img_data_path=args.img_data_path, out_path=args.out_path,
                           num_aug=args.num_aug, min_x_offset=30, max_x_offset=60, min_y_offset=20,
                           max_y_offset=40, )
    else:
        raise ValueError(f"Illegal aug fn {args.aug_fn}")
