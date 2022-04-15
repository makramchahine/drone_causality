# Created by Patrick Kao at 4/11/22
from enum import Enum
from typing import Sequence, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame
from simple_pid import PID

# processed (cropped) image gains
from preprocess.aug_utils import zoom_at, generate_crop_location, THROT_ROLL_P, YAW_P, YAW_I, \
    YAW_D, \
    THROT_ROLL_I, THROT_ROLL_D, FORWARD_D, FORWARD_I, FORWARD_P, compute_crop_offsets
from utils.data_utils import CSV_COLUMNS


# PID gains for privileged controller


class TurnChannel(Enum):
    YAW = "yaw"
    ROLL = "roll"


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


def is_positive(num):
    return 1 if num > 0 else 0


def get_synthetic_params(image_path: str, target_location: Sequence[int],
                         min_x_offset: int, max_x_offset: int, min_y_offset: int, max_y_offset: int,
                         min_seq_len: int, max_seq_len: int, min_static_fraction: float, max_static_fraction: float,
                         frame_size_padding: Optional[int] = None, **kwargs) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    Calculates random offsets and good frame size for augmentations. For param meanings, see generate_sequence
    """
    assert min_x_offset < max_x_offset and min_y_offset < max_y_offset, "min should be less than max"

    img = Image.open(image_path)

    to_ret = {}

    def valid_fn(target_location: int, crop_size: int, candidate_coord: int, img_dim: int) -> bool:
        end_lower = target_location - crop_size // 2
        end_upper = target_location + crop_size // 2
        last_frame_ok = end_lower >= 0 and end_upper < img_dim
        start_lower = end_lower - candidate_coord  # minus because offset should shift target, not borders, in +x, +y d
        start_upper = end_upper - candidate_coord
        first_frame_ok = start_lower >= 0 and start_upper < img_dim
        return last_frame_ok and first_frame_ok

    # generate params
    params = generate_crop_location(target_location=target_location, min_x_offset=min_x_offset,
                                    max_x_offset=max_x_offset, min_y_offset=min_y_offset,
                                    max_y_offset=max_y_offset,
                                    frame_size_padding=frame_size_padding,
                                    img_width=img.width, img_height=img.height, valid_fn=valid_fn)
    if params is None:
        return None

    crop_size, start_offset = params
    to_ret["crop_size"] = crop_size
    to_ret["start_offset"] = start_offset

    # generate misc params
    to_ret["seq_len"] = np.random.randint(min_seq_len, max_seq_len)
    to_ret["static_fraction"] = np.random.uniform(min_static_fraction, max_static_fraction)
    # sve misc params
    to_ret["image"] = img
    to_ret["target_location"] = target_location
    to_ret.update(kwargs)

    aug_class = is_positive(start_offset[0])*2+is_positive(start_offset[1])

    return to_ret, aug_class
