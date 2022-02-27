import argparse
import json
import os
from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from pandas import DataFrame
from simple_pid import PID

from data_preproc.process_data import process_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# num times to retry generating offsets
NUM_RNG_ATTEMPTS = 100

# PID gains for privileged controller
YAW_P = 0.01
YAW_I = 0  # no actual feedback, so I should be 0
YAW_D = 0

THROT_P = 0.01
THROT_I = 0
THROT_D = 0

FORWARD_P = 0.4
FORWARD_I = 0
FORWARD_D = 0


def random_sign():
    return np.random.choice([-1, 1])


# from https://stackoverflow.com/questions/46149003/pil-zoom-into-image-at-a-particular-point
def zoom_at(img: Image.Image, x: int, y: int, zoom: float) -> Image.Image:
    """
    PIL helper that combines crop and resize to zooom into an image at a point
    :param img: image to transform
    :param x: x coord to zoom into
    :param y: y coord to zoom into
    :param zoom: zoom factor
    :return: zoomed in image
    """
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((int(x - w / zoom2), int(y - h / zoom2),
                    int(x + w / zoom2), int(y + h / zoom2)))
    return img.resize((w, h), Image.LANCZOS)


def generate_sequence(
        image: Image.Image,
        target_location: Sequence[int],
        crop_size: Sequence[int],
        seq_len: int,
        start_offset: Sequence[int],
        lateral_motion: bool = True,
        max_zoom: Optional[float] = None,
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
    :return:
    """
    x_target, y_target = target_location
    x_size, y_size = crop_size

    yaw_pid = PID(Kp=YAW_P, Ki=YAW_I, Kd=YAW_D)
    throt_pid = PID(Kp=THROT_P, Ki=THROT_I, Kd=THROT_D)
    pitch_pid = PID(Kp=FORWARD_P, Ki=FORWARD_I, Kd=FORWARD_D)

    out_seq = []
    # in order, controls are forward, right, up, and yaw clockwise (rad/s)
    control_outputs = pd.DataFrame(columns=["vx", "vy", "vz", "omega_z"])
    for cur_index in range(seq_len):
        yaw_command = 0
        pitch_command = 0
        throt_command = 0
        roll_command = 0

        # augment image
        aug_img = image
        seq_frac = (seq_len - cur_index) / seq_len  # goes down linearly from 1 to 0 as cur_index increases
        if lateral_motion:
            # compute crop linearly based on offset
            x_offset = start_offset[0] * seq_frac
            y_offset = start_offset[1] * seq_frac
            frame_x_center = x_target - x_offset
            frame_y_center = y_target - y_offset
            left_crop = frame_x_center - x_size // 2
            right_crop = frame_x_center + x_size // 2
            top_crop = frame_y_center - y_size // 2
            bot_crop = frame_y_center + y_size // 2
            aug_img = aug_img.crop((left_crop, top_crop, right_crop, bot_crop))
            yaw_command = yaw_pid(-x_offset)
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
        control_outputs.loc[cur_index, "vy"] = roll_command  # always 0 for now
        control_outputs.loc[cur_index, "vz"] = throt_command
        control_outputs.loc[cur_index, "omega_z"] = yaw_command

    return out_seq, control_outputs


def augment_image(image_path: str, target_location: Sequence[int], out_path: str, seq_len: int,
                  min_x_offset: int, max_x_offset: int, min_y_offset: int, max_y_offset: int,
                  frame_size_padding: Optional[int] = None, max_zoom: Optional[float] = None):
    """
    Calculates random offsets and good frame size for augmentations. For param meanings, see generate_sequence
    """
    assert min_x_offset < max_x_offset and min_y_offset < max_y_offset, "min should be less than max"
    # automatically calculate average frame size padding if not provided
    if frame_size_padding is None:
        frame_size_padding = np.mean([min_x_offset, max_x_offset, min_y_offset, max_y_offset])

    img = Image.open(image_path)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # generate params
    aspect_ratio = img.height / img.width
    crop_width = img.width - 2.2 * frame_size_padding  # if max is selected, 2 would fill if obj was in middle. ALlow some leeway
    crop_size = (crop_width, crop_width * aspect_ratio)
    # generate sequence offsets
    x_offset = None
    y_offset = None
    for _ in range(NUM_RNG_ATTEMPTS):
        candidate_x = np.random.uniform(min_x_offset, max_x_offset) * random_sign()
        candidate_y = np.random.uniform(min_y_offset, max_y_offset) * random_sign()
        # check validity
        x_valid = target_location[0] - crop_size[0] // 2 + candidate_x > 0 and target_location[0] + crop_size[
            0] // 2 + candidate_x < img.width
        y_valid = target_location[1] - crop_size[1] // 2 + candidate_y > 0 and target_location[1] + crop_size[
            1] // 2 + candidate_y < img.height
        if x_valid and y_valid:
            x_offset = candidate_x
            y_offset = candidate_y
            break

    if x_offset is None:
        print(f"Could not find valid augmentations for image {image_path}")
        return
    # augment and save result
    out_seq, control_inputs = generate_sequence(image=img, target_location=target_location, crop_size=crop_size,
                                                seq_len=seq_len, start_offset=(x_offset, y_offset), lateral_motion=True,
                                                max_zoom=max_zoom)
    for i, aug_img in enumerate(out_seq):
        processed = process_image(aug_img)
        processed.save(os.path.join(out_path, f"{str(i).zfill(6)}.png"))

    control_inputs.to_csv(os.path.join(out_path, "data_out.csv"), index=False)


def augment_image_list(img_data_path: str, out_path: str, num_aug: int = 10, seq_len: int = 250, min_x_offset: int = 90,
                       max_x_offset: int = 180, min_y_offset: int = 60, max_y_offset: int = 120,
                       max_zoom: Optional[float] = None):
    """
    Augments all images and target locations found in a json at img_data_path. For param meanings, see generate_sequence
    """
    with open(os.path.join(SCRIPT_DIR, img_data_path), "r") as f:
        img_data = json.load(f)

    for i, (img_path, target_loc) in enumerate(img_data):
        for j in range(num_aug):
            seq_out = os.path.join(out_path, f"{i}_{j}")
            augment_image(image_path=img_path, target_location=target_loc, out_path=seq_out, seq_len=seq_len,
                          min_x_offset=min_x_offset, max_x_offset=max_x_offset, min_y_offset=min_y_offset,
                          max_y_offset=max_y_offset, max_zoom=max_zoom)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_data_path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--seq_len", type=int, default=250)
    parser.add_argument("--num_aug", type=int, default=10)
    args = parser.parse_args()
    augment_image_list(img_data_path=args.img_data_path, out_path=args.out_path, seq_len=args.seq_len,
                       num_aug=args.num_aug, )
