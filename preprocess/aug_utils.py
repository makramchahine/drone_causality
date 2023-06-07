import os
from pathlib import Path
from typing import Sequence, Optional, Tuple, Callable

import numpy as np
from PIL import Image
from pandas import DataFrame

# num times to retry generating offsets
from process_data import process_image

NUM_RNG_ATTEMPTS = 100

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
YAW_P = 0.01
YAW_I = 0  # no actual feedback, so I should be 0
YAW_D = 0

THROT_ROLL_P = 0.01
THROT_ROLL_I = 0
THROT_ROLL_D = 0

FORWARD_P = 0.3 * 1.5  # these shouldn't change
FORWARD_I = 0
FORWARD_D = 0


def random_sign():
    return np.random.choice([-1, 1])


def generate_crop_location(target_location: Sequence[int], min_x_offset: int,
                           max_x_offset: int, min_y_offset: int, max_y_offset: int, frame_size_padding: Optional[float],
                           img_width: int, img_height: int, valid_fn: Callable[[int, int, int, int], bool]):
    # automatically calculate average frame size padding if not provided
    if frame_size_padding is None:
        frame_size_padding = np.mean([min_x_offset, max_x_offset, min_y_offset, max_y_offset])

    aspect_ratio = img_height / img_width
    crop_width = img_width - 2.2 * frame_size_padding  # if max is selected, 2 would fill if obj was in middle. Allow some leeway
    crop_size = (crop_width, crop_width * aspect_ratio)
    # generate sequence offsets.
    x_offset = None
    y_offset = None
    for _ in range(NUM_RNG_ATTEMPTS):
        candidate_x = np.random.uniform(min_x_offset, max_x_offset) * random_sign()
        candidate_y = np.random.uniform(min_y_offset, max_y_offset) * random_sign()
        # check validity
        x_valid = valid_fn(target_location[0], crop_size[0], candidate_x, img_width)
        y_valid = valid_fn(target_location[1], crop_size[1], candidate_y, img_height)
        if x_valid and y_valid:
            x_offset = candidate_x
            y_offset = candidate_y
            break

    if x_offset is None:
        return

    return crop_size, (x_offset, y_offset)


def compute_crop_offsets(frame_center: Sequence[int], crop_size: Sequence[int],
                         offset: Sequence[int]) -> Tuple:
    x_target, y_target = frame_center
    x_offset, y_offset = offset
    x_size, y_size = crop_size
    frame_x_center = x_target - x_offset  # minus because offset should shift target, not borders, in +x, +y d
    frame_y_center = y_target - y_offset
    left_crop = frame_x_center - x_size // 2
    right_crop = frame_x_center + x_size // 2
    top_crop = frame_y_center - y_size // 2
    bot_crop = frame_y_center + y_size // 2

    return left_crop, top_crop, right_crop, bot_crop


def save_processsed_seq(out_path: str, out_seq: Sequence[Image.Image], control_inputs: DataFrame,
                        process_seq: bool = True):
    Path(out_path).mkdir(parents=True, exist_ok=True)
    for i, aug_img in enumerate(out_seq):
        flip_channels = not process_seq
        processed = process_image(aug_img, flip_channels=flip_channels)
        processed.save(os.path.join(out_path, f"{str(i).zfill(6)}.png"))

    control_inputs.to_csv(os.path.join(out_path, "data_out.csv"), index=False)


def zoom_at(img: Image.Image, x: int, y: int, zoom: float) -> Image.Image:
    """
    PIL helper that combines crop and resize to zoom into an image at a point. From
    https://stackoverflow.com/questions/46149003/pil-zoom-into-image-at-a-particular-point

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
