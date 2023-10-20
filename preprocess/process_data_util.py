from typing import List, Tuple
import numpy as np
from PIL import Image

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from keras_models import IMAGE_SHAPE

def resize_and_crop(img: Image.Image, desired_shape: tuple) -> Image.Image:
    aspect_ratio = img.size[0] / img.size[1]
    desired_aspect_ratio = desired_shape[0] / desired_shape[1]

    if aspect_ratio > desired_aspect_ratio:
        new_width = int(desired_shape[1] * aspect_ratio)
        new_height = desired_shape[1]
    else:
        new_width = desired_shape[0]
        new_height = int(desired_shape[0] / aspect_ratio)

    img = img.resize((new_width, new_height), Image.ANTIALIAS)

    left = (new_width - desired_shape[0]) / 2
    top = (new_height - desired_shape[1]) / 2
    right = (new_width + desired_shape[0]) / 2
    bottom = (new_height + desired_shape[1]) / 2

    img = img.crop((left, top, right, bottom))
    return img

def process_image(img: Image.Image, flip_channels: bool = True) -> Image.Image:
    """
    Applies image transformations to training Data. Resizes to IMAGE_SIZE
    :param img:
    :return:
    """
    desired_shape = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
    if img.size != desired_shape:
        img = resize_and_crop(img, desired_shape)
        # img = img.resize(desired_shape, resample=PIL.Image.BICUBIC)
    if flip_channels:
        # noinspection PyTypeChecker
        img = np.array(img)[:, :, ::-1]
        img = Image.fromarray(img)

    return img