import os
from typing import Tuple, Optional

import PIL.Image
import numpy as np
import tensorflow as tf
from numpy import ndarray

CSV_COLUMNS = ("vx", "vy", "vz", "omega_z")


def load_image(img_path: str, img_shape: Tuple[int, int, int], reverse_channels: bool = True) -> Optional[ndarray]:
    img = PIL.Image.open(img_path)
    if img is not None:
        # image shape is height, width, PIL takes width, height
        resized = img.resize(img_shape[:2][::-1], PIL.Image.BILINEAR)
        img_numpy = tf.keras.preprocessing.image.img_to_array(resized).astype(np.uint8)
        if reverse_channels:
            # reverse channels of image to match training
            img_numpy = img_numpy[..., ::-1]

        # add batch dim
        img_numpy = np.expand_dims(img_numpy, axis=0)
        return img_numpy
    else:
        return None


def image_dir_generator(data_path: str, image_shape: Tuple[int, int, int], reverse_channels: bool = False):
    """
    Iterates through all of the pngs in the data folder, loads them as numpy array, and resizes them
    to IMAGE_SHAPE. Yields the images using a generator for iteration
    @param data_path: file path of dir with images
    @param image_shape: height, width, channels tuple
    @param reverse_channels: reverse image channels
    """
    contents = os.listdir(data_path)
    contents = [os.path.join(data_path, c) for c in contents if 'png' in c]
    contents.sort()
    for path in contents:
        img = load_image(img_path=path, img_shape=image_shape, reverse_channels=reverse_channels)
        if img is not None:
            yield img
