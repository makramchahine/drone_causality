import os
from typing import Iterable

import PIL.Image
import tensorflow as tf


def image_dir_generator(data_path: str, image_shape: Iterable):
    """
    Iterates through all of the pngs in the data_path folder, loads them as numpy array, and resizes them
    to IMAGE_SHAPE. Yields the images using a generator for iteration
    @param data_path: file path of dir with images
    @param image_shape: height, width, channels tuple
    """
    contents = os.listdir(data_path)
    contents = [os.path.join(data_path, c) for c in contents if 'png' in c]
    contents.sort()
    for path in contents:
        img = PIL.Image.open(path)
        if img is not None:
            # image shape is height, width, PIL takes width, height
            resized = img.resize(image_shape[:2][::-1], PIL.Image.BILINEAR)
            img_numpy = tf.keras.preprocessing.image.img_to_array(resized)
            yield img_numpy
