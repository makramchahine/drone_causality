# Created by Patrick Kao at 4/26/22
from collections import defaultdict

import numpy as np
import tensorflow as tf
from numpy import ndarray
# perturb functions
# below vars are wrappers around tf functions, only explicity defined because want to have easy string name
from tensorflow.python.keras.layers import GaussianNoise

brightness_perturbation = tf.image.adjust_brightness
contrast_perturbation = tf.image.adjust_contrast
saturation_perturbation = tf.image.adjust_saturation


def darkness_perturbation(img: ndarray, delta: float):
    return tf.image.adjust_brightness(img, delta=-delta)


def noise_perturbation(img: ndarray, delta: float):
    img = img.astype(np.float32)
    noise_layer = GaussianNoise(delta)
    return noise_layer(img, training=True)


def final_distance(seq_1: ndarray, seq_2: ndarray):
    return np.sum(np.abs(seq_1[-1] - seq_2[-1]))


def pointwise_distance(seq_1: ndarray, seq_2: ndarray):
    return np.sum(np.abs(seq_1 - seq_2))


def switch_top_keys(distances):
    to_ret = defaultdict(list)
    for delta, delta_distance in distances.items():
        for model_name, distance in delta_distance.items():
            to_ret[model_name].append(distance)

    return to_ret
