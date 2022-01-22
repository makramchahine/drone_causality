import os
from pathlib import Path
from typing import List, Iterable, Optional

import PIL.Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras, Tensor
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.models import Functional

from keras_models import load_model_from_weights

IMAGE_SHAPE = (256, 144, 3)


def convert_to_color_frame(saliency_map: Tensor):
    """
    Converts tensorflow tensor (1 channel) to 3-channel grayscale numpy array for use with OpenCV
    """
    one_channel_frame = saliency_map.numpy()
    repeated = np.repeat(one_channel_frame, 3, axis=-1)
    int_image = cv2.normalize(repeated, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return int_image


def image_dir_generator(data_path: str):
    """
    Iterates through all of the pngs in the data_path folder, loads them as numpy array, and resizes them
    to IMAGE_SHAPE. Yields the images using a generator for iteration
    """
    contents = os.listdir(data_path)
    contents = [os.path.join(data_path, c) for c in contents if 'png' in c]
    contents.sort()
    for path in contents:
        img = PIL.Image.open(path)
        if img is not None:
            resized = img.resize(IMAGE_SHAPE[:2], PIL.Image.BILINEAR)
            img_numpy = tf.keras.preprocessing.image.img_to_array(resized)
            yield img_numpy


def visualbackprop_activations(activation_model: Functional,
                               activations: List[Tensor],
                               kernels: List[Iterable] = None,
                               strides: List[Iterable] = None):
    """
    Compute the saliency maps for activation_model by running the VisualBackProp algorithm
    as described in https://arxiv.org/pdf/1611.05418.pdf
    """
    # infer CNN kernels, strides, from layers
    if not (kernels and strides):
        kernels, strides = [], []
        # don't infer form initial input layer so start at 1
        for layer in activation_model.layers[1:]:
            assert isinstance(layer, Conv2D), "Expect model to have only convolutional layers"
            kernels.append(layer.kernel_size)
            strides.append(layer.strides)

    average_layer_maps = []
    for layer_activation in activations:  # Only the convolutional layers
        feature_maps = layer_activation[0]
        n_features = feature_maps.shape[-1]
        average_feature_map = np.sum(feature_maps, axis=-1) / n_features

        # normalize map
        map_min = np.min(average_feature_map)
        map_max = np.max(average_feature_map)
        normal_map = (average_feature_map - map_min) / (map_max - map_min + 1e-6)
        # dim: height x width
        average_layer_maps.append(normal_map)

    # add batch and channels dimension to tensor
    average_layer_maps = [fm[np.newaxis, :, :, np.newaxis] for fm in average_layer_maps]  # dim: bhwc
    saliency_mask = tf.convert_to_tensor(average_layer_maps[-1])
    for l in reversed(range(0, len(average_layer_maps))):
        kernel = np.ones((*kernels[l], 1, 1))

        if l > 0:
            output_shape = average_layer_maps[l - 1].shape
        else:
            # TODO for some reason, the layer maps are transposed compared to the images (short dimension first)
            # therefore, the height and width in the image shape need to be reversed
            output_shape = (1, *(IMAGE_SHAPE[:2][::-1]), 1)

        saliency_mask = tf.nn.conv2d_transpose(saliency_mask, kernel, output_shape, strides[l], padding='VALID')
        if l > 0:
            saliency_mask = tf.multiply(saliency_mask, average_layer_maps[l - 1])

    saliency_mask = saliency_mask[0]  # remove batch dimension
    return saliency_mask


def get_conv_head(model_path: str, model_type: str):
    """
    Loads the model at model_path from weights and extracts only the convolutional
    pre-processing layers in a new Tensorflow model
    """
    # TODO: investigate single step vs sequence input types
    # don't care about loading initial hidden states, use _ to not worry about type of hidden state returned
    vis_model, _ = load_model_from_weights(model_type, model_path)
    # cleave off only convolutional head
    num_conv_layers = 4 if model_type == "ncp_old" else 5
    # slice at 1 to throw away first input layer
    conv_layers = vis_model.layers[1:num_conv_layers + 1]

    act_model_inputs = vis_model.input[0]  # don't want to take in hidden state, just image
    activation_model = keras.models.Model(inputs=act_model_inputs,
                                          outputs=[layer.output for layer in conv_layers])
    return activation_model


def run_visualbackprop(model_path: str, model_type: str, data_path: str, image_output_path: Optional[str] = None,
                       video_output_path: Optional[str] = None):
    """
    Runner script that loads images, runs VisualBackProp, and saves saliency maps
    """
    assert image_output_path or video_output_path, "No output creation set"

    # create output_dir if not present
    Path(image_output_path).mkdir(parents=True, exist_ok=True)
    activation_model = get_conv_head(model_path, model_type)
    if video_output_path:
        image_shape = list(IMAGE_SHAPE[:2])
        # assume og and saliency shapes the same, and stacked vertically
        image_shape[1] *= 2  # opencv wants frame size as width, height, so don't reverse
        writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, image_shape)

    for i, img in enumerate(image_dir_generator(data_path)):
        # comptute saliency map
        img_batched_tensor = tf.expand_dims(img, axis=0)
        # reverse channels of image to match training
        img_batched_tensor_reversed = img_batched_tensor[..., ::-1]
        activations = activation_model(img_batched_tensor_reversed)
        saliency = visualbackprop_activations(activation_model, activations)
        # save saliency map
        saliency_writeable = convert_to_color_frame(saliency)
        if image_output_path:
            cv2.imwrite(f"{image_output_path}/saliency_mask_{i}.png", saliency_writeable)
        if video_output_path:
            # display OG frame and saliency map stacked top and bottom
            og_int = np.uint8(img)
            side_by_side = np.concatenate([og_int, saliency_writeable], axis=0)
            writer.write(side_by_side)

    if video_output_path:
        writer.release()
