from typing import List, Iterable, Optional, Union

import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow import keras, Tensor
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.models import Functional

from keras_models import IMAGE_SHAPE
from utils.model_utils import ModelParams, load_model_from_weights, load_model_no_params


def compute_visualbackprop(img: Union[Tensor, ndarray],
                           activation_model: Functional,
                           hiddens: Optional[List[Tensor]] = None,
                           kernels: Optional[List[Iterable]] = None,
                           strides: Optional[List[Iterable]] = None):
    """
    Compute the saliency maps for activation_model by running the VisualBackProp algorithm
    as described in https://arxiv.org/pdf/1611.05418.pdf
    :param activation_model: keras model that only has convolutional layers of model. used to infer kernels and strides
    :param kernels: alternative to passing in model, num_conv_layers x 2 list of kernel sizes
    :param strides: num_conv_layers long list of model strides
    :return: keras tensor of shape hxwx1 that represents the saliency map of the given activations
    """
    # infer CNN kernels, strides, from layers
    if not (kernels and strides):
        kernels, strides = [], []
        # don't infer form initial input layer so start at 1
        for layer in activation_model.layers[1:]:
            if isinstance(layer, Conv2D):
                kernels.append(layer.kernel_size)
                strides.append(layer.strides)

    activations = activation_model.predict(img)
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
            # therefore, the height and width in the image shape need to be reversed
            output_shape = (1, *(IMAGE_SHAPE[:2]), 1)

        saliency_mask = tf.nn.conv2d_transpose(saliency_mask, kernel, output_shape, strides[l], padding='VALID')
        if l > 0:
            saliency_mask = tf.multiply(saliency_mask, average_layer_maps[l - 1])

    saliency_mask = tf.squeeze(saliency_mask, axis=0)  # remove batch dimension
    return saliency_mask, hiddens if hiddens else saliency_mask


def get_conv_head(model_path: str, model_params: Optional[ModelParams] = None):
    """
    Loads the model at model_path from weights and extracts only the convolutional
    pre-processing layers in a new Tensorflow model
    """
    # don't care about loading initial hidden states, use _ to not worry about type of hidden state returned
    if model_params is not None:
        model_params.single_step = True
        vis_model = load_model_from_weights(model_params, checkpoint_path=model_path)
    else:
        vis_model = load_model_no_params(model_path, single_step=True)
    # cleave off only convolutional head
    conv_layers = [layer for layer in vis_model.layers if isinstance(layer, Conv2D)]

    act_model_inputs = vis_model.inputs[0]  # don't want to take in hidden state, just image
    activation_model = keras.models.Model(inputs=act_model_inputs,
                                          outputs=[layer.output for layer in conv_layers])
    return activation_model


