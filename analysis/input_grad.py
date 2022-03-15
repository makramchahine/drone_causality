# Created by Patrick Kao at 3/10/22
from typing import Sequence, Union

import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.python.keras.models import Functional


def compute_input_grad(img: Union[Tensor, ndarray], model: Functional, hiddens: Sequence[Tensor]):
    """
    Computes gradients of model output with respect to img
    :param img:
    :param model:
    :param hiddens:
    :return: tuple of image tensor (height, width, 1) with shape of input img, list of hidden vectors each of shape
    (batch, hidden_dim)
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        out = model([img, *hiddens])
        preds = out[0]
        hiddens = out[1:]

    grads = tape.jacobian(preds, img)[0]  # shape: 4 x 1 x height x width x channels
    grads = tf.math.abs(grads)  # take absolute value so + and - impacts don't cancel each other out

    heatmap = tf.math.reduce_sum(grads, axis=0)  # shape 1 x height x width x channels
    heatmap = tf.squeeze(heatmap, axis=0)
    # convert heatmap to black and white by summing channels
    heatmap = tf.math.reduce_sum(heatmap, axis=-1, keepdims=True)
    return heatmap, hiddens
