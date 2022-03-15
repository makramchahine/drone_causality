# Created by Patrick Kao at 3/9/22
from math import ceil
from typing import Optional, Sequence, Union

import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.models import Functional

from analysis.vis_utils import image_grid
from utils.model_utils import load_model_from_weights, load_model_no_params, ModelParams


def compute_gradcam(img: Union[Tensor, ndarray], grad_model: Functional, hiddens: Sequence[Tensor],
                    pred_index: Optional[Sequence[Tensor]] = None):
    heatmaps, hiddens = _compute_gradcam(img=img, grad_model=grad_model, hiddens=hiddens, pred_index=pred_index)
    avg_heat = tf.math.add_n(heatmaps)
    avg_heat = tf.expand_dims(avg_heat, axis=-1)
    return avg_heat, hiddens


def compute_gradcam_tile(img: Union[Tensor, ndarray], grad_model: Functional, hiddens: Sequence[Tensor],
                    pred_index: Optional[Sequence[Tensor]] = None):
    heatmaps, hiddens = _compute_gradcam(img=img, grad_model=grad_model, hiddens=hiddens, pred_index=pred_index)
    num_rows = ceil(len(heatmaps)/2)
    return image_grid(imgs=heatmaps, rows=num_rows, cols=2), hiddens



def _compute_gradcam(img: Union[Tensor, ndarray], grad_model: Functional, hiddens: Sequence[Tensor],
                     pred_index: Optional[Sequence[Tensor]] = None):
    """
    Adaptation of grad-cam code at https://keras.io/examples/vision/grad_cam/ with
    the following adjustments:

    - because we want the impact of pixels not on a class decision but on any of the 4 axes, sum heatmaps for all 4 outputs
    - because we don't care about positive or negative impact, drop the ReLU (wasn't in this implementation to begin with)
    - Before adding heatmaps together, take absolute value of each heatmap because don't care if positive or negative contribution to direction
    - idea (not implemented): instead of just upscaling, multiply grad-cam heatmap against visual backprop heatmap to model actual network weights
    :return tuple of image tensor (height, width, 1) with shape of final conv layer, list of hidden vectors each of shape
    (batch, hidden_dim)
    """
    if pred_index is None:
        pred_index = range(grad_model.output_shape[1][-1])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        out = grad_model([img, *hiddens])
        last_conv_layer_output = out[0]
        preds = out[1]
        hiddens = out[2:]

    heatmaps = []
    # for each element of preds, compute gradient of last_conv_out wrt this element of pred, abs and sum these gradients
    # strip batch dim
    # jacobian shape 4x last_conv_layer_output.shape where each element is gradient, preds[:,i] wrt last_conv_layer_out
    grads = tape.jacobian(preds, last_conv_layer_output)[0]
    last_conv_layer_output = last_conv_layer_output[0]
    for pred in pred_index:
        # This is the gradient of the output neuron (top pred1icted or chosen)
        # with regard to the output feature map of the last conv layer
        grad = grads[pred]

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grad, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # patrick edit: absolute value heatmaps to not discount/cancel negative and positive contributions
        heatmap = tf.math.abs(heatmap)

        heatmaps.append(heatmap)

    return heatmaps, hiddens


def get_last_conv(model_path: str, model_params: Optional[ModelParams] = None) -> Model:
    if model_params is not None:
        model_params.single_step = True
        vis_model = load_model_from_weights(model_params, checkpoint_path=model_path)
    else:
        vis_model = load_model_no_params(model_path, single_step=True)

    # get last conv layer
    # cleave off only convolutional head
    conv_layers = [layer for layer in vis_model.layers if isinstance(layer, Conv2D)]

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    return tf.keras.models.Model(
        [vis_model.inputs], [conv_layers[-1].output, *vis_model.output]
    )
