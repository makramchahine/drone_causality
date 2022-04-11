# Created by Patrick Kao at 4/6/22
# file can't be called shap or iport doesn't work
import copy
import os
import pickle
import random
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
from numpy import ndarray
from tensorflow import Tensor, keras
from tensorflow.python.keras.models import Functional, Model
from tqdm import tqdm

from keras_models import IMAGE_SHAPE
from utils.data_utils import image_dir_generator
from utils.model_utils import generate_hidden_list


# to get this to work with shap v0.40.0 and tf 2.4.1, need to use my fork at https://github.com/dolphonie/shap
# this library has the following changes to deep_tf.py library in shap
# replace line 284 with model_output_ranks = np.mean(np.tile(np.arange(len(self.phi_symbolics)), (X[0].shape[0], 1)), axis=0, keepdims=True, dtype=int)
# replace line 312 with phis[l][j] = np.mean(sample_phis[l][bg_data[l].shape[0]:] * (X[l][j] - bg_data[l]), axis=0)
# on line 329, replace assert with assert np.abs(diffs).max() < 1

class FirstOutput(keras.layers.Layer):
    def __init__(self):
        super(FirstOutput, self).__init__()

    def call(self, inputs):
        return inputs[0]


HIDDEN_TEXT_HEIGHT = 30


def compute_shap(img: Union[Tensor, ndarray],
                 model: Functional,
                 hiddens: List[Tensor], dataset_path: str, cache_path: str, show_hidden_contribution: bool = True):
    # TODO: maybe display image graph?
    # don't require this import unless necessary
    import shap

    # run inference
    model_inputs = [img, *hiddens]
    out = model.predict(model_inputs)
    hiddens = out[1:]

    # create new model that only outputs control and not hidden, so only control is looked at by shap
    x = FirstOutput()(model.layers[-1].output)
    control_model = Model(inputs=model.input, outputs=x)
    example_dataset = generate_shap_dataset(model, dataset_path, cache_path=cache_path)
    example_dataset = [el[:100] for el in example_dataset]
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    e = shap.DeepExplainer(control_model, example_dataset)

    # list of 4 values for each output signal, value is list of 2 ndarrays, shap for image and shap for hidden
    shap_values = e.shap_values(model_inputs)
    image_shaps = [np.abs(dim_shap[0]) for dim_shap in shap_values]
    sum_shap = np.sum(image_shaps, axis=(0, -1))
    h_w_1_shap = np.expand_dims(np.squeeze(sum_shap, 0), -1)

    hidden_contribution = None
    if show_hidden_contribution:
        text_img = np.zeros((HIDDEN_TEXT_HEIGHT, img.shape[2], 3), dtype=np.uint8)
        hidden_shaps = [np.abs(dim_shap[1:]) for dim_shap in shap_values]
        hidden_contribution = np.sum(hidden_shaps)
        hidden_text = f"Hidden Contrib: {hidden_contribution:.2f}"
        cv2.putText(text_img, hidden_text, (0, HIDDEN_TEXT_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255),
                    1,
                    cv2.LINE_AA)
        hidden_contribution = [text_img]

    return h_w_1_shap, hiddens, hidden_contribution


def generate_shap_dataset(model: Functional, dataset_path: str, num_runs: int = 1, cache_path: str = ""):
    """
    Helper function that doesn't need to be called during normal use, but instead generates numpy arrays that are used
    to feed the deepexplainer
    :return:
    """
    if not os.path.exists(cache_path):
        print(f"Generating cache at {cache_path}")
        Path(os.path.dirname(cache_path)).mkdir(exist_ok=True, parents=True)
        candidates = os.listdir(dataset_path)
        candidates = [os.path.join(dataset_path, rel) for rel in candidates]
        random.shuffle(candidates)
        candidates = candidates[:num_runs]
        sample_imgs = []
        sample_hiddens = []
        for data_dir in candidates:
            hiddens = generate_hidden_list(model=model, return_numpy=True)
            for img in tqdm(image_dir_generator(data_dir, IMAGE_SHAPE)):
                img_expanded = np.expand_dims(img, 0)
                sample_imgs.append(copy.deepcopy(img_expanded))
                sample_hiddens.append(copy.deepcopy(hiddens))
                out = model.predict([img_expanded, *hiddens])
                hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim

        # convert to numpy arrays
        sample_imgs = np.concatenate(sample_imgs, axis=0)
        sample_hiddens = [np.concatenate([hid[hid_index] for hid in sample_hiddens], axis=0) for hid_index in
                          range(len(sample_hiddens[0]))]
        with open(cache_path, "wb") as f:
            pickle.dump([sample_imgs, *sample_hiddens], f)

    with open(cache_path, "rb") as f:
        return pickle.load(f)
