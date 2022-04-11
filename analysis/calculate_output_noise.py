# Created by Patrick Kao at 4/4/22
import argparse

import numpy as np
from PIL import Image
from tensorflow import keras

from analysis.vis_utils import parse_params_json
from utils.model_utils import load_model_from_weights, get_readable_name, generate_hidden_list


def calculate_output_noise(params_path: str, input_img_path: str, noise: float, n_trials=50):
    input_img = np.expand_dims(np.array(Image.open(input_img_path), dtype=float), axis=0)
    model_variances = {}
    var_layer = keras.layers.GaussianNoise(stddev=noise)
    for local_path, model_path, model_params in parse_params_json(params_path):
        model = load_model_from_weights(model_params, model_path)
        imgs = []
        for _ in range(n_trials):
            # might want to try with nonzero hiddens at some point
            hiddens = generate_hidden_list(model=model, return_numpy=True)
            noise_img = var_layer(input_img, training=True)
            output = model.predict([noise_img, *hiddens])
            imgs.append(output[0])  # don't save hidden output

        channel_variances = np.var(imgs, axis=0)
        avg_variance = np.mean(channel_variances)
        model_variances[get_readable_name(model_params)] = avg_variance

    return model_variances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params_path")
    parser.add_argument("input_img")
    parser.add_argument("noise", type=float)
    args = parser.parse_args()
    print(calculate_output_noise(args.params_path, args.input_img, args.noise))
