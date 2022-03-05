#!/usr/bin/env python
import argparse
import os
from typing import Optional, Union

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib
import pandas as pd
from tensorflow import Tensor
from tqdm import tqdm

from helper_scripts.visualize_training_runs import visualize_run
from node_cell import *
from utils.data_utils import image_dir_generator, CSV_COLUMNS
from utils.model_utils import load_model_from_weights, load_model_no_params, \
    generate_hidden_list, NCPParams, LSTMParams, CTRNNParams, TCNParams

rnn_size = 128
IMAGE_SHAPE = (144, 256, 3)


# test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected_test/1628114177.28/'


def load_data(data_path: str):
    labels = np.genfromtxt(os.path.join(data_path, 'data_out.csv'), delimiter=',', skip_header=1)
    frames = list(image_dir_generator(data_path, IMAGE_SHAPE))
    frame_stack_np = np.expand_dims(np.stack(frames, axis=0), axis=0)  # stack and add batch dim
    return frame_stack_np, labels


def load_model(model_path: str, params_repr: Optional[str] = None):
    # infer params representation
    if params_repr is None:
        return load_model_no_params(model_path, single_step=True)
    else:
        params: Union[NCPParams, LSTMParams, CTRNNParams, TCNParams] = eval(params_repr)
        params.single_step = True
        return load_model_from_weights(params, model_path)


def infer_hidden_states(model, data_x):
    """
        Infers the hidden states of a single-step RNN model

        @param model: single-step RNN model. Takes in ist where first elements are of shape (batch, hidden_dim) and last
        el is (batch, h, w, channels)
        @param data_x: dataset in shape (batch, h, w, channels)
        @return:
            outputs: Tensor of shape (batch_size,sequence_length+1,state_size)
            hiddens: List num_hiddens long with tensors of shape (batch_size, sequence_length+1, hidden_dim)

    """
    seq_len = data_x.shape[1]

    # assume 1st input is image, all other inputs are hidden with shape batch, hidden_dim. Create all hidden states
    hiddens = generate_hidden_list(model=model, return_numpy=True)

    hidden_states = []  # shape: seq_len x (number of hidden states x hidden shape)
    outputs = []
    for t in tqdm(range(seq_len)):
        hidden_states.append(hiddens)
        # Compute new hidden state from old hidden state + input at time t
        out = model.predict([data_x[:, t], *hiddens])
        motor_out = out[0]
        hiddens = out[1:]  # list num_hidden long, each el is batch x hidden_dim
        outputs.append(motor_out)

    hiddens_stacked = []
    for i in range(len(hidden_states[0])):
        # extract ith column of hiddens and stack as tensor
        hiddens_stacked.append(tf.stack([el[i] for el in hidden_states], axis=1))
    return tf.stack(outputs, axis=1), hiddens_stacked


def plot_outputs(labels, outputs):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(labels[:, 0])
    axs[0, 0].plot(outputs[0, :, 0])
    axs[0, 0].set_title('V_X')

    axs[0, 1].plot(labels[:, 1])
    axs[0, 1].plot(outputs[0, :, 1])
    axs[0, 1].set_title('V_Y')

    axs[1, 0].plot(labels[:, 2])
    axs[1, 0].plot(outputs[0, :, 2])
    axs[1, 0].set_title('V_Z')

    axs[1, 1].plot(labels[:, 3])
    axs[1, 1].plot(outputs[0, :, 3])
    axs[1, 1].set_title('W')
    plt.show()


def save_out_video(outputs: Tensor, data_path: str, output_path: str = "test_out"):
    """

    :param outputs: Tensor of shape 1, time, 4
    :param data_path:
    :param output_path:
    :return:
    """
    df = pd.DataFrame(columns=CSV_COLUMNS)
    for i, out in enumerate(outputs[0]):
        df.loc[i, "vx"] = float(out[0])
        df.loc[i, "vy"] = float(out[1])
        df.loc[i, "vz"] = float(out[2])
        df.loc[i, "omega_z"] = float(out[3])

    csv_path = f"{output_path}.csv"
    df.to_csv(csv_path, index=False)

    visualize_run(run_dir=data_path, output_path=f"{output_path}.mp4", csv_path=csv_path)


def test_models(data_path: str, model_path: str, params_repr: Optional[str] = None):
    eval_data, labels = load_data(data_path)
    model = load_model(model_path, params_repr=params_repr)
    outputs, hiddens = infer_hidden_states(model, eval_data)

    plot_outputs(labels, outputs)

    save_out_video(outputs, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Filepath of test dataset")
    parser.add_argument("model_path", type=str, help="Filepath of model checkpoint")
    parser.add_argument("--params", type=str, help="repr() string of model params used during training", default=None)
    args = parser.parse_args()
    test_models(data_path=args.data_path, model_path=args.model_path, params_repr=args.params)
