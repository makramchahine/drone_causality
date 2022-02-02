#!/usr/bin/env python
import argparse
import os
from typing import Optional

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib
from tqdm import tqdm

from utils.model_utils import NCPParams, LSTMParams, CTRNNParams, load_model_from_weights, load_model_no_params
from node_cell import *
from utils.data_utils import image_dir_generator

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
        params = exec(params_repr)
        return load_model_from_weights(params, model_path, single_step=True)


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
    batch_size = data_x.shape[0]
    seq_len = data_x.shape[1]

    # assume 1st input is image, all other inputs are hidden with shape batch, hidden_dim. Create all hidden states
    hiddens = [np.zeros((batch_size, input_shape[1])) for input_shape in model.input_shape[1:]]

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


def test_models(data_path: str, model_path: str, params_repr: Optional[str] = None):
    eval_data, labels = load_data(data_path)
    model = load_model(model_path, params_repr=params_repr)
    outputs, hiddens = infer_hidden_states(model, eval_data)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Filepath of test dataset")
    parser.add_argument("model_path", type=str, help="Filepath of model checkpoint")
    parser.add_argument("--params", type=str, help="repr() string of model params used during training", default=None)
    args = parser.parse_args()
    test_models(data_path=args.data_path, model_path=args.model_path, params_repr=args.params)
