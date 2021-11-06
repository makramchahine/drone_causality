#!/usr/bin/python3

import tensorflow as tf
from tf_data_loader import get_output_normalization
from tensorflow import keras
import kerasncp as kncp
from keras_models import generate_lstm_model
import glob
import os
import numpy as np
import matplotlib
from node_cell import *
from tf_cfc import CfcCell, MixedCfcCell

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from kerasncp.tf import LTCCell

model_name = 'ncp'
rnn_size = 128

IMAGE_SHAPE = (144, 256, 3)
DROPOUT = 0.1
training_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected'
output_means, output_std = get_output_normalization(training_root)
test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected_test/1628106965.56'
#test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected/1628106140.64/'
# test_root = '/home/ramin/devens_drone_data/october_devens/1635523982.23'
print('Loading data for run: %s' % test_root)

outputs = []
labels = np.genfromtxt(os.path.join(test_root, 'data_out.csv'), delimiter=',', skip_header=1)
n_frames = len([f for f in os.listdir(test_root) if 'png' in f])
frame_stack_np = np.zeros((n_frames, 144, 256, 3), dtype=np.float32)
for ix in range(n_frames):
    frame_stack_np[ix] = Image.open(os.path.join(test_root, '%06d.png' % ix))
frame_stack_np = np.expand_dims(frame_stack_np, 0)
evaluation_data = frame_stack_np


test_model_names = [
    #'rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-020_val_loss:0.2127_mse:0.1679_2021:09:20:02:24:31']
    #'rev-0_model-lstm_seq-64_opt-adam_lr-0.001000_crop-0.000000_epoch-046_val_loss:0.2867_mse:0.0046_2021:09:09:02:33:09']
    'rev-0_model-ctrnn_ct_type-mixedcfc_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-023_val_loss:0.2439_mse:0.0607_2021:09:19:11:25:20']

checkpoint = ['./model-checkpoints/' + n for n in test_model_names]


if 'ctrnn' in checkpoint:
    pass
else:
    last_model = tf.keras.models.load_model(checkpoint)
    weights_list = last_model.get_weights()

#layers = [l for l in last_model.layers]


inputs = keras.Input(shape=IMAGE_SHAPE)

rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)
normalization_layer = keras.layers.experimental.preprocessing.Normalization(mean=[0.41718618, 0.48529191, 0.38133072],
                                                                            variance=[.057, .05, .061])
x = rescaling_layer(inputs)
x = normalization_layer(x)
# my_input_model = keras.Model(inputs=inputs, outputs=x)

# model = keras.models.Sequential()
# model.add(my_input_model)
# Conv Layers
x = keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(3, 3), activation='relu')(x)
x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
# fully connected layers
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=128, activation='linear')(x)
pre_recurrent_layer = keras.layers.Dropout(rate=DROPOUT)(x)

if model_name == 'ncp':

    wiring = kncp.wirings.NCP(
        inter_neurons=18,  # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        motor_neurons=4,  # Number of motor neurons
        sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )
    rnn_cell = LTCCell(wiring)
    inputs_state = tf.keras.Input(shape=(rnn_cell.state_size,))

    motor_out, output_states = rnn_cell(pre_recurrent_layer, inputs_state)
    single_step_model = tf.keras.Model([inputs, inputs_state], [motor_out, output_states])

    # single_step_model.load_weights(checkpoint)
    single_step_model.set_weights(weights_list)


    def infer_hidden_states(single_step_model, state_size, data_x):
        """
            Infers the hidden states of a single-step RNN model
        Args:
            single_step_model: RNN model taking a pair (inputs,old_hidden_state) as input and outputting new_hidden_state
            state_size: Size of the RNN model (=number of units)
            data_x: Input data for which the hidden states should be inferred
        Returns:
            Tensor of shape (batch_size,sequence_length+1,state_size). The sequence starts with the initial hidden state
            (all zeros) and is therefore one time-step longer than the input sequence
        """
        batch_size = data_x.shape[0]
        seq_len = data_x.shape[1]
        hidden = tf.zeros((batch_size, state_size))
        hidden_states = [hidden]
        outputs = []
        for t in range(seq_len):
            # Compute new hidden state from old hidden state + input at time t
            print("hidden.shape", hidden)
            motor_out, hidden = single_step_model([data_x[:, t], hidden])
            print("all", hidden)
            print("all", len(hidden))
            hidden_states.append(np.reshape(hidden, (1, state_size))) ## @Ramin please confirm hidden[0] vs hidden
            outputs.append(motor_out)
        return tf.stack(outputs, axis=1), tf.stack(hidden_states, axis=1)

    
    # Now we can infer the hidden state
    outputs, states = infer_hidden_states(single_step_model, rnn_cell.state_size, evaluation_data)
    print("Hidden states of first example ", states[0])

    for i in range(wiring.units):
        print("Neuron {:0d} is a {:} neuron".format(i, wiring.get_type_of_neuron(i)))

# elif model_name == 'lstm':
#     rnn_cell = keras.layers.LSTMCell(rnn_size)
#     c_state = tf.keras.Input(shape=(rnn_cell.state_size[0]))
#     h_state = tf.keras.Input(shape=(rnn_cell.state_size[1]))
#
#     output, [next_c, next_h] = rnn_cell(pre_recurrent_layer, [c_state, h_state])
#     single_step_model = tf.keras.Model([inputs, c_state, h_state], [next_c, next_h, output])
#
#     single_step_model.set_weights(weights_list)
#
# elif model_name == 'mmrnn':
#     rnn_cell = mmRNN(units=rnn_size)
#     c_state = tf.keras.Input(shape=(rnn_cell.state_size[0]))
#     h_state = tf.keras.Input(shape=(rnn_cell.state_size[1]))
#
#     output, [next_c, next_h] = rnn_cell(pre_recurrent_layer, [c_state, h_state])
#     single_step_model = tf.keras.Model([inputs, c_state, h_state], [next_c, next_h, output])
#
#     single_step_model.set_weights(weights_list)

print(np.shape(outputs))
plt.plot(labels[:,1])
plt.plot(outputs[0,:,1])
plt.show()
