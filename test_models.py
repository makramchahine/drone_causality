#%%

#!/usr/bin/python3

import tensorflow as tf
from tf_data_loader import get_output_normalization
from tensorflow import keras
import kerasncp as kncp
from keras_models import generate_lstm_model
import glob
import os
import numpy as np
#import matplotlib
from node_cell import *
from tf_cfc import CfcCell, MixedCfcCell

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from kerasncp.tf import LTCCell

rnn_size = 128

IMAGE_SHAPE = (144, 256, 3)
DROPOUT = 0.1
training_root = '/home/ramin/devens_drone_data/devens_12102021_sliced/1628106866.84_1'
output_means, output_std = get_output_normalization(training_root)
test_root = '/home/ramin/devens_drone_data/devens_12102021_sliced/1628106866.84'
#test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected_test/1628114177.28/'
#test_root = '/home/ramin/devens_drone_data/october_devens/1635524963.70/'
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


#test_model_names = [
#    #'rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-047_val_loss:0.2574_mse:0.0683_2021:09:24:09:22:31.hdf5']
#    #'models-from-server/rev-0_model-ncp_seq-64_opt-adam_lr-0.000800_crop-0.000000_epoch-049_val_loss:0.2528_mse:0.2151_2021:11:10:19:28:04.hdf5']
#    'models-from-server/rev-0_model-lstm_seq-256_opt-adam_lr-0.000800_crop-0.000000_epoch-036_val_loss:0.2496_mse:0.0147_2021:11:11:11:10:54.hdf5']
#    #'models-from-server/rev-0_model-ctrnn_ctt-mixedcfc_cn-1.000000_bba-silu_bb-dr-0.100000_fb-1.600000_bbu-128_bbl-1_wd-0.000001_mixed-0_seq-128_opt-adam_lr-0.000800_crop-0.000000_epoch-039_val-loss:0.3340_mse:0.0866_2021:11:10:20:29:31.hdf5']
#

# test_model_names = ['models-from-server/rev-0_model-ncp_seq-64_opt-adam_lr-0.000800_crop-0.000000_epoch-049_val_loss:0.2528_mse:0.2151_2021:11:10:19:28:04.hdf5']
# test_model_names = ['models-from-server/rev-0_model-ctrnn_ctt-mixedcfc_cn-1.000000_bba-silu_bb-dr-0.100000_fb-1.600000_bbu-128_bbl-1_wd-0.000001_mixed-0_seq-128_opt-adam_lr-0.000800_crop-0.000000_epoch-039_val-loss:0.3340_mse:0.0866_2021:11:10:20:29:31.hdf5']
# test_model_names = ['models-from-server/rev-0_model-lstm_seq-256_opt-adam_lr-0.000800_crop-0.000000_epoch-036_val_loss:0.2496_mse:0.0147_2021:11:11:11:10:54.hdf5']
# test_model_names = ['rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-047_val_loss:0.2574_mse:0.0683_2021:09:24:09:22:31.hdf5']

#test_model_names = ['rev-0_model-ctrnn_ctt-mixedcfc_cn-1.000000_bba-silu_bb-dr-0.100000_fb-1.600000_bbu-128_bbl-1_wd-0.000001_mixed-0_seq-68_opt-adam_lr-0.000900_crop-0.000000_epoch-099_val-loss:0.2612_mse:0.0076_2021:12:10:18:32:23.hdf5']
test_model_names = ['rev-0_model-ctrnn_ctt-mixedcfc_cn-1.000000_bba-silu_bb-dr-0.100000_fb-1.600000_bbu-128_bbl-1_wd-0.000001_mixed-0_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-097_val-loss:0.3013_mse:0.0213_2021:12:11:19:15:47.hdf5']

#test_model_names = ['rev-0_model-lstm_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-099_val_loss:0.2741_mse:0.0130_2021:12:12:13:27:59.hdf5']

test_model_names = ['rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-063_val_loss:0.2497_mse:0.0999_2021:12:11:16:22:54.hdf5']
test_model_names = ['rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-029_val_loss:0.2241_mse:0.2651_2021:12:13:10:59:52.hdf5']

checkpoint = ['./model-checkpoints/' + n for n in test_model_names]
checkpoint = checkpoint[0]


if 'ncp' in test_model_names[0]:
    model_name = 'ncp'
elif 'mixedcfc' in test_model_names[0]:
    model_name = 'mixedcfc'
elif 'lstm' in test_model_names[0]:
    model_name = 'lstm'
else:
    print('You shall not pass!')

weights_file_name = checkpoint

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

# # Conv Layers for 'rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-047_val_loss:0.2574_mse:0.0683_2021:09:24:09:22:31.hdf5'
# x = keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(3, 3), activation='relu')(x)
# x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
# x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
# x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)

x = keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
x = keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
x = keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)

# fully connected layers
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=128, activation='linear')(x)
pre_recurrent_layer = keras.layers.Dropout(rate=DROPOUT)(x)

if model_name == 'ncp':

    wiring = kncp.wirings.NCP(
        inter_neurons=18,  # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        motor_neurons=4,  # Number of motor neurons
        sensory_fanout=8,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
        seed = 5555
        )
    rnn_cell = LTCCell(wiring, ode_unfolds=6)
    inputs_state = tf.keras.Input(shape=(rnn_cell.state_size,))

    motor_out, output_states = rnn_cell(pre_recurrent_layer, inputs_state)
    single_step_model = tf.keras.Model([inputs, inputs_state], [motor_out, output_states])

    single_step_model.load_weights(weights_file_name)
    #single_step_model.set_weights(weights_list)


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
            #print("hidden.shape", hidden)
            motor_out, hidden = single_step_model([data_x[:, t], hidden])
            #print("all", hidden)
            #print("all", len(hidden))
            hidden_states.append(np.reshape(hidden, (1, state_size))) ## @Ramin please confirm hidden[0] vs hidden
            outputs.append(motor_out)
        return tf.stack(outputs, axis=1), tf.stack(hidden_states, axis=1)

    
    # Now we can infer the hidden state
    outputs, states = infer_hidden_states(single_step_model, rnn_cell.state_size, evaluation_data)
    print("Hidden states of first example ", states[0])

    for i in range(wiring.units):
        print("Neuron {:0d} is a {:} neuron".format(i, wiring.get_type_of_neuron(i)))

elif model_name == 'lstm':
    rnn_cell = tf.keras.layers.LSTMCell(rnn_size)
    c_state = tf.keras.Input(shape=(rnn_cell.state_size[0]))
    h_state = tf.keras.Input(shape=(rnn_cell.state_size[1]))

    output, [next_c, next_h] = rnn_cell(pre_recurrent_layer, [c_state, h_state])
    output = tf.keras.layers.Dense(units=4, activation='linear')(output)
    single_step_model = tf.keras.Model([inputs, c_state, h_state], [next_c, next_h, output])

    single_step_model.load_weights(weights_file_name)

    def infer_hidden_states(single_step_model, state_size, data_x):

        batch_size = data_x.shape[0]
        seq_len = data_x.shape[1]
        hidden_c = tf.zeros((batch_size, state_size[0]))
        hidden_h = tf.zeros((batch_size, state_size[1]))
        hidden_state_c = [hidden_c]
        hidden_state_h = [hidden_h]
        outputs = []
        for t in range(seq_len):
            # Compute new hidden state from old hidden state + input at time t
            #print("hidden.shape", hidden_c)
            hidden_c, hidden_h, motor_out= single_step_model([data_x[:, t], hidden_c, hidden_h])
            #print("all", hidden)
            #print("all", len(hidden))
            hidden_state_c.append(hidden_c[0])
            hidden_state_h.append(hidden_h[0])
            outputs.append(motor_out)
        return tf.stack(outputs, axis=1), tf.stack(hidden_c, axis=1), tf.stack(hidden_h, axis=1)

    # Now we can infer the hidden state
    outputs, states_c, states_h = infer_hidden_states(single_step_model, rnn_cell.state_size, evaluation_data)
    #print("Hidden states of first example ", states[0])

elif model_name == 'mixedcfc':
    # CONFIG = {
    #     "clipnorm": 1,
    #     "size": 128,
    #     "backbone_activation": "silu",
    #     "backbone_dr": 0.1,
    #     "forget_bias": 1.6,
    #     "backbone_units": 128,
    #     "backbone_layers": 1,
    #     "weight_decay": 1e-06,
    #     "use_mixed": False,
    # }

    CONFIG = {
        "clipnorm": 1,
        "size": rnn_size,
        "backbone_activation": "silu",
        "backbone_dr": 0.1,
        "forget_bias": 1.6,
        "backbone_units": 128,
        "backbone_layers": 1,
        "weight_decay": 1e-06,
        "use_mixed": True,
    }

    rnn_cell = MixedCfcCell(units=rnn_size, hparams=CONFIG)

    c_state = tf.keras.Input(shape=(rnn_cell.state_size[0]))
    h_state = tf.keras.Input(shape=(rnn_cell.state_size[1]))

    output, [next_c, next_h] = rnn_cell(pre_recurrent_layer, [c_state, h_state])
    output = tf.keras.layers.Dense(units=4, activation='linear')(output)
    single_step_model = tf.keras.Model([inputs, c_state, h_state], [next_c, next_h, output])

    single_step_model.load_weights(weights_file_name)

    def infer_hidden_states(single_step_model, state_size, data_x):

        batch_size = data_x.shape[0]
        seq_len = data_x.shape[1]
        hidden_c = tf.zeros((batch_size, state_size[0]))
        hidden_h = tf.zeros((batch_size, state_size[1]))
        hidden_state_c = [hidden_c]
        hidden_state_h = [hidden_h]
        outputs = []
        for t in range(seq_len):
            # Compute new hidden state from old hidden state + input at time t
            #print("hidden.shape", hidden_c)
            hidden_c, hidden_h, motor_out= single_step_model([data_x[:, t], hidden_c, hidden_h])
            #print("all", hidden)
            #print("all", len(hidden))
            hidden_state_c.append(hidden_c[0])
            hidden_state_h.append(hidden_h[0])
            outputs.append(motor_out)
        return tf.stack(outputs, axis=1), tf.stack(hidden_c, axis=1), tf.stack(hidden_h, axis=1)

    # Now we can infer the hidden state
    outputs, states_c, states_h = infer_hidden_states(single_step_model, rnn_cell.state_size, evaluation_data)


# print(np.shape(outputs))
# plt.plot(labels[:,1])
# plt.plot(outputs[0,:,1])
# plt.show()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(labels[:,0])
axs[0, 0].plot(outputs[0,:,0])
axs[0, 0].set_title('V_X')

axs[0, 1].plot(labels[:,1])
axs[0, 1].plot(outputs[0,:,1])
axs[0, 1].set_title('V_Y')

axs[1, 0].plot(labels[:,2])
axs[1, 0].plot(outputs[0,:,2])
axs[1, 0].set_title('V_Z')

axs[1, 1].plot(labels[:,3])
axs[1, 1].plot(outputs[0,:,3])
axs[1, 1].set_title('W')

plt.show()


