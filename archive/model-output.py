from re import S
import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp
from kerasncp.tf import LTCCell
from node_cell import *

import matplotlib.pyplot as plt
import matplotlib
import PIL.Image
import numpy as np
import os
import datetime
from functools import reduce
from operator import length_hint, mul
import pickle
import argparse

import rowan
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description='Train the model on deepdrone data')
parser.add_argument('--rnn_size', type=int, default=32, help='Select the size of RNN network you would like to train')
parser.add_argument('--seq_len', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--data_dir', type=str, default="./data/2021-06-19-03-37-00", help='Path to training data')
args = parser.parse_args()

IMAGE_OUTPUT_DIRECTORY    = "C:\\Users\\MIT Driverless\\Documents\\deepdrone\\image_output"
WEIGHTS_PATH              = 'C:\\Users\\MIT Driverless\\Documents\\deepdrone\\model-checkpoints\\ncp_encoder_decoder-2021_06_24_23_25_30-rev=13.0-weights.026-0.0494.hdf5'

# Setup the network
IMAGE_SHAPE     = (256,256,3)

encoder_inputs = keras.Input(shape=(args.seq_len, *IMAGE_SHAPE))
encoder        = encoder_inputs

convolutionalLayers = [
    keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu')),
    keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')),
    keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2), activation='relu')),
    keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8,  kernel_size=(2,2), strides=(2,2), activation='relu'))
]

for l in convolutionalLayers:
    encoder = l(encoder)

encoder = keras.layers.TimeDistributed(keras.layers.Flatten())(encoder)
encoder = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5))(encoder)
encoder = keras.layers.TimeDistributed(keras.layers.Dense(units=64, activation='linear'))(encoder)

encoderRnnCell = LTCCell(kncp.wirings.FullyConnected(64, 13))
encoder_outputs, hidden_state = keras.layers.RNN(encoderRnnCell, return_sequences=True, return_state=True)(encoder)

decoder_inputs = keras.Input(shape=(args.seq_len, *IMAGE_SHAPE))
decoder = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu'))(decoder_inputs)
decoder = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu'))(decoder)
decoder = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2), activation='relu'))(decoder)
decoder = keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8,  kernel_size=(2,2), strides=(2,2), activation='relu'))(decoder)
decoder = keras.layers.TimeDistributed(keras.layers.Flatten())(decoder)
decoder = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5))(decoder)
decoder = keras.layers.TimeDistributed(keras.layers.Dense(units=64, activation='linear'))(decoder)

decoderRnnCell = LTCCell(kncp.wirings.FullyConnected(64, 13))
decoder_outputs = keras.layers.RNN(decoderRnnCell, return_sequences=True)(decoder, initial_state=hidden_state)

encoder_decoder_model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model = encoder_decoder_model

model.compile(
    optimizer=keras.optimizers.Adam(0.0005), loss="mean_squared_error",
)

# Load weights
model.load_weights(WEIGHTS_PATH)
model.summary(line_length=80)
tf.keras.utils.plot_model(
    model,
)

# Load images
images = np.load(f'{args.data_dir}/images.npy')
states = np.load(f'{args.data_dir}/vectors.npy').view('<f4').reshape(args.seq_len, -1)

batch = np.zeros((1, args.seq_len, *IMAGE_SHAPE))
k = 0
statePredictions = []

for i in range(images.shape[0]):
    if i < args.seq_len:
        batch[0, i]  = images[i]
    else:
        batch[0]     = np.roll(batch[0], -1, axis=0)  
        batch[0][-1] = images[i]

    # if i == 9:
    #     plt.imshow(images[i])
    #     plt.show()

    prediction = model.predict([batch, batch])[0, min(i, args.seq_len-1)]
    statePredictions.append(prediction)

statePredictions = np.array(statePredictions)

print(statePredictions.shape)

# with open('vectors.p', 'wb') as f:
#     pickle.dump(statePredictions, f)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

# with open('vectors.p', 'rb') as f:
#     statePredictions = np.array(pickle.load(f))

print(statePredictions.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.quiver(statePredictions[:, 0], statePredictions[:, 1], statePredictions[:, 2], 
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[0, 0] for i in range(statePredictions.shape[0])],
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[1, 0] for i in range(statePredictions.shape[0])],
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[2, 0] for i in range(statePredictions.shape[0])],
    length=0.1,
    normalize=True,
    color='red'
)

ax.quiver(statePredictions[:, 0], statePredictions[:, 1], statePredictions[:, 2], 
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[0, 1] for i in range(statePredictions.shape[0])],
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[1, 1] for i in range(statePredictions.shape[0])],
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[2, 1] for i in range(statePredictions.shape[0])],
    length=0.1,
    normalize=True,
    color='blue'
)

ax.quiver(statePredictions[:, 0], statePredictions[:, 1], statePredictions[:, 2], 
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[0, 2] for i in range(statePredictions.shape[0])],
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[1, 2] for i in range(statePredictions.shape[0])],
    [R.from_quat(statePredictions[i, 3:7]).as_matrix()[2, 2] for i in range(statePredictions.shape[0])],
    length=0.1,
    normalize=True,
    color='green'
)

ax.quiver(states[:, 0], states[:, 1], states[:, 2], 
    [R.from_quat(states[i, 3:7]).as_matrix()[0, 0] for i in range(states.shape[0])],
    [R.from_quat(states[i, 3:7]).as_matrix()[1, 0] for i in range(states.shape[0])],
    [R.from_quat(states[i, 3:7]).as_matrix()[2, 0] for i in range(states.shape[0])],
    length=0.1,
    normalize=True,
    color='red'
)

ax.quiver(states[:, 0], states[:, 1], states[:, 2], 
    [R.from_quat(states[i, 3:7]).as_matrix()[0, 1] for i in range(states.shape[0])],
    [R.from_quat(states[i, 3:7]).as_matrix()[1, 1] for i in range(states.shape[0])],
    [R.from_quat(states[i, 3:7]).as_matrix()[2, 1] for i in range(states.shape[0])],
    length=0.1,
    normalize=True,
    color='blue'
)

ax.quiver(states[:, 0], states[:, 1], states[:, 2], 
    [R.from_quat(states[i, 3:7]).as_matrix()[0, 2] for i in range(states.shape[0])],
    [R.from_quat(states[i, 3:7]).as_matrix()[1, 2] for i in range(states.shape[0])],
    [R.from_quat(states[i, 3:7]).as_matrix()[2, 2] for i in range(states.shape[0])],
    length=0.1,
    normalize=True,
    color='green'
)

ax.scatter(statePredictions[:, 0], statePredictions[:, 1], statePredictions[:, 2])
ax.scatter(states[:, 0], states[:, 1], states[:, 2])
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='orange', marker = '*')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='blue', marker = '*')
ax.legend([scatter1_proxy, scatter2_proxy], ['position', 'prediction'], numpoints = 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.axes.set_xlim3d(left=-1, right=0) 
ax.axes.set_ylim3d(bottom=0, top=1) 
ax.axes.set_zlim3d(bottom=0, top=1)
plt.title('Drone Demonstration Task Path and Orientation')
plt.show()