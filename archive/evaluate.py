import pickle

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from tensorflow import keras
import kerasncp as kncp

TRAINING_SEQUENCE_LENGTH = 64
IMAGE_SHAPE              = (256,256,3)

# Test data
images     = np.array([np.load('data/2020-09-10-13-04-32/images.npy')])
directions = np.array([np.load('data/2020-09-10-13-04-32/positions.npy')])

# Setup the network
wiring = kncp.wirings.NCP(
    inter_neurons=12,   # Number of inter neurons
    command_neurons=8,  # Number of command neurons
    motor_neurons=3,    # Number of motor neurons
    sensory_fanout=4,   # How many outgoing synapses has each sensory neuron
    inter_fanout=4,     # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=4,   # Now many recurrent synapses are in the
                                    # command neuron layer
    motor_fanin=6,      # How many incomming syanpses has each motor neuron
)

rnnCell = kncp.LTCCell(wiring)

model = keras.models.Sequential()
model.add(keras.Input(shape=(None, *IMAGE_SHAPE)))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(2,2), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1000, activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=100,  activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.3)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=24,   activation='relu')))
model.add(keras.layers.RNN(rnnCell, return_sequences=True))

model.compile(
    optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
)

# Load weights
model.load_weights('model-checkpoints/weights.132--0.91.hdf5')
predictions = model.predict(images)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(predictions.shape)

ax.scatter(predictions[0,:,0], predictions[0,:,1], predictions[0,:,2], c='r', marker='o')
ax.scatter(directions[0,:,0], directions[0,:,1], directions[0,:,2], c='b', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
