import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp

import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import os
import datetime
# from mxnet.gluon import nn
# from mxnet import np, npx, init

TRAIN_LSTM = False
MODEL_RECORDING_DIRECTORY = "C:\\Users\\MIT Driverless\\Documents\\deepdrone\\model-piloted-runs\\2020-11-04-21-44-59\\images"
IMAGE_OUTPUT_DIRECTORY    = "C:\\Users\\MIT Driverless\\Documents\\deepdrone\\image_output"
WEIGHTS_PATH              = 'C:\\Users\\MIT Driverless\\Documents\\deepdrone\\model-checkpoints\\new-ncp-2020_10_29_16_56_05-weights.022--0.8495.hdf5'

# Setup the network
SEQUENCE_LENGTH = 32
IMAGE_SHAPE     = (256,256,3)

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

kernels = [
    (5,5),
    (5,5),
    (3,3),
    (3,3),
    (3,3),
]

strides = [
    (2,2),
    (2,2),
    (2,2),
    (1,1),
    (1,1),
]

inputs = keras.Input(shape=IMAGE_SHAPE)
x = keras.layers.Conv2D(filters=24, kernel_size=kernels[0], strides=strides[0], activation='relu')(inputs)
x = keras.layers.Conv2D(filters=36, kernel_size=kernels[1], strides=strides[1], activation='relu')(x)
x = keras.layers.Conv2D(filters=48, kernel_size=kernels[2], strides=strides[2], activation='relu')(x)
x = keras.layers.Conv2D(filters=64, kernel_size=kernels[3], strides=strides[3], activation='relu')(x)
x = keras.layers.Conv2D(filters=64, kernel_size=kernels[4], strides=strides[4], activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(units=24, activation='linear')(x)
rnn = keras.layers.RNN(rnnCell, return_sequences=True, return_state=True)(x)

model = keras.models.Model(inputs=x, outputs = [rnn])

model.compile(
    optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
)

# Load weights
model.load_weights(WEIGHTS_PATH)

model.summary()

images = np.zeros((1, SEQUENCE_LENGTH, *IMAGE_SHAPE))
for i, imageFile in enumerate(os.listdir(MODEL_RECORDING_DIRECTORY)):
    print("Creating Image: ", i)
    try:
        image = np.array(PIL.Image.open(MODEL_RECORDING_DIRECTORY + '\\' + imageFile).convert('RGB'), dtype=np.float32) / 255
    except PIL.UnidentifiedImageError:
        print("Image: ", imageFile, " is corrupt.")

    # Correctly add image to sliding window
    if i < SEQUENCE_LENGTH:
        images[0, i] = image
    else:
        images[0] = np.roll(images[0], -1, axis=0)
        images[0, -1] = image

    # Visualization
    activations, next_state = model.predict(images)
