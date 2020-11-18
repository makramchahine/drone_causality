import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp

import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import os
import datetime
from functools import reduce
from operator import mul
import pickle

# from mxnet.gluon import nn
# from mxnet import np, npx, init
NEURON = 0

TRAIN_LSTM = False
MODEL_RECORDING_DIRECTORY = "C:\\Users\\MIT Driverless\\Documents\\deepdrone\\model-piloted-runs\\2020-11-04-21-44-59\\images"
IMAGE_OUTPUT_DIRECTORY    = "C:\\Users\\MIT Driverless\\Documents\\deepdrone\\image_output"
WEIGHTS_PATH              = 'C:\\Users\\MIT Driverless\\Documents\\deepdrone\\model-checkpoints\\new-ncp-2020_10_29_16_56_05-weights.022--0.8495.hdf5'

# Setup the network
BATCH_SIZE      = 1
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

inputs = keras.Input(batch_size = BATCH_SIZE, shape = (SEQUENCE_LENGTH, *IMAGE_SHAPE))
c1 = keras.layers.Conv2D(filters=24, kernel_size=kernels[0], strides=strides[0], activation='relu')(inputs)
c2 = keras.layers.Conv2D(filters=36, kernel_size=kernels[1], strides=strides[1], activation='relu')(c1)
c3 = keras.layers.Conv2D(filters=48, kernel_size=kernels[2], strides=strides[2], activation='relu')(c2)
c4 = keras.layers.Conv2D(filters=64, kernel_size=kernels[3], strides=strides[3], activation='relu')(c3)
c5 = keras.layers.Conv2D(filters=64, kernel_size=kernels[4], strides=strides[4], activation='relu')(c4)
f1 = keras.layers.Reshape((SEQUENCE_LENGTH, reduce(mul, c5.shape[2:])))(c5)
d1 = keras.layers.Dropout(rate=0.5)(f1)
den2 = keras.layers.Dense(units=24, activation='linear')(d1)

rnn, state = keras.layers.RNN(rnnCell, return_state=True)(den2)

model = keras.models.Model(inputs=inputs, outputs = [rnn, state])

model.compile(
    optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
)

# Load weights
model.load_weights(WEIGHTS_PATH)

model.summary()

# Load images
imageFiles = os.listdir(MODEL_RECORDING_DIRECTORY)
images = np.zeros((len(imageFiles), *IMAGE_SHAPE))
for i, imageFile in enumerate(imageFiles):
    try:
        image = np.array(PIL.Image.open(MODEL_RECORDING_DIRECTORY + '\\' + imageFile).convert('RGB'), dtype=np.float32) / 255
    except PIL.UnidentifiedImageError:
        raise Exception("Image: ", MODEL_RECORDING_DIRECTORY, "\\", imageFile, " is corrupt.")

    images[i] = image

batch = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, *IMAGE_SHAPE))
k = 0
vectors = []
activations = []
while (k + 1) * BATCH_SIZE * SEQUENCE_LENGTH < len(imageFiles):
    for i in range(BATCH_SIZE):
        for j in range(SEQUENCE_LENGTH):
            batch[i, j] = images[k * BATCH_SIZE * SEQUENCE_LENGTH + i * SEQUENCE_LENGTH + j]

            prediction, state = model.predict(batch)
            print(prediction.shape)
            vectors.append(prediction[0])
            activations.append(state[0, NEURON])
    k += 1

with open('vectors.p', 'wb') as f:
    pickle.dump(vectors, f)

with open('activations.p', 'wb') as f:
    pickle.dump(activations, f)