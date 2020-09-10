import numpy as np
import os
# from tensorflow import keras
# import kerasncp as kncp
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import PIL.Image
from tensorflow import keras
import kerasncp as kncp
from datetime import datetime

TRAINING_DATA_DIRECTORY    = os.getcwd() + '/data'
MODEL_CHECKPOINT_DIRECTORY = os.getcwd() + '/model-checkpoints'
TRAINING_MAX_SAMPLES       = 32
TRAINING_SEQUENCE_LENGTH   = 32
IMAGE_WIDTH                = 256
IMAGE_HEIGHT               = 256
CHANNELS                   = 3


# Load the data from file
# TODO(cvorbach) do this data processing in other script
# TODO(cvorbach) stream data from file with generator
# TODO(cvorbach) validation set
trainingSamples = min(len(os.listdir(TRAINING_DATA_DIRECTORY)), TRAINING_MAX_SAMPLES)
print(f"{trainingSamples} Training Samples")

xTrain = np.zeros((trainingSamples, TRAINING_SEQUENCE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
yTrain = np.zeros((trainingSamples, TRAINING_SEQUENCE_LENGTH, 3))

for i, runDirectory in enumerate(os.listdir(TRAINING_DATA_DIRECTORY)):
    # stop loading data if we have enough samples
    if i >= TRAINING_MAX_SAMPLES:
        break

    imageDirectory = TRAINING_DATA_DIRECTORY + '/' + runDirectory + '/images'
    odometryFile   = TRAINING_DATA_DIRECTORY + '/' + runDirectory + '/airsim_rec.npy'

    for j, imageFile in enumerate(os.listdir(imageDirectory)):
        if j >= TRAINING_SEQUENCE_LENGTH:
            break
        xTrain[i][j] = np.load(imageDirectory + '/' + imageFile)

    odometry = np.load(odometryFile)
    for j, record in enumerate(odometry[1:]): # image[i] predicts position[i+1]
        if j >= TRAINING_SEQUENCE_LENGTH:
            break
        yTrain[i][j-1] = np.array([record['x'], record['y'], record['z']])


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
model.add(keras.Input(shape=xTrain.shape[1:]))
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
    optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error",
)

model.summary(line_length=80)

# Train
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_CHECKPOINT_DIRECTORY,
    save_weights_only=True,
    save_freq=5
)

model.fit(
    xTrain,
    yTrain,
    batch_size      = 8,
    epochs          = 200,
    # validation_data = (xValid, yValid),
    callbacks       = [checkpointCallback]
)

# # Offload 

# # Visualize?
