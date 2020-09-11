import os
import pickle
import random
from datetime import datetime

import numpy as np

from tensorflow import keras
import kerasncp as kncp

TRAINING_DATA_DIRECTORY    = os.getcwd() + '/data/'
MODEL_CHECKPOINT_DIRECTORY = os.getcwd() + '/model-checkpoints/'
SAMPLES                    = 10
BATCH_SIZE                 = 1  # TODO(cvorbach) fix me
EPOCHS                     = 100
TRAINING_SEQUENCE_LENGTH   = 64
IMAGE_SHAPE                = (256, 256, 3)
POSITION_SHAPE             = (3,)
VALIDATION_PROPORTION      = 0.2 # TODO(cvorbach) fix me

# Utilities

class DataGenerator(keras.utils.Sequence):
    def __init__(self, runDirectories, batch_size, xDims, yDims):
        self.runDirectories = runDirectories
        self.batch_size     = batch_size
        self.xDims          = xDims
        self.yDims          = yDims

        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        return int(len(self.runDirectories) / self.batch_size)

    def on_epoch_end(self):
        'Shuffle indexes to randomize batches each epoch'
        self.indexes = np.arange(len(self.runDirectories))
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'

        # data runs in this batch
        directories = self.runDirectories[index*self.batch_size:(index+1)*self.batch_size]

        # load data
        X, Y = self.__load_data(directories)

        return X, Y

    def __load_data(self, directories):
        X = np.empty((self.batch_size, TRAINING_SEQUENCE_LENGTH, *self.xDims))
        Y = np.empty((self.batch_size, TRAINING_SEQUENCE_LENGTH, *self.yDims))

        for i, directory in enumerate(directories):
            try:
                X[i,] = np.load(TRAINING_DATA_DIRECTORY + directory + '/images.npy')
                Y[i,] = np.load(TRAINING_DATA_DIRECTORY + directory + '/positions.npy')
            except Exception as e:
                print("Failed on directory: ", directory)
                raise e

        return X, Y

# Partition data into training and validation sets

paritions = dict()

sampleDirectories = list(os.listdir(TRAINING_DATA_DIRECTORY))[:SAMPLES] # TODO(cvorbach) remove me
random.shuffle(sampleDirectories)

k = int(VALIDATION_PROPORTION * len(sampleDirectories))
paritions['valid'] = sampleDirectories[:k]
paritions['train'] = sampleDirectories[k:]

trainData = DataGenerator(paritions['train'], BATCH_SIZE, IMAGE_SHAPE, POSITION_SHAPE)
validData = DataGenerator(paritions['valid'], BATCH_SIZE, IMAGE_SHAPE, POSITION_SHAPE)

if len(sampleDirectories) == 0:
    raise ValueError("No samples in " + TRAINING_DATA_DIRECTORY)

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
model.add(keras.Input(shape=(TRAINING_SEQUENCE_LENGTH, *IMAGE_SHAPE)))
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
    optimizer=keras.optimizers.Adam(0.01), loss="cosine_similarity",
)

model.summary(line_length=80)

# Train
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_CHECKPOINT_DIRECTORY+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    save_weights_only=True,
    save_best_only=True,
    save_freq='epoch'
)

model.fit(
    x                   = trainData,
    validation_data     = validData,
    epochs              = EPOCHS,
    use_multiprocessing = False,
    workers             = 1,
    verbose             = 1,
    callbacks           = [checkpointCallback]
)

# Dump history
with open(f'histories/{datetime.now()}-history.p', 'wb') as fp:
    pickle.dump(model.history.history, fp)

# Dump model
model.save(f'models/{datetime.now()}-model.p')
