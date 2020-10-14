import os
import pickle
import random
import time

import numpy as np

from tensorflow import keras
import kerasncp as kncp

TRAIN_LSTM                 = False
TRAINING_DATA_DIRECTORY    = os.getcwd() + '/data/'

if TRAIN_LSTM:
    MODEL_NAME             = 'lstm'
else:
    MODEL_NAME             = 'ncp'
MODEL_CHECKPOINT_DIRECTORY = os.getcwd() + '/model-checkpoints/'
SAMPLES                    = -1
BATCH_SIZE                 = 8  
EPOCHS                     = 50
TRAINING_SEQUENCE_LENGTH   = 32
IMAGE_SHAPE                = (256, 256, 3)
POSITION_SHAPE             = (3,)
VALIDATION_PROPORTION      = 0.1 

STARTING_WEIGHTS           = 'model-checkpoints/weights.007--0.9380.hdf5'

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
                Y[i,] = np.load(TRAINING_DATA_DIRECTORY + directory + '/vectors.npy')
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

print('Training:   ', paritions['train'])
print('Validation: ', paritions['valid'])

trainData = DataGenerator(paritions['train'], min(BATCH_SIZE, len(paritions['train'])), IMAGE_SHAPE, POSITION_SHAPE)
validData = DataGenerator(paritions['valid'], min(BATCH_SIZE, len(paritions['valid'])), IMAGE_SHAPE, POSITION_SHAPE)

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

ncpModel = keras.models.Sequential()
ncpModel.add(keras.Input(shape=(TRAINING_SEQUENCE_LENGTH, *IMAGE_SHAPE)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1000, activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=100,  activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.3)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=24,   activation='relu')))
ncpModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

# LSTM network
penultimateOutput = ncpModel.layers[-2].output
lstmOutput        = keras.layers.SimpleRNN(units=3, return_sequences=True, activation='relu')(penultimateOutput)
lstmModel = keras.models.Model(ncpModel.input, lstmOutput)

# Configure the model we will train
if TRAIN_LSTM:
    trainingModel = lstmModel
else:
    trainingModel = ncpModel

trainingModel.compile(
    optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
)

# Load weights
if STARTING_WEIGHTS is not None:
    trainingModel.load_weights(STARTING_WEIGHTS)

trainingModel.summary(line_length=80)

# Train
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_CHECKPOINT_DIRECTORY + '/' + MODEL_NAME + f'-{time.strftime("%Y:%m:%d:%H:%M:%S")}' + '-weights.{epoch:03d}-{val_loss:.4f}.hdf5',
    save_weights_only=True,
    save_best_only=True,
    save_freq='epoch'
)

try: 
    h = trainingModel.fit(
        x                   = trainData,
        validation_data     = validData,
        epochs              = EPOCHS,
        use_multiprocessing = False,
        workers             = 1,
        max_queue_size      = 5,
        verbose             = 1,
        callbacks           = [checkpointCallback]
    )
finally:
    # Dump history
    with open(f'histories/{MODEL_NAME}-' + time.strftime("%Y:%m:%d:%H:%M:%S") + '-history.p', 'wb') as fp:
        pickle.dump(trainingModel.history.history, fp)
