import os
import pickle
import random
import time

import numpy as np

from tensorflow import keras
import kerasncp as kncp

TRAIN_LSTM                 = True
TRAINING_DATA_DIRECTORY    = 'C:\\Users\MIT Driverless\\Documents\\AirSim\\target-redwood-parsed\\'

if TRAIN_LSTM:
    MODEL_NAME             = 'lstm'
else:
    MODEL_NAME             = 'new-ncp'
MODEL_CHECKPOINT_DIRECTORY = os.getcwd() + '/model-checkpoints/'
SAMPLES                    = -1
BATCH_SIZE                 = 8  
EPOCHS                     = 10
TRAINING_SEQUENCE_LENGTH   = 32
IMAGE_SHAPE                = (256, 256, 3)
POSITION_SHAPE             = (3,)
VALIDATION_PROPORTION      = 0.1 

STARTING_WEIGHTS           = 'model-checkpoints/new-ncp-2020_11_12_23_55_01-weights.010--0.8645.hdf5'

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
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=24, activation='linear')))
ncpModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

ncpModel.load_weights(STARTING_WEIGHTS)

# LSTM network
penultimateOutput = ncpModel.layers[-2].output
lstmLayer1        = keras.layers.LSTM(units=64, return_sequences=True, activation='relu')(penultimateOutput)
lstmLayer2        = keras.layers.Dense(units=3, activation='linear')(lstmLayer1)
lstmModel = keras.models.Model(ncpModel.input, lstmLayer2)


# Configure the model we will train
if TRAIN_LSTM:
    trainingModel = lstmModel
else:
    trainingModel = ncpModel

# Load weights
# if STARTING_WEIGHTS is not None:
#     trainingModel.load_weights(STARTING_WEIGHTS)

trainingModel.compile(
    optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
)

trainingModel.summary(line_length=80)

# Train
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_CHECKPOINT_DIRECTORY + '/' + MODEL_NAME + f'-{time.strftime("%Y_%m_%d_%H_%M_%S")}' + '-weights.{epoch:03d}-{val_loss:.4f}.hdf5',
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
    with open(f'histories/{MODEL_NAME}-' + time.strftime("%Y_%m_%d_%H_%M_%S") + '-history.p', 'wb') as fp:
        pickle.dump(trainingModel.history.history, fp)
