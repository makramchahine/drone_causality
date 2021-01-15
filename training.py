import argparse
import os
import pickle
import random
import time

import numpy as np

from tensorflow import keras
import kerasncp as kncp


parser = argparse.ArgumentParser(description='Train the model on deepdrone data')
parser.add_argument('--model', type=str, default="ncp", help='The type of model (ncp, lstm)')
parser.add_argument('--data_dir', type=str, default="./data", help='Path to training data')
parser.add_argument('--save_dir', type=str, default="./model-checkpoints", help='Path to save checkpoints')
parser.add_argument('--history_dir', type=str, default="./histories", help='Path to save history')
parser.add_argument('--samples', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seq_len', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--hotstart', type=str, default=None, help="Starting weights to use for pretraining")
args = parser.parse_args()


IMAGE_SHAPE                = (256, 256, 3)
POSITION_SHAPE             = (3,)

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
        X = np.empty((self.batch_size, args.seq_len, *self.xDims))
        Y = np.empty((self.batch_size, args.seq_len, *self.yDims))

        for i, directory in enumerate(directories):
            try:
                X[i,] = np.load(os.path.join(args.data_dir, directory, 'images.npy'))
                Y[i,] = np.load(os.path.join(args.data_dir, directory, 'vectors.npy'))
            except Exception as e:
                print("Failed on directory: ", directory)
                raise e

        return X, Y

# Partition data into training and validation sets

paritions = dict()

sampleDirectories = list(os.listdir(args.data_dir))[:args.samples] # TODO(cvorbach) remove me
sampleDirectories = [d for d in sampleDirectories if os.path.isfile(os.path.join(args.data_dir, d, "vectors.npy"))]
random.shuffle(sampleDirectories)

k = int(args.val_split * len(sampleDirectories))
paritions['valid'] = sampleDirectories[:k]
paritions['train'] = sampleDirectories[k:]

print('Training:   ', paritions['train'])
print('Validation: ', paritions['valid'])

trainData = DataGenerator(paritions['train'], min(args.batch_size, len(paritions['train'])), IMAGE_SHAPE, POSITION_SHAPE)
validData = DataGenerator(paritions['valid'], min(args.batch_size, len(paritions['valid'])), IMAGE_SHAPE, POSITION_SHAPE)

if len(sampleDirectories) == 0:
    raise ValueError("No samples in " + args.data_dir)

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
ncpModel.add(keras.Input(shape=(args.seq_len, *IMAGE_SHAPE)))
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
if args.model == "lstm":
    trainingModel = lstmModel
elif args.model == "ncp":
    trainingModel = ncpModel
else:
    raise ValueError(f"Unsupported model type: {args.model}")


trainingModel.compile(
    optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
)

# Load weights
if args.hotstart is not None:
    trainingModel.load_weights(args.hotstart)

trainingModel.summary(line_length=80)

# Train
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.save_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + '-weights.{epoch:03d}-{val_loss:.4f}.hdf5'),
    save_weights_only=True,
    save_best_only=True,
    save_freq='epoch'
)

try:
    h = trainingModel.fit(
        x                   = trainData,
        validation_data     = validData,
        epochs              = args.epochs,
        use_multiprocessing = False,
        workers             = 1,
        max_queue_size      = 5,
        verbose             = 1,
        callbacks           = [checkpointCallback]
    )
finally:
    # Dump history
    with open(os.path.join(args.history_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + '-history.p'), 'wb') as fp:
        pickle.dump(trainingModel.history.history, fp)
