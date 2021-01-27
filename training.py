import argparse
import os
import pickle
import random
import time
import pathlib
from operator import mul
from functools import reduce
import sys

import numpy as np

from tensorflow import keras
import kerasncp as kncp
from node_cell import *

parser = argparse.ArgumentParser(description='Train the model on deepdrone data')
parser.add_argument('--model', type=str, default="ncp", help='The type of model (ncp, lstm, cnn, odernn, rnn, gru, ctgru)')
# Revisiont 4: rnn_size from 64 to 32
parser.add_argument('--rnn_size', type=int, default=32, help='Select the size of RNN network you would like to train')
parser.add_argument('--data_dir', type=str, default="./data", help='Path to training data')
parser.add_argument('--save_dir', type=str, default="./model-checkpoints", help='Path to save checkpoints')
parser.add_argument('--history_dir', type=str, default="./histories", help='Path to save history')
parser.add_argument('--samples', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--seq_len', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--hotstart', type=str, default=None, help="Starting weights to use for pretraining")
parser.add_argument("--gps_signal", dest="gps_signal", action="store_true")
parser.add_argument('--cnn_units', type=int, default=1000)
parser.set_defaults(gps_signal=False)
args = parser.parse_args()


IMAGE_SHAPE                = (256, 256, 3)
POSITION_SHAPE             = (3,)

print("Creating logging directories, if they dont exist")
for dir in [args.data_dir, args.save_dir, args.history_dir]:
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

# Utilities
class DataGenerator(keras.utils.Sequence):
    def __init__(self, runDirectories, batch_size, xDims, yDims, ):
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

class GPSDataGenerator(keras.utils.Sequence):
    def __init__(self, runDirectories, batch_size, xDims, yDims, ):
        self.runDirectories = runDirectories
        self.batch_size     = batch_size
        self.xDims          = xDims
        self.yDims          = yDims
        self.gpsVectorShape = 3

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
        X1, X2, Y = self.__load_data(directories)

        return [X1, X2], Y

    def __load_data(self, directories):
        X1 = np.empty((self.batch_size, args.seq_len, *self.xDims))
        X2 = np.empty((self.batch_size, args.seq_len, self.gpsVectorShape))
        Y = np.empty((self.batch_size, args.seq_len, *self.yDims))

        for i, directory in enumerate(directories):
            try:
                X1[i,] = np.load(os.path.join(args.data_dir, directory, 'images.npy'))
                X2[i,] = np.load(os.path.join(args.data_dir, directory, 'gps.npy'))
                Y[i,] = np.load(os.path.join(args.data_dir, directory, 'vectors.npy'))
            except Exception as e:
                print("Failed on directory: ", directory)
                raise e

        # Correction for functional API
        if args.model == "ncp":
            Y = Y[:,-1,:]

        return X1, X2, Y

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

if not args.gps_signal:
    trainData = DataGenerator(paritions['train'], min(args.batch_size, len(paritions['train'])), IMAGE_SHAPE, POSITION_SHAPE)
    validData = DataGenerator(paritions['valid'], min(args.batch_size, len(paritions['valid'])), IMAGE_SHAPE, POSITION_SHAPE)
else:
    trainData = GPSDataGenerator(paritions['train'], min(args.batch_size, len(paritions['train'])), IMAGE_SHAPE, POSITION_SHAPE)
    validData = GPSDataGenerator(paritions['valid'], min(args.batch_size, len(paritions['valid'])), IMAGE_SHAPE, POSITION_SHAPE)

if len(sampleDirectories) == 0:
    raise ValueError("No samples in " + args.data_dir)

# Setup the network
# Revision 2: 8 to 16 command neurons
# Revision 3: 16 to 32 command neurons
wiring = kncp.wirings.NCP(
    inter_neurons=16,   # Number of inter neurons
    command_neurons=8,  # Number of command neurons
    motor_neurons=3,    # Number of motor neurons
    sensory_fanout=8,   # How many outgoing synapses has each sensory neuron
    inter_fanout=4,     # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=4,   # Now many recurrent synapses are in the
                                    # command neuron layer
    motor_fanin=6,      # How many incomming syanpses has each motor neuron
)

rnnCell = kncp.LTCCell(wiring)

# Revision 5: quarter the number of cnn parameters
ncpModel = keras.models.Sequential()
ncpModel.add(keras.Input(shape=(args.seq_len, *IMAGE_SHAPE)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=24,   activation='linear')))
ncpModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

# NCP network with multiple input (Requires the Functional API)
imageInput        = ncpModel.layers[0].input
penultimateOutput = ncpModel.layers[-2].output
imageFeatures     = keras.layers.Dense(units=12, activation="linear")(penultimateOutput)

gpsInput    = keras.Input(batch_size = min(args.batch_size, len(paritions["train"])), shape = (args.seq_len, 3))
gpsFeatures = keras.layers.Dense(units=12, activation='linear')(gpsInput)

multiFeatures = keras.layers.concatenate([imageFeatures, gpsFeatures])

rnn, state = keras.layers.RNN(rnnCell, return_state=True)(multiFeatures)
npcMultiModel = keras.models.Model(inputs=[imageInput, gpsInput], outputs = [rnn])

# LSTM network
penultimateOutput = ncpModel.layers[-2].output
lstmOutput        = keras.layers.LSTM(units=args.rnn_size, return_sequences=True)(penultimateOutput)
lstmOutput        = keras.layers.Dense(units=3, activation='linear')(lstmOutput)
lstmModel = keras.models.Model(ncpModel.input, lstmOutput)

# LSTM multiple input network
lstmMultiOutput        = keras.layers.LSTM(units=args.rnn_size, return_sequences=True)(multiFeatures)
lstmMultiOutput        = keras.layers.Dense(units=3, activation='linear')(lstmMultiOutput)
lstmMultiModel = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[lstmMultiOutput])

# for x in trainData:
#     (x1, x2), y = x
#     print(x1.shape)
#     print(x2.shape)
#     print(y.shape)
#     print(lstmMultiOutput.shape)
#     sys.exit()
    

# Vanilla RNN network
penultimateOutput = ncpModel.layers[-2].output
rnnOutput         = keras.layers.SimpleRNN(units=args.rnn_size, return_sequences=True)(penultimateOutput)
rnnOutput         = keras.layers.Dense(units=3, activation='linear')(rnnOutput)
rnnModel          = keras.models.Model(ncpModel.input, rnnOutput)

# Vanilla RNN multiple input network
rnnMultiOutput = keras.layers.SimpleRNN(units=args.rnn_size, return_sequences=True)(multiFeatures)
rnnMultiOutput = keras.layers.Dense(units=3, activation='linear')(rnnMultiOutput)
rnnMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[rnnMultiOutput])

# GRU network
penultimateOutput = ncpModel.layers[-2].output
gruOutput        = keras.layers.GRU(units=args.rnn_size, return_sequences=True)(penultimateOutput)
gruOutput        = keras.layers.Dense(units=3, activation='linear')(gruOutput)
gruModel = keras.models.Model(ncpModel.input, gruOutput)

# GRU multiple input network
gruMultiOutput = keras.layers.GRU(units=args.rnn_size, return_sequences=True)(multiFeatures)
gruMultiOutput = keras.layers.Dense(units=3, activation='linear')(gruMultiOutput)
gruMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[gruMultiOutput])

# # CT-GRU network
# penultimateOutput = ncpModel.layers[-2].output
# ctgruOutput        = CTGRU(units=64)(penultimateOutput)
# ctgruOutput        = keras.layers.Dense(units=3, activation='linear')(ctgruOutput)
# ctgruModel = keras.models.Model(ncpModel.input, ctgruOutput)

# CT-GRU multiple input network

# # ODE-RNN network
# penultimateOutput = ncpModel.layers[-2].output
# odernnOutput        = CTRNNCell(units=64, method='dopri5')(penultimateOutput)
# odernnOutput        = keras.layers.Dense(units=3, activation='linear')(odernnOutput)
# odernnModel = keras.models.Model(ncpModel.input, odernnOutput)

# ODE-RNN multiple input network

# CNN network
# Revision 2: 1000 and 100 units to 500 and 50 units
remove_ncp_layer = ncpModel.layers[-3].output
cnnOutput = keras.layers.TimeDistributed(keras.layers.Dense(units=250, activation='relu'))(remove_ncp_layer)
cnnOutput = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5))(cnnOutput)
cnnOutput = keras.layers.TimeDistributed(keras.layers.Dense(units=25, activation='relu'))(cnnOutput)
cnnOutput = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.3))(cnnOutput)
cnnOutput = keras.layers.Dense(units=3, activation='linear')(cnnOutput)
cnnModel  = keras.models.Model(ncpModel.input, cnnOutput)

# CNN multiple input network

# TODO(cvorbach) Not sure if this makes sense for a cnn?


# Configure the model we will train
if not args.gps_signal:
    if args.model == "lstm":
        trainingModel = lstmModel
    elif args.model == "ncp":
        trainingModel = ncpModel
    elif args.model == "cnn":
        trainingModel = cnnModel
    elif args.model == "odernn":
        trainingModel = odernnModel
    elif args.model == "gru":
        trainingModel = gruModel
    elif args.model == "rnn":
        trainingModel = rnnModel
    elif args.model == "ctgru":
        trainingModel = ctgruModel
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
else:
    if args.model == "lstm":
        trainingModel = lstmMultiModel
    elif args.model == "ncp":
        trainingModel = npcMultiModel
    elif args.model == "cnn":
        raise ValueError(f"Unsupported model type: {args.model}")
    elif args.model == "odernn":
        trainingModel = odernnMultiModel
    elif args.model == "gru":
        trainingModel = gruMultiModel
    elif args.model == "rnn":
        trainingModel = rnnMultiModel
    elif args.model == "ctgru":
        trainingModel = ctgruMultiModel
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

trainingModel.compile(
    optimizer=keras.optimizers.Adam(0.0005), loss="cosine_similarity",
)

# Load weights
if args.hotstart is not None:
    trainingModel.load_weights(args.hotstart)

trainingModel.summary(line_length=80)

# Train
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.save_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + "rev-6.0" + '-weights.{epoch:03d}-{val_loss:.4f}.hdf5'),
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
    with open(os.path.join(args.history_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + f'-history-rev-{6.0}.p'), 'wb') as fp:
        pickle.dump(trainingModel.history.history, fp)
