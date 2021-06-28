#!/usr/bin/python3
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
from kerasncp.tf import LTCCell
from node_cell import *

import matplotlib.pyplot as plt

MODEL_REVISION_LABEL = 13.0

parser = argparse.ArgumentParser(description='Train the model on deepdrone data')
parser.add_argument('--model', type=str, default="ncp", help='The type of model (ncp, lstm, cnn, odernn, rnn, gru, ctgru)')
# Revisiont 4: rnn_size from 64 to 32
parser.add_argument('--rnn_size', type=int, default=32, help='Select the size of RNN network you would like to train')
parser.add_argument('--data_dir', type=str, default="./data", help='Path to training data')
parser.add_argument('--save_dir', type=str, default="./model-checkpoints", help='Path to save checkpoints')
parser.add_argument('--history_dir', type=str, default="./histories", help='Path to save history')
parser.add_argument('--samples', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--hotstart', type=str, default=None, help="Starting weights to use for pretraining")
parser.add_argument("--gps_signal", dest="gps_signal", action="store_true")
parser.add_argument('--cnn_units', type=int, default=1000)
parser.add_argument('--infer_only', action='store_true', help='Use to run inference only. Must use with --hotstart')
parser.add_argument('--plot_dir', type=str, default='none', help="Name of directory for inference plots")
parser.add_argument('--tb_dir', type=str, default='tb_logs', help="Name of directory to save tensorboard logs")
parser.add_argument('--lr', type=float, default='.001', help="Learning Rate")
parser.add_argument('--opt', type=str, default='adam', help="Optimizer to use (adam, sgd)")
parser.add_argument('--augment', type=bool, action='store_true', help="Whether to turn on data augmentation in network")
parser.set_defaults(gps_signal=False)
args = parser.parse_args()

IMAGE_SHAPE                = (256, 256, 3)
#POSITION_SHAPE             = (3,)
POSITION_SHAPE             = (4,)

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
                img = np.load(os.path.join(args.data_dir, directory, 'images.npy')).astype(np.float32)
                #img = np.load(os.path.join(args.data_dir, directory, 'images.npy'))
                img = ((img / 255.) - 0.5) / 0.03
                X[i,] = img
                Y[i,] = np.load(os.path.join(args.data_dir, directory, 'vectors.npy')) * 5
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

#if len(sampleDirectories) == 0:
#    raise ValueError("No samples in " + args.data_dir)

# Setup the network
wiring = kncp.wirings.NCP(
    inter_neurons=18,   # Number of inter neurons
    command_neurons=12,  # Number of command neurons
    #motor_neurons=3,    # Number of motor neurons
    motor_neurons=4,    # Number of motor neurons
    sensory_fanout=6,   # How many outgoing synapses has each sensory neuron
    inter_fanout=4,     # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=4,   # Now many recurrent synapses are in the
                                    # command neuron layer
    motor_fanin=6,      # How many incoming syanpses has each motor neuron
)

#rnnCell = kncp.LTCCell(wiring)
rnnCell = LTCCell(wiring)

ncpModel = keras.models.Sequential()
ncpModel.add(keras.Input(shape=(args.seq_len, *IMAGE_SHAPE)))

if args.model_normalization:
    ncpModel.add(keras.layers.experimental.preprocessing.Rescaling(1./255))
    normalization_layer = keras.layers.experimental.preprocessing.Normalization()
    normalization_layer.adapt(DATA) # DATA is all the data after loading into RAM (single array)
    ncpModel.add(normalization_layer)

if args.augment:
    # translate -> rotate -> zoom
    ncpModel.add(keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, weidth_factor=0.1))
    ncpModel.add(keras.layers.experimental.preprocessing.RandomRotation(0.05))
    ncpModel.add(keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.1, weidth_factor=0.1))

ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu'))) # added
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=128, kernel_size=(2,2), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
#kprint(tf.shape(keras.layers.TimeDistributed(keras.layers.Flatten())))

ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
#ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64,   activation='relu')))
#ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64,   activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64,   activation='linear')))
ncpModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

# NCP network with multiple input (Requires the Functional API)
imageInput        = ncpModel.layers[0].input
penultimateOutput = ncpModel.layers[-2].output
imageFeatures     = keras.layers.Dense(units=48, activation="linear")(penultimateOutput)

gpsInput    = keras.Input(batch_size = min(args.batch_size, len(paritions["train"])), shape = (args.seq_len, 3))
gpsFeatures = keras.layers.Dense(units=16, activation='linear')(gpsInput)

multiFeatures = keras.layers.concatenate([imageFeatures, gpsFeatures])

rnn, state = keras.layers.RNN(rnnCell, return_state=True)(multiFeatures)
npcMultiModel = keras.models.Model(inputs=[imageInput, gpsInput], outputs = [rnn])

# LSTM network
penultimateOutput = ncpModel.layers[-2].output
lstmOutput        = keras.layers.LSTM(units=args.rnn_size, return_sequences=True)(penultimateOutput)
lstmOutput        = keras.layers.Dense(units=4, activation='linear')(lstmOutput)
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
gruOutput         = keras.layers.GRU(units=args.rnn_size, return_sequences=True)(penultimateOutput)
gruOutput         = keras.layers.Dense(units=3, activation='linear')(gruOutput)
gruModel          = keras.models.Model(ncpModel.input, gruOutput)

# GRU multiple input network
gruMultiOutput = keras.layers.GRU(units=args.rnn_size, return_sequences=True)(multiFeatures)
gruMultiOutput = keras.layers.Dense(units=3, activation='linear')(gruMultiOutput)
gruMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[gruMultiOutput])

# CT-GRU network
penultimateOutput  = ncpModel.layers[-2].output
ctgruCell          = CTGRU(units=args.rnn_size)
ctgruOutput        = keras.layers.RNN(ctgruCell, return_sequences=True)(penultimateOutput)
ctgruOutput        = keras.layers.Dense(units=3, activation='linear')(ctgruOutput)
ctgruModel         = keras.models.Model(ncpModel.input, ctgruOutput)

# CT-GRU multiple input network
ctgruMultiCell   = CTGRU(units=args.rnn_size)
ctgruMultiOutput = keras.layers.RNN(ctgruMultiCell, return_sequences=True)(multiFeatures)
ctgruMultiOutput = keras.layers.Dense(units=3, activation="linear")(ctgruMultiOutput)
ctgruMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[ctgruMultiOutput])

# ODE-RNN network
penultimateOutput = ncpModel.layers[-2].output
odernnCell        = CTRNNCell(units=args.rnn_size, method='dopri5')
odernnOutput      = keras.layers.RNN(odernnCell, return_sequences=True)(penultimateOutput)
odernnOutput      = keras.layers.Dense(units=3, activation='linear')(odernnOutput)
odernnModel       = keras.models.Model(ncpModel.input, odernnOutput)

# ODE-RNN multiple input network
odernnMultiCell   = CTRNNCell(units=args.rnn_size, method='dopri5')
odernnMultiOutput = keras.layers.RNN(odernnMultiCell, return_sequences=True)(multiFeatures)
odernnMultiOutput = keras.layers.Dense(units=3, activation='linear')(odernnMultiOutput)
odernnMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[odernnMultiOutput])

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

if args.opt == 'adam':
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
elif args.opt == 'sgd':
    optimizer = keras.optimizers.SGD(learning_rate=args.lr)
else:
    raise Exception('Unsupported optimizer type %s' % args.opt)

trainingModel.compile(optimizer=optimizer, loss="mean_squared_error")

# Load weights
if args.hotstart is not None:
    trainingModel.load_weights(args.hotstart)

trainingModel.summary(line_length=80)

if not args.infer_only:
    # Train
    checkpointCallback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + f"-rev={MODEL_REVISION_LABEL}" + '-weights.{epoch:03d}-{val_loss:.4f}.hdf5'),
        save_weights_only=True,
        save_best_only=False,
        save_freq='epoch'
    )

    log_dir = args.tb_dir
    if not os.path.exists(log_dir):
        #os.mkdir(log_dir)
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    from video_frame_generator import VideoFrameGenerator
    #train_datagen = VideoFrameGenerator(rescale=1./255, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, brightness_range=(-.1,.1), featurewise_center=True, featurewise_std_normalization=True, label_scale=5)
    train_datagen = VideoFrameGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1, featurewise_center=True, featurewise_std_normalization=True, label_scale=5)
    #train_datagen = VideoFrameGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True, label_scale=5)
    train_datagen.mean = 0.5
    train_datagen.std = 0.03
    train_generator = train_datagen.flow_from_directory(os.path.join(args.data_dir,'training'), target_size=(256, 256), batch_size=16, frames_per_step=64, class_mode='npy')

    test_datagen = VideoFrameGenerator(rescale=1./255, featurewise_center=True, featurewise_std_normalization=True, label_scale=5)
    test_datagen.mean = 0.5
    test_datagen.std = 0.03
    validation_generator = test_datagen.flow_from_directory(os.path.join(args.data_dir,'validation'), target_size=(256,256), batch_size=16, frames_per_step=64, class_mode='npy')
    try:
        h = trainingModel.fit(
            #x                   = trainData,
            x                   = train_generator,
            #validation_data     = validData,
            validation_data     = validation_generator,
            epochs              = args.epochs,
            use_multiprocessing = True,
            workers             = 16,
            max_queue_size      = 20,
            verbose             = 1,
            callbacks           = [checkpointCallback, tensorboard_callback]
        )
    finally:
        # Dump history
        with open(os.path.join(args.history_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + f'-history-rev={MODEL_REVISION_LABEL}.p'), 'wb') as fp:
            pickle.dump(trainingModel.history.history, fp)

else:

    dirs = list(os.listdir(args.data_dir))
    for d in dirs:
        img = np.load(os.path.join(args.data_dir, d, 'images.npy'))
        img = ((img / 255.) - 0.5) / 0.03
        
        prediction = trainingModel.predict(np.array([img]))
        gt = np.load(os.path.join(args.data_dir, d, 'vectors.npy'))

        plt.figure()
        plt.plot(prediction[0,:,0])
        plt.plot(5*gt[:,0])
        plt.title('%s, x' % d)
        plt.legend(['prediction', 'actual'])
        plt.ylim([-5, 5])
        plt.savefig(os.path.join(args.plot_dir, 'vx' + d + '.png'))

        plt.figure()
        plt.plot(prediction[0,:,1])
        plt.plot(5*gt[:,1])
        plt.title('%s, y' % d)
        plt.legend(['prediction', 'actual'])
        plt.ylim([-5, 5])
        plt.savefig(os.path.join(args.plot_dir, 'vy' + d + '.png'))

        plt.figure()
        plt.plot(prediction[0,:,2])
        plt.plot(5*gt[:,2])
        plt.title('%s, z' % d)
        plt.legend(['prediction', 'actual'])
        plt.ylim([-5, 5])
        plt.savefig(os.path.join(args.plot_dir, 'vz' + d + '.png'))

        plt.figure()
        plt.plot(prediction[0,:,3])
        plt.plot(5*gt[:,3])
        plt.title('%s, yawrate' % d)
        plt.legend(['prediction', 'actual'])
        plt.ylim([-5, 5])
        plt.savefig(os.path.join(args.plot_dir, 'yawrate' + d + '.png'))

        plt.close('all')

