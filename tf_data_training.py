#!/usr/bin/python3
import argparse
import os
import pickle
import random
import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
#from node_cell import *


from tf_data_loader import load_dataset
from keras_models import generate_ncp_model, generate_lstm_model

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
parser.add_argument('--cnn_units', type=int, default=1000)
parser.add_argument('--tb_dir', type=str, default='tb_logs', help="Name of directory to save tensorboard logs")
parser.add_argument('--lr', type=float, default='.001', help="Learning Rate")
parser.add_argument('--momentum', type=float, default='0.0', help="Momentum (for use with SGD)")
parser.add_argument('--opt', type=str, default='adam', help="Optimizer to use (adam, sgd)")
parser.add_argument('--augment', action='store_true', help="Whether to turn on data augmentation in network")
parser.add_argument('--normalize', action='store_true', help="Whether to have float conversion and normalization inside network layers")
parser.add_argument('--label_scale', type=float, default=1, help='Scale factor to apply to labels')
parser.add_argument('--translation_factor', type=float, default=0.1, help='Amount to (randomly) translate width and height (0 - 1.0). Must be used with --augment.')
parser.add_argument('--rotation_factor', type=float, default=0.1, help='Amount to (randomly) rotate (0.0 - 1.0). Must be used with --augment.')
parser.add_argument('--zoom_factor', type=float, default=0.1, help='Amount to (randomly) zoom. Must be used with --augment.')

parser.set_defaults(gps_signal=False)
args = parser.parse_args()


augmentation_params = {}
augmentation_params['translation'] = args.translation_factor
augmentation_params['rotation'] = args.rotation_factor
augmentation_params['zoom'] = args.zoom_factor

IMAGE_SHAPE                = (256, 256, 3)
POSITION_SHAPE             = (4,)


training_np, validation_np = load_dataset(args.data_dir, args.label_scale)

print('============================')
print('Training Input Shape: ', training_np[0].shape)
print('Training Labels Shape: ', training_np[1].shape)
print('Training Input Size in RAM: ' + str(training_np[0].size * training_np[0].itemsize / 1e9) + ' GB')
print('----------------------------')
print('Validation Input Shape: ', validation_np[0].shape)
print('Validation Labels Shape: ', validation_np[1].shape)
print('Validation Input Size in RAM: ' + str(validation_np[0].size * validation_np[0].itemsize / 1e9) + ' GB')

training_dataset = tf.data.Dataset.from_tensor_slices(training_np).shuffle(100).batch(args.batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices(validation_np).batch(args.batch_size)


if args.model == 'ncp':
    model = generate_ncp_model(args.seq_len, IMAGE_SHAPE, args.normalize, args.augment, training_np[0], augmentation_params)
elif args.model == 'lstm':
    model = generate_lstm_model(args.rnn_size, args.seq_len, IMAGE_SHAPE, args.normalize, args.augment, training_np[0], augmentation_params)
else:
    raise Exception('Unsupported model type: %s' % args.model)


if args.opt == 'adam':
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
elif args.opt == 'sgd':
    optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)
else:
    raise Exception('Unsupported optimizer type %s' % args.opt)

model.compile(optimizer=optimizer, loss="mean_squared_error")

# Load weights
if args.hotstart is not None:
    model.load_weights(args.hotstart)

model.summary(line_length=80)

# Train
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.save_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + f"-rev={MODEL_REVISION_LABEL}" + '-weights.{epoch:03d}-{val_loss:.4f}.hdf5'),
    save_weights_only=True,
    save_best_only=False,
    #save_freq='epoch'
)

log_dir = args.tb_dir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10, 15')

try:
    h = model.fit(
        x                   = training_dataset,
        validation_data     = validation_dataset,
        epochs              = args.epochs,
        use_multiprocessing = False,
        workers             = 1,
        max_queue_size      = 5,
        verbose             = 1,
        callbacks           = [checkpointCallback, tensorboard_callback]
    )
finally:
    # Dump history
    with open(os.path.join(args.history_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + f'-history-rev={MODEL_REVISION_LABEL}.p'), 'wb') as fp:
        pickle.dump(model.history.history, fp)

