#!/usr/bin/python3
import argparse
import os
import pickle
import random
import pathlib
import time
import glob

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tf_data_loader import load_dataset, get_dataset_multi, get_output_normalization, load_dataset_rnn
from keras_models import generate_ncp_model, generate_lstm_model

def tlen(dataset):
    for (ix, _) in enumerate(dataset):
        pass
    return ix

MODEL_REVISION_LABEL = 13.0

parser = argparse.ArgumentParser(description='Train the model on deepdrone data')
parser.add_argument('--model', type=str, default="lstm", help='The type of model (ncp, lstm, cnn, odernn, rnn, gru, ctgru)')
parser.add_argument('--rnn_sizes', type=int, nargs='+', help='Select the size of RNN network you would like to train')
parser.add_argument('--data_dir', type=str, default="./data", help='Path to training data')
parser.add_argument('--cached_data_dir', type=str, default=None, help='Path to pre-cached dataset')
parser.add_argument('--extra_data_dir', type=str, default=None, help='Path to extra training data, used for training but not validation')
parser.add_argument('--save_dir', type=str, default="./model-checkpoints", help='Path to save checkpoints')
parser.add_argument('--history_dir', type=str, default="./histories", help='Path to save history')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--hotstart', type=str, default=None, help="Starting weights to use for pretraining")
parser.add_argument('--tb_dir', type=str, default='tb_logs', help="Name of directory to save tensorboard logs")
parser.add_argument('--lr', type=float, default='.001', help="Learning Rate")
parser.add_argument('--momentum', type=float, default='0.0', help="Momentum (for use with SGD)")
parser.add_argument('--opt', type=str, default='adam', help="Optimizer to use (adam, sgd)")
parser.add_argument('--augment', action='store_true', help="Whether to turn on data augmentation in network")
#parser.add_argument('--normalize', action='store_true', help="Whether to have float conversion and normalization inside network layers")
parser.add_argument('--label_scale', type=float, default=1, help='Scale factor to apply to labels')
parser.add_argument('--translation_factor', type=float, default=0.1, help='Amount to (randomly) translate width and height (0 - 1.0). Must be used with --augment.')
parser.add_argument('--rotation_factor', type=float, default=0.1, help='Amount to (randomly) rotate (0.0 - 1.0). Must be used with --augment.')
parser.add_argument('--zoom_factor', type=float, default=0.1, help='Amount to (randomly) zoom. Must be used with --augment.')
parser.add_argument('--data_stride', type=int, default=1, help='Stride within image sequence. Default=1.')
parser.add_argument('--data_shift', type=int, default=1, help='Window shift between windows. Default=1.')
parser.add_argument('--top_crop', type=float, default=0.0, help='Proportion of height to clip from image')
parser.add_argument('--training_duplication_multiplier', type=int, default=1, help='Number of times to duplicate training data. Ask Ramin.')

parser.set_defaults(gps_signal=False)
args = parser.parse_args()


augmentation_params = {}
augmentation_params['translation'] = args.translation_factor
augmentation_params['rotation'] = args.rotation_factor
augmentation_params['zoom'] = args.zoom_factor

#IMAGE_SHAPE                = (256, 256, 3)
IMAGE_SHAPE                = (144, 256, 3)
#IMAGE_SHAPE = (256 - int(args.top_crop * 256), 256, 3)

POSITION_SHAPE             = (4,)
REV = 0


#training_np, validation_np = load_dataset(args.data_dir, args.label_scale)

#print('============================')
#print('Training Input Shape: ', training_np[0].shape)
#print('Training Labels Shape: ', training_np[1].shape)
#print('Training Input Size in RAM: ' + str(training_np[0].size * training_np[0].itemsize / 1e9) + ' GB')
#print('----------------------------')
#print('Validation Input Shape: ', validation_np[0].shape)
#print('Validation Labels Shape: ', validation_np[1].shape)
#print('Validation Input Size in RAM: ' + str(validation_np[0].size * validation_np[0].itemsize / 1e9) + ' GB')

#training_dataset = tf.data.Dataset.from_tensor_slices(training_np).shuffle(100).batch(args.batch_size)
#validation_dataset = tf.data.Dataset.from_tensor_slices(validation_np).batch(args.batch_size)


batch_size, validation_batch_size, training_data, validation_data = load_dataset_rnn(args.data_dir, IMAGE_SHAPE, args.seq_len, args.val_split)

# hack to double the training data
td = training_data[0]
tl = training_data[1]
tdsingle = training_data[0]
tlsingle = training_data[1]
print('Data shape before duplication: ', td.shape)
if args.training_duplication_multiplier > 1 and args.training_duplication_multiplier < 2:
    ix_duplicate = (args.training_duplication_multiplier - 1) * len(tdsingle)
    td = np.append(td, tdsingle[:ix_duplicate], axis=1)
    tl = np.append(tl, tlsingle[:ix_duplicate], axis=1)
else:
    for data_duplication_ix in range(args.training_duplication_multiplier - 1):
        td = np.append(td, tdsingle, axis=1)
        tl = np.append(tl, tlsingle, axis=1)
print('Data shape after duplication: ', td.shape)

tl = np.reshape(tl, (tl.shape[0]*tl.shape[1], tl.shape[2], tl.shape[3]))
td = np.reshape(td, (td.shape[0]*td.shape[1], td.shape[2], td.shape[3], td.shape[4], td.shape[5]))
training_data = (td, tl)
print(training_data[0].shape)
print(training_data[1].shape)

vl = validation_data[1] # n batches * batch_size * data

vl_extended = np.zeros((vl.shape[0], batch_size, *vl.shape[2:]))
for ix in range(batch_size):
    vl_extended[:, ix] = vl[:, ix % vl.shape[1]]

vl_extended = np.reshape(vl_extended, (vl_extended.shape[0]*vl_extended.shape[1], vl_extended.shape[2], vl_extended.shape[3]))

vd = validation_data[0]

vd_extended = np.zeros((vd.shape[0], batch_size, *vd.shape[2:]), dtype=np.uint8)
for ix in range(batch_size):
    vd_extended[:, ix] = vd[:, ix % vd.shape[1]]
vd_extended = np.reshape(vd_extended, (vd_extended.shape[0]*vd_extended.shape[1], vd_extended.shape[2], vd_extended.shape[3], vd_extended.shape[4], vd_extended.shape[5]))

validation_data = (vd_extended, vl_extended)

print(validation_data[0].shape)
print(validation_data[1].shape)

#if args.cached_data_dir is not None:
#    cached_training_fn = os.path.join(args.cached_data_dir, 'cached_dataset_%d_%d_%d.tf' % (args.seq_len, args.data_stride, args.data_shift))
#    cached_validation_fn = os.path.join(args.cached_data_dir, 'cached_dataset_validation_%d_%d_%d.tf' % (args.seq_len, args.data_stride, args.data_shift))
#
#if args.cached_data_dir is not None and os.path.exists(cached_training_fn) and os.path.exists(cached_validation_fn):
#        print('Loading cached dataset from %s' % cached_training_fn)
#        training_dataset = tf.data.experimental.load(cached_training_fn)
#        print('Loading cached dataset from %s' % cached_validation_fn)
#        validation_dataset = tf.data.experimental.load(cached_validation_fn)
#else:
#    
#    print('Loading data from: ' + args.data_dir)
#    training_dataset, validation_dataset = get_dataset_multi(args.data_dir, IMAGE_SHAPE, args.seq_len, args.data_shift, args.data_stride, args.val_split, args.label_scale, args.extra_data_dir)
#    cached_training_fn = os.path.join(args.data_dir, 'cached_dataset_%d_%d_%d.tf' % (args.seq_len, args.data_stride, args.data_shift))
#    cached_validation_fn = os.path.join(args.data_dir, 'cached_dataset_validation_%d_%d_%d.tf' % (args.seq_len, args.data_stride, args.data_shift))
#
#    print('Saving cached training data at %s' % cached_training_fn)
#    tf.data.experimental.save(training_dataset, cached_training_fn)
#
#    print('Saving cached validation data at %s' % cached_validation_fn)
#    tf.data.experimental.save(validation_dataset, cached_validation_fn)



if args.model == 'ncp':
    #model = generate_ncp_model(args.seq_len, IMAGE_SHAPE, args.normalize, args.augment, training_np[0], augmentation_params)
    model = generate_ncp_model(args.seq_len, IMAGE_SHAPE, True, args.augment, None, augmentation_params)
elif args.model == 'lstm':
    #model = generate_lstm_model(args.rnn_size, args.seq_len, IMAGE_SHAPE, args.normalize, args.augment, training_np[0], augmentation_params)
    model = generate_lstm_model(args.rnn_sizes, args.seq_len, IMAGE_SHAPE, False, False, None, augmentation_params, rnn_stateful=True, batch_size=batch_size)
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
time_str = time.strftime("%Y:%m:%d:%H:%M:%S")
checkpointCallback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.save_dir, 'rev-%d_model-%s_seq-%d_opt-%s_lr-%f_crop-%f_epoch-{epoch:03d}_val_loss:{val_loss:.4f}_%s' % (REV, args.model, args.seq_len, args.opt, args.lr, args.top_crop, time_str)),
    save_weights_only=False,
    save_best_only=False,
    save_freq='epoch'
)

log_dir = args.tb_dir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10, 15')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=100)

print('\n====== Debug ==========')
print(training_data[0].shape)
print(training_data[1].shape)
print('\n=======================')
try:
    h = model.fit(
        x                   = training_data[0],
        y                   = training_data[1],
        validation_data     = validation_data,
        epochs              = args.epochs,
        use_multiprocessing = False,
        workers             = 1,
        max_queue_size      = 5,
        verbose             = 1,
        batch_size = batch_size,
        validation_batch_size = batch_size,
        callbacks           = [checkpointCallback, tensorboard_callback]
    )
finally:
    # Dump history
    pass
    #with open(os.path.join(args.history_dir, args.model + '-' + time.strftime("%Y:%m:%d:%H:%M:%S") + f'-history-rev={MODEL_REVISION_LABEL}.p'), 'wb') as fp:
    #    pickle.dump(model.history.history, fp)



#if args.cached_data_dir is not None:
#    training_root = args.cached_data_dir
#else:
#    training_root = args.data_dir
#output_means, output_std = get_output_normalization(training_root)
#
#
#lof = glob.glob('./model-checkpoints/*')
#last_checkpoint = max(lof, key=os.path.getmtime)
#last_model = tf.keras.models.load_model(last_checkpoint)
#
#evaluation_model = generate_lstm_model(args.rnn_sizes, args.seq_len, IMAGE_SHAPE, True, args.augment, None, augmentation_params, rnn_stateful=True)
#
#
#test_image_stacks = []
#test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_testset'
#dirs = sorted(os.listdir(test_root))
#dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
#for d in dirs:
#    n_frames = len([f for f in os.listdir(os.path.join(test_root, d)) if 'png' in f])
#    frame_stack_np = np.empty((n_frames, 144, 256, 3))
#    for ix in range(n_frames):
#        frame_stack_np[ix] = imread(os.path.join(root, d, '%06d.png' % ix))
#    test_image_stacks.append(frame_stack_np)
#
#
#raw_outputs = [np.array([evaluation_model(np.expand_dims(np.expand_dims(img, axis=0), axis=0)) for img in stack]) for stack in test_image_stacks]
#
#scaled_outputs = [(m * output_std) + output_means for m in raw_outputs]
