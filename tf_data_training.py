#!/usr/bin/python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import pickle
import random
import pathlib
import time
import glob

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tf_data_loader import load_dataset, get_dataset_multi, get_output_normalization
from keras_models import generate_ncp_model, generate_lstm_model, generate_ctrnn_model

def tlen(dataset):
    for (ix, _) in enumerate(dataset):
        pass
    return ix

MODEL_REVISION_LABEL = 13.0

# This is for CfC models
DEFAULT_CONFIG = {
    "clipnorm": 1,
    "size": 64,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
}

parser = argparse.ArgumentParser(description='Train the model on deepdrone data')

parser.add_argument('--model', type=str, default="ncp",
                    help='The type of model (ncp, lstm, ctrnn)')

parser.add_argument('--ct_type', type=str, default="ctrnn",
                    help='The type of the continuous model (ctrnn, node, cfc, '
                         'ctgru, grud, mmrnn, mixedcfc, '
                         'bidirect, vanilla, phased, gruode, hawk, ltc)')

parser.add_argument('--rnn_sizes', type=int, nargs='+',
                    help='Select the size of RNN network you would like to train')

parser.add_argument('--data_dir', type=str, default="./data",
                    help='Path to training data')

parser.add_argument('--cached_data_dir', type=str, default=None,
                    help='Path to pre-cached dataset')

parser.add_argument('--extra_data_dir', type=str, default=None,
                    help='Path to extra training data, used for training '
                         'but not validation')

parser.add_argument('--save_dir', type=str, default="./model-checkpoints",
                    help='Path to save checkpoints')

parser.add_argument('--history_dir', type=str, default="./histories",
                    help='Path to save history')

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--seq_len', type=int, default=64)

parser.add_argument('--epochs', type=int, default=30)

parser.add_argument('--val_split', type=float, default=0.1)

parser.add_argument('--hotstart', type=str, default=None,
                    help="Starting weights to use for pretraining")

#parser.add_argument('--tb_dir', type=str, default='tb_logs',
#                    help="Name of directory to save tensorboard logs")

parser.add_argument('--lr', type=float, default='.001',
                    help="Learning Rate")

parser.add_argument('--momentum', type=float, default='0.0',
                    help="Momentum (for use with SGD)")

parser.add_argument('--opt', type=str, default='adam',
                    help="Optimizer to use (adam, sgd)")

parser.add_argument('--augment', action='store_true',
                    help="Whether to turn on data augmentation in network")

# parser.add_argument('--normalize', action='store_true',
#                     help="Whether to have float conversion
#                     and normalization inside network layers")

parser.add_argument('--label_scale', type=float, default=1,
                    help='Scale factor to apply to labels')

parser.add_argument('--translation_factor', type=float, default=0.1,
                    help='Amount to (randomly) translate width and height '
                         '(0 - 1.0). Must be used with --augment.')

parser.add_argument('--rotation_factor', type=float, default=0.1,
                    help='Amount to (randomly) rotate (0.0 - 1.0). '
                         'Must be used with --augment.')

parser.add_argument('--zoom_factor', type=float, default=0.1,
                    help='Amount to (randomly) zoom. Must be used with --augment.')

parser.add_argument('--data_stride', type=int, default=1,
                    help='Stride within image sequence. Default=1.')

parser.add_argument('--data_shift', type=int, default=1,
                    help='Window shift between windows. Default=1.')

parser.add_argument('--top_crop', type=float, default=0.0,
                    help='Proportion of height to clip from image')

parser.set_defaults(gps_signal=False)
args = parser.parse_args()

augmentation_params = {}
augmentation_params['translation'] = args.translation_factor
augmentation_params['rotation'] = args.rotation_factor
augmentation_params['zoom'] = args.zoom_factor

IMAGE_SHAPE                = (144, 256, 3)
# IMAGE_SHAPE = (256 - int(args.top_crop * 256), 256, 3)

POSITION_SHAPE             = (4,)
REV = 0


# training_np, validation_np = load_dataset(args.data_dir, args.label_scale)

# print('============================')
# print('Training Input Shape: ', training_np[0].shape)
# print('Training Labels Shape: ', training_np[1].shape)
# print('Training Input Size in RAM: ' + str(training_np[0].size * training_np[0].itemsize / 1e9) + ' GB')
# print('----------------------------')
# print('Validation Input Shape: ', validation_np[0].shape)
# print('Validation Labels Shape: ', validation_np[1].shape)
# print('Validation Input Size in RAM: ' + str(validation_np[0].size * validation_np[0].itemsize / 1e9) + ' GB')

# training_dataset = tf.data.Dataset.from_tensor_slices(training_np).shuffle(100).batch(args.batch_size)
# validation_dataset = tf.data.Dataset.from_tensor_slices(validation_np).batch(args.batch_size)


if args.cached_data_dir is not None:
    cached_training_fn = os.path.join(args.cached_data_dir,
                                      'cached_dataset_%d_%d_%d.tf'
                                      % (args.seq_len,
                                         args.data_stride,
                                         args.data_shift))
    cached_validation_fn = os.path.join(args.cached_data_dir,
                                        'cached_dataset_validation_%d_%d_%d.tf'
                                        % (args.seq_len,
                                           args.data_stride,
                                           args.data_shift))

if args.cached_data_dir is not None and os.path.exists(cached_training_fn) \
    and os.path.exists(cached_validation_fn):

    print('Loading cached dataset from %s' % cached_training_fn)
    training_dataset = tf.data.experimental.load(cached_training_fn)
    print('Loading cached dataset from %s' % cached_validation_fn)
    validation_dataset = tf.data.experimental.load(cached_validation_fn)

else:
    
    print('Loading data from: ' + args.data_dir)
    training_dataset, validation_dataset = get_dataset_multi(args.data_dir,
                                                             IMAGE_SHAPE,
                                                             args.seq_len,
                                                             args.data_shift,
                                                             args.data_stride,
                                                             args.val_split,
                                                             args.label_scale,
                                                             args.extra_data_dir)
    cached_training_fn = os.path.join(args.data_dir,
                                      'cached_dataset_%d_%d_%d.tf'
                                      % (args.seq_len, args.data_stride, args.data_shift))
    cached_validation_fn = os.path.join(args.data_dir,
                                        'cached_dataset_validation_%d_%d_%d.tf'
                                        % (args.seq_len, args.data_stride, args.data_shift))

    #print('Saving cached training data at %s' % cached_training_fn)
    #tf.data.experimental.save(training_dataset, cached_training_fn)

    #print('Saving cached validation data at %s' % cached_validation_fn)
    #tf.data.experimental.save(validation_dataset, cached_validation_fn)



print('\n\nTraining Dataset Size: %d\n\n' % tlen(training_dataset))
training_dataset = training_dataset.shuffle(100).batch(args.batch_size)


validation_dataset = validation_dataset.batch(args.batch_size)


if args.model == 'ncp':
    model = generate_ncp_model(args.seq_len,
                               IMAGE_SHAPE,
                               False,
                               args.augment,
                               None,
                               augmentation_params,
                               rnn_stateful=False)
elif args.model == 'lstm':
    model = generate_lstm_model(args.rnn_sizes,
                                args.seq_len,
                                IMAGE_SHAPE, False,
                                args.augment, None,
                                augmentation_params,
                                rnn_stateful=False)
elif args.model == 'ctrnn':
    model = generate_ctrnn_model(args.rnn_sizes,
                                 args.seq_len,
                                 IMAGE_SHAPE, False,
                                 args.augment, None,
                                 augmentation_params,
                                 rnn_stateful=False,
                                 batch_size=None,
                                 ct_network_type=args.ct_type,
                                 config=DEFAULT_CONFIG)
else:
    raise Exception('Unsupported model type: %s' % args.model)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.lr,
    decay_steps=500,
    decay_rate=0.95)

if args.opt == 'adam':
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
elif args.opt == 'sgd':
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule,
                                     momentum=args.momentum)
else:
    raise Exception('Unsupported optimizer type %s' % args.opt)

model.compile(optimizer=optimizer,
              loss="mean_squared_error",
              metrics=['mse'])

# Load pretrained weights
if args.hotstart is not None:
    model.load_weights(args.hotstart)

model.summary(line_length=80)

# Train
time_str = time.strftime("%Y:%m:%d:%H:%M:%S")
if args.model == 'ctrnn':
    file_path = os.path.join(args.save_dir,
                             'rev-%d_model-%s_ct_type-%s'
                             '_seq-%d_opt-%s_lr-%f_crop-%f'
                             '_epoch-{epoch:03d}_val_loss:{val_loss:.4f}'
                             '_mse:{mse:.4f}_%s'
                             % (REV, args.model, args.ct_type, args.seq_len,
                                args.opt, args.lr, args.top_crop, time_str))
else:
    file_path = os.path.join(args.save_dir,
                             'rev-%d_model-%s_seq-%d_opt-%s'
                             '_lr-%f_crop-%f_epoch-{epoch:03d}'
                             '_val_loss:{val_loss:.4f}_mse:{mse:.4f}_%s'
                             % (REV, args.model, args.seq_len, args.opt,
                                args.lr, args.top_crop, time_str))

checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                     save_weights_only=False,
                                                     save_best_only=False,
                                                     save_freq='epoch'
                                                     )

# log_dir = args.tb_dir
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10, 15')
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=100)

try:
    h = model.fit(
        x                   = training_dataset,
        validation_data     = validation_dataset,
        epochs              = args.epochs,
        use_multiprocessing = False,
        workers             = 1,
        max_queue_size      = 5,
        verbose             = 1,
        callbacks           = [checkpointCallback]
    )
finally:
    # Dump history
    pass
    # with open(os.path.join(args.history_dir, args.model + '-' +
    # time.strftime("%Y:%m:%d:%H:%M:%S") + f'-history-rev={MODEL_REVISION_LABEL}.p'), 'wb') as fp:
    #    pickle.dump(model.history.history, fp)



# if args.cached_data_dir is not None:
#    training_root = args.cached_data_dir
# else:
#    training_root = args.data_dir
# output_means, output_std = get_output_normalization(training_root)
#
#
# lof = glob.glob('./model-checkpoints/*')
# last_checkpoint = max(lof, key=os.path.getmtime)
# last_model = tf.keras.models.load_model(last_checkpoint)
#
# evaluation_model = generate_lstm_model(args.rnn_sizes,
#                                        args.seq_len,
#                                        IMAGE_SHAPE,
#                                        True,
#                                        args.augment,
#                                        None,
#                                        augmentation_params,
#                                        rnn_stateful=True)
#
#
# test_image_stacks = []
# test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_testset'
# dirs = sorted(os.listdir(test_root))
# dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
# for d in dirs:
#    n_frames = len([f for f in os.listdir(os.path.join(test_root, d)) if 'png' in f])
#    frame_stack_np = np.empty((n_frames, 144, 256, 3))
#    for ix in range(n_frames):
#        frame_stack_np[ix] = imread(os.path.join(root, d, '%06d.png' % ix))
#    test_image_stacks.append(frame_stack_np)
#
#
# raw_outputs = [np.array([evaluation_model(
#                                           np.expand_dims(np.expand_dims(img, axis=0), axis=0))
#                                           for img in stack]) for stack in test_image_stacks]
#
# scaled_outputs = [(m * output_std) + output_means for m in raw_outputs]
