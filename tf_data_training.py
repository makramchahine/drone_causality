#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import time

import tensorflow as tf
from tensorflow import keras

from tf_data_loader import get_dataset_multi
from keras_models import ModelParams, NCPParams, \
    LSTMParams, CTRNNParams, IMAGE_SHAPE, get_readable_name, get_skeleton


def tlen(dataset):
    for (ix, _) in enumerate(dataset):
        pass
    return ix


def train_model(model_params: ModelParams, data_dir: str = "./data", cached_data_dir: str = None,
                extra_data_dir: str = None, save_dir: str = "./model_checkpoints", batch_size: int = 32,
                epochs: int = 30, val_split: float = 0.1, hotstart: str = None, lr: float = 0.001, momentum: float = 0,
                opt: str = "adam", label_scale: float = 1, data_shift: int = 1, data_stride: int = 1,
                top_crop: float = 0.0, decay_rate: float = 0.95, callbacks: List = None):
    # create model checkpoint directory if doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # IMAGE_SHAPE = (256 - int(top_crop * 256), 256, 3)
    # POSITION_SHAPE = (4,)

    if cached_data_dir is not None:
        cached_training_fn = os.path.join(cached_data_dir, 'cached_dataset_%d_%d_%d.tf' % (
            model_params.seq_len, data_stride, data_shift))
        cached_validation_fn = os.path.join(cached_data_dir, 'cached_dataset_validation_%d_%d_%d.tf' % (
            model_params.seq_len, data_stride, data_shift))

    if cached_data_dir is not None and os.path.exists(cached_training_fn) and os.path.exists(cached_validation_fn):

        print('Loading cached dataset from %s' % cached_training_fn)
        training_dataset = tf.data.experimental.load(cached_training_fn)
        print('Loading cached dataset from %s' % cached_validation_fn)
        validation_dataset = tf.data.experimental.load(cached_validation_fn)

    else:

        print('Loading data from: ' + data_dir)
        training_dataset, validation_dataset = get_dataset_multi(data_dir, IMAGE_SHAPE, model_params.seq_len,
                                                                 data_shift,
                                                                 data_stride, val_split, label_scale, extra_data_dir)
        cached_training_fn = os.path.join(data_dir, 'cached_dataset_%d_%d_%d.tf' % (
            model_params.seq_len, data_stride, data_shift))
        cached_validation_fn = os.path.join(data_dir, 'cached_dataset_validation_%d_%d_%d.tf' % (
            model_params.seq_len, data_stride,
            data_shift))  # print('Saving cached training data at %s' % cached_training_fn)  # tf.data.experimental.save(training_dataset, cached_training_fn)

        # print('Saving cached validation data at %s' % cached_validation_fn)  # tf.data.experimental.save(validation_dataset, cached_validation_fn)

    print('\n\nTraining Dataset Size: %d\n\n' % tlen(training_dataset))
    training_dataset = training_dataset.shuffle(100).batch(batch_size)

    validation_dataset = validation_dataset.batch(batch_size)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
                                                              decay_rate=decay_rate, staircase=True)

    if opt == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif opt == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)
    else:
        raise Exception('Unsupported optimizer type %s' % opt)

    time_str = time.strftime("%Y:%m:%d:%H:%M:%S")
    REV = 0
    if isinstance(model_params, CTRNNParams):
        dc = model_params.config  # for abbreviated writing
        file_path = os.path.join(save_dir,
                                 'rev-%d' % REV + '_model-ctrnn' + '_ctt-%s' % model_params.ct_network_type + '_cn-%f' %
                                 dc['clipnorm'] + '_bba-%s' % dc['backbone_activation'] + '_bb-dr-%f' % dc[
                                     'backbone_dr'] + '_fb-%f' % dc['forget_bias'] + '_bbu-%d' % dc[
                                     'backbone_units'] + '_bbl-%d' % dc['backbone_layers'] + '_wd-%f' % dc[
                                     'weight_decay'] + '_seq-%d' % model_params.seq_len + '_opt-%s' % opt + '_lr-%f' % lr + '_crop-%f' % top_crop + '_epoch-{epoch:03d}' + '_val-loss:{val_loss:.4f}' + '_mse:{mse:.4f}' + '_%s.hdf5' % time_str)
    else:
        file_path = os.path.join(save_dir, 'rev-%d_model-%s_seq-%d_opt-%s'
                                           '_lr-%f_crop-%f_epoch-{epoch:03d}'
                                           '_val_loss:{val_loss:.4f}_mse:{mse:.4f}_%s.hdf5' % (
                                     REV, get_readable_name(model_params), model_params.seq_len, opt, lr, top_crop,
                                     time_str))

    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True,
                                                          save_best_only=False, save_freq='epoch')

    if callbacks is None:
        callbacks = []

    callbacks.append(checkpoint_callback)
    print(f"Saving checkpoints at {file_path}")

    # use data parallelism to split data across GPUs
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        model = get_skeleton(params=model_params, single_step=False)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])
        # Load pretrained weights
        if hotstart is not None:
            model.load_weights(hotstart)

        model.summary(line_length=80)

    # Train
    history = model.fit(x=training_dataset, validation_data=validation_dataset, epochs=epochs,
                        use_multiprocessing=False, workers=1, max_queue_size=5, verbose=1, callbacks=callbacks)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model on deepdrone data')
    parser.add_argument('--model', type=str, default="ncp", help='The type of model (ncp, lstm, ctrnn)')
    parser.add_argument('--ct_type', type=str, default="ctrnn",
                        help='The type of the continuous model (ctrnn, node, cfc, '
                             'ctgru, grud, mmrnn, mixedcfc, '
                             'bidirect, vanilla, phased, gruode, hawk, ltc)')
    parser.add_argument('--rnn_sizes', type=int, nargs='+',
                        help='Select the size of RNN network you would like to train')
    parser.add_argument('--data_dir', type=str, default="./data", help='Path to training data')
    parser.add_argument('--test_data_dir', type=str, default="./data", help='Path to test data')
    parser.add_argument('--cached_data_dir', type=str, default=None, help='Path to pre-cached dataset')
    parser.add_argument('--extra_data_dir', type=str, default=None,
                        help='Path to extra training data, used for training '
                             'but not validation')
    parser.add_argument('--save_dir', type=str, default="./model_checkpoints", help='Path to save checkpoints')
    parser.add_argument('--history_dir', type=str, default="./histories", help='Path to save history')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--hotstart', type=str, default=None, help="Starting weights to use for pretraining")
    # parser.add_argument('--tb_dir', type=str, default='tb_logs',
    #                    help="Name of directory to save tensorboard logs")
    parser.add_argument('--lr', type=float, default='.001', help="Learning Rate")
    parser.add_argument('--momentum', type=float, default='0.0', help="Momentum (for use with SGD)")
    parser.add_argument('--opt', type=str, default='adam', help="Optimizer to use (adam, sgd)")
    parser.add_argument('--augmentation', action='store_true', help="Whether to turn on data augmentation in network")
    # parser.add_argument('--normalize', action='store_true',
    #                     help="Whether to have float conversion
    #                     and normalization inside network layers")
    parser.add_argument('--label_scale', type=float, default=1, help='Scale factor to apply to labels')
    parser.add_argument('--translation_factor', type=float, default=0.1,
                        help='Amount to (randomly) translate width and height '
                             '(0 - 1.0). Must be used with --augment.')
    parser.add_argument('--rotation_factor', type=float, default=0.1, help='Amount to (randomly) rotate (0.0 - 1.0). '
                                                                           'Must be used with --augment.')
    parser.add_argument('--zoom_factor', type=float, default=0.1,
                        help='Amount to (randomly) zoom. Must be used with --augment.')
    parser.add_argument('--data_stride', type=int, default=1, help='Stride within image sequence. Default=1.')
    parser.add_argument('--data_shift', type=int, default=1, help='Window shift between windows. Default=1.')
    parser.add_argument('--top_crop', type=float, default=0.0, help='Proportion of height to clip from image')
    parser.add_argument('--decay_rate', type=float, default=0.95, help="Exponential decay rate of the lr scheduler")
    parser.add_argument("--ncp_seed", type=int, default=2222, help="Seed for ncp")
    parser.set_defaults(gps_signal=False)
    args = parser.parse_args()
    # setup model params and augment params dataclasses
    augmentation_params = {"translation_factor": args.translation_factor, "rotation_factor": args.rotation_factor,
                           "zoom_factor": args.zoom_factor}

    if args.model == "ncp":
        model_params_constructed = NCPParams(seq_len=args.seq_len, do_augmentation=args.augmentation,
                                             augmentation_params=augmentation_params, seed=args.ncp_seed)
    elif args.model == "lstm":
        model_params_constructed = LSTMParams(seq_len=args.seq_len, do_augmentation=args.augmentation,
                                              augmentation_params=augmentation_params, rnn_sizes=args.rnn_sizes, )
    elif args.model == "ctrnn":
        model_params_constructed = CTRNNParams(seq_len=args.seq_len, do_augmentation=args.augmentation,
                                               augmentation_params=augmentation_params, rnn_sizes=args.rnn_sizes,
                                               ct_network_type=args.ct_type)
    else:
        raise ValueError(f"Passed in illegal model type {args.model_type}")

    train_model(data_dir=args.data_dir, epochs=args.epochs, val_split=args.val_split,
                opt=args.opt, lr=args.lr, data_shift=args.data_shift, data_stride=args.data_stride,
                batch_size=args.batch_size, save_dir=args.save_dir, hotstart=args.hotstart, momentum=args.momentum,
                cached_data_dir=args.cached_data_dir, label_scale=args.label_scale,
                top_crop=args.top_crop, model_params=model_params_constructed, decay_rate=args.decay_rate)
