#!/usr/bin/env python3
import functools
import os
from pathlib import Path
from typing import List, Dict, Any
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import time

import tensorflow as tf
from tensorflow import keras, TensorSpec

from tf_data_loader import get_dataset_multi
from keras_models import IMAGE_SHAPE
from utils.model_utils import ModelParams, NCPParams, LSTMParams, CTRNNParams, TCNParams, get_skeleton, \
    get_readable_name

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def tlen(dataset):
    for (ix, _) in enumerate(dataset):
        pass
    return ix


@tf.function
def sequence_augmentation(x, y, aug_params: Dict[str, Any]):
    """
    Apply augmentations where params have to be fixed per-sequence, and not per-sample. Therefore, these augmentations
    can't go in a layer, as TimeDistribtued would call the layer again and again for each timestep

    Note: to set breakpoints in a tf.function, you need to run tf.config.run_functions_eagerly(True) after import

    :param x: data input, has shape batch x seq_len x height x width x channels
    :param y: data labels, have shape batch x seq_len x 4
    :param aug_params: dictionary containing intensity of augmentations. Keys can include brightness, contrast, and
    saturation
    :return: augmented data input, same data labels
    """
    xi = x["input_image"]
    xi2 = x["input_image2"]
    xv = x["input_vector"]
    bright_range = aug_params.get("brightness", None)
    if bright_range is not None:
        delta = tf.random.uniform((), -bright_range, bright_range)
        xi = tf.image.adjust_brightness(xi, delta)
        xi2 = tf.image.adjust_brightness(xi2, delta)

    contrast_range = aug_params.get("contrast", None)
    if contrast_range is not None:
        contrast_factor = tf.random.uniform((), 1 - contrast_range, 1 + contrast_range)
        xi = tf.image.adjust_contrast(xi, contrast_factor)
        xi2 = tf.image.adjust_contrast(xi2, contrast_factor)

    saturation_range = aug_params.get("saturation", None)
    if saturation_range is not None:
        saturation_factor = tf.random.uniform((), 1 - saturation_range, 1 + saturation_range)
        xi = tf.image.adjust_saturation(xi, saturation_factor)
        xi2 = tf.image.adjust_saturation(xi2, saturation_factor)

    return {"input_image":xi, "input_image2":xi2, "input_vector":xv}, y


def train_model(model_params: ModelParams, data_dir: str = "./data", cached_data_dir: str = None,
                extra_data_dir: str = None, save_dir: str = "./model_checkpoints", batch_size: int = 32,
                epochs: int = 30, val_split: float = 0.1, hotstart: str = None, lr: float = 0.001, momentum: float = 0,
                opt: str = "adam", label_scale: float = 1, data_shift: int = 1, data_stride: int = 1,
                decay_rate: float = 0.95, callbacks: List = None, save_period: int = 1):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # create model checkpoint directory if doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # make sure data loading happens on CPU
    with tf.device('/cpu:0'):
        if cached_data_dir is not None:
            Path(cached_data_dir).mkdir(parents=True, exist_ok=True)
            data_folder = os.path.basename(data_dir)
            extra_data_str = f"_{os.path.basename(extra_data_dir)}" if extra_data_dir is not None else ""
            cached_training_fn = os.path.join(cached_data_dir, 'cached_dataset_%s%s_%d_%d_%d.tf' % (
                data_folder, extra_data_str, model_params.seq_len, data_stride, data_shift))
            cached_validation_fn = os.path.join(cached_data_dir, 'cached_dataset_%s%s_validation_%d_%d_%d.tf' % (
                data_folder, extra_data_str, model_params.seq_len, data_stride, data_shift))
            dataset_spec = os.path.join(cached_data_dir,
                                        f"cached_{data_folder}{extra_data_str}_{model_params.seq_len}_{data_stride}_{data_shift}_spec.txt")

        if cached_data_dir is not None and os.path.exists(cached_training_fn) and os.path.exists(
                cached_validation_fn) and os.path.exists(dataset_spec):
            # loading datasets in older versions of tensorflow requires a TensorSpec to describe
            with open(dataset_spec, "r") as f:
                spec_str = f.readlines()[0]
            spec: TensorSpec = eval(spec_str)
            print('Loading cached dataset from %s' % cached_training_fn)
            training_dataset = tf.data.experimental.load(cached_training_fn, spec)
            print('Loading cached dataset from %s' % cached_validation_fn)
            validation_dataset = tf.data.experimental.load(cached_validation_fn, spec)
        else:
            print('Loading data from: ' + data_dir)
            training_dataset, validation_dataset = get_dataset_multi(data_dir, IMAGE_SHAPE, model_params.seq_len,
                                                                     data_shift,
                                                                     data_stride, val_split, label_scale,
                                                                     extra_data_dir)

            if cached_data_dir is not None:
                print('Saving cached training data at %s' % cached_training_fn)
                tf.data.experimental.save(training_dataset, cached_training_fn)
                print('Saving cached validation data at %s' % cached_validation_fn)
                tf.data.experimental.save(validation_dataset, cached_validation_fn)
                with open(dataset_spec, "w") as f:
                    f.write(repr(training_dataset.element_spec))

        print('\n\nTraining Dataset Size: %d\n\n' % tlen(training_dataset))
        training_dataset = training_dataset.shuffle(100).batch(batch_size)
        # handle sequence augmentations differently
        seq_params = model_params.augmentation_params.get("sequence_params", None)
        if isinstance(seq_params, dict):
            print("Performing sequence aug on training dataset")
            seq_aug_fn = functools.partial(sequence_augmentation, aug_params=seq_params)
            training_dataset = training_dataset.map(
                seq_aug_fn, num_parallel_calls=tf.data.AUTOTUNE
            )
        validation_dataset = validation_dataset.batch(batch_size)
        # remove annoying TF warning about dataset sharding across multiple GPUs
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        training_dataset = training_dataset.with_options(options)
        validation_dataset = validation_dataset.with_options(options)
        # Have GPU prefetch next training batch while first one runs
        training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=500,
                                                              decay_rate=decay_rate, staircase=True)

    time_str = time.strftime("%Y:%m:%d:%H:%M:%S")

    file_path = os.path.join(save_dir, 'model-%s_seq-%d_lr-%f_epoch-{epoch:03d}'
                                       '_val-loss:{val_loss:.4f}_train-loss:{loss:.4f}_mse:{mse:.4f}_%s.hdf5' % (
                                 get_readable_name(model_params), model_params.seq_len, lr, time_str))

    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=file_path, save_weights_only=True,
                                                          save_best_only=False, save_freq='epoch', period=save_period)

    if callbacks is None:
        callbacks = []

    callbacks.append(checkpoint_callback)
    print(f"Saving checkpoints at {file_path}")

    # use data parallelism to split data across GPUs
    gpus = tf.config.list_logical_devices('GPU')
    #strategy = tf.distribute.MirroredStrategy(gpus)

    #tf_config = {
    #    'cluster': {
    #        'worker': ['localhost:12345', 'localhost:23456']
    #     },
    #    'task': {'type': 'worker', 'index': 0}
    #}
    #os.environ['TF_CONFIG'] = json.dumps(tf_config)
    with strategy.scope():
        if opt == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        elif opt == 'sgd':
            optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)
        else:
            raise Exception('Unsupported optimizer type %s' % opt)

        model = get_skeleton(params=model_params)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])
        # Load pretrained weights
        if hotstart is not None:
            model.load_weights(hotstart)

        model.summary(line_length=80)

    # Train
    #training_dataset = strategy.distribute_datasets_from_function(training_dataset)
    #validation_dataset = strategy.distribute_datasets_from_function(validation_dataset)
    history = model.fit(x=training_dataset, validation_data=validation_dataset, epochs=epochs,
                        use_multiprocessing=False, workers=1, max_queue_size=5, verbose=1, callbacks=callbacks)
    return history, time_str


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
    parser.add_argument('--batch_size', type=int, default=32, help="Number of sequences in one training batch")
    parser.add_argument('--seq_len', type=int, default=64, help="Number of data points per sequence within each batch")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs to train for")
    parser.add_argument('--val_split', type=float, default=0.1, help="Fraction of dataset that becomes validation set")
    parser.add_argument('--hotstart', type=str, default=None, help="Starting weights to use for pretraining")
    parser.add_argument('--lr', type=float, default='.001', help="Learning Rate")
    parser.add_argument('--momentum', type=float, default='0.0', help="Momentum (for use with SGD)")
    parser.add_argument('--opt', type=str, default='adam', help="Optimizer to use (adam, sgd)")
    parser.add_argument('--augmentation', action='store_true', help="Whether to turn on data augmentation in network")
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
    parser.add_argument('--decay_rate', type=float, default=0.95, help="Exponential decay rate of the lr scheduler")
    parser.add_argument("--ncp_seed", type=int, default=2222, help="Seed for ncp")
    parser.add_argument("--tcn_nb_filters", type=int, default=128, help="Number of tcn filters")
    parser.add_argument("-t", "--tcn_dilations", action='append', help='tcn dilations, use flag multiple times',
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--tcn_kernel", type=int, default=2, help="Size of tcn kernel")
    parser.set_defaults(gps_signal=False)
    args = parser.parse_args()
    # setup model params and augment params dataclasses
    augmentation_params = {"translation_factor": args.translation_factor, "rotation_factor": args.rotation_factor,
                           "zoom_factor": args.zoom_factor} if args.augmentation else None

    if args.model == "ncp":
        model_params_constructed = NCPParams(seq_len=args.seq_len,
                                             augmentation_params=augmentation_params, seed=args.ncp_seed)
    elif args.model == "lstm":
        model_params_constructed = LSTMParams(seq_len=args.seq_len,
                                              augmentation_params=augmentation_params, rnn_sizes=args.rnn_sizes, )
    elif args.model == "ctrnn":
        model_params_constructed = CTRNNParams(seq_len=args.seq_len,
                                               augmentation_params=augmentation_params, rnn_sizes=args.rnn_sizes,
                                               ct_network_type=args.ct_type)
    elif args.model == "tcn":
        model_params_constructed = TCNParams(seq_len=args.seq_len, nb_filters=args.tcn_nb_filters,
                                             augmentation_params=augmentation_params, kernel_size=args.tcn_kernel,
                                             dilations=args.tcn_dilations)
    else:
        raise ValueError(f"Passed in illegal model type {args.model_type}")

    train_model(data_dir=args.data_dir, epochs=args.epochs, val_split=args.val_split,
                opt=args.opt, lr=args.lr, data_shift=args.data_shift, data_stride=args.data_stride,
                batch_size=args.batch_size, save_dir=args.save_dir, hotstart=args.hotstart, momentum=args.momentum,
                cached_data_dir=args.cached_data_dir, label_scale=args.label_scale,
                model_params=model_params_constructed, decay_rate=args.decay_rate)
