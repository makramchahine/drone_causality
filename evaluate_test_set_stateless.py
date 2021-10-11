#!/usr/bin/python3

import tensorflow as tf
from tf_data_loader import get_output_normalization
from keras_models import generate_lstm_model
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_SHAPE = (144, 256, 3)

training_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected'

output_means, output_std = get_output_normalization(training_root)


#lof = glob.glob('./model-checkpoints/*')
#last_checkpoint = max(lof, key=os.path.getmtime)
#last_checkpoint = './model-checkpoints/rev-0_model-lstm_seq-64_opt-adam_lr-0.010000_crop-0.000000_epoch-008_val_loss:0.6405_2021:08:24:16:12:05'


#test_model_names = ['rev-0_model-ctrnn_ct_type-mixedcfc_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-008_val_loss:0.2011_mse:0.1416_2021:09:19:13:06:47',
#                    'rev-0_model-lstm_seq-64_opt-adam_lr-0.001000_crop-0.000000_epoch-007_val_loss:0.2061_mse:0.1219_2021:09:09:02:33:09',
#                    'rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-020_val_loss:0.2127_mse:0.1679_2021:09:20:02:24:31',
#                    'rev-0_model-ctrnn_ct_type-mixedcfc_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-012_val_loss:0.2142_mse:0.1129_2021:09:19:13:06:47',
#                    'rev-0_model-ctrnn_ct_type-mixedcfc_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-014_val_loss:0.2359_mse:0.0989_2021:09:19:13:06:47',
#                    'rev-0_model-lstm_seq-256_opt-adam_lr-0.000900_crop-0.000000_epoch-012_val_loss:0.2495_mse:0.0408_2021:09:14:13:35:09',
#                    'rev-0_model-ctrnn_ct_type-mixedcfc_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-024_val_loss:0.2584_mse:0.0580_2021:09:19:11:25:20',
#                    'rev-0_model-ctrnn_ct_type-mixedcfc_seq-512_opt-adam_lr-0.000900_crop-0.000000_epoch-043_val_loss:0.2591_mse:0.0016_2021:09:17:08:30:51']

# test_model_names = ['rev-0_model-lstm_seq-64_opt-adam_lr-0.001000_crop-0.000000_epoch-007_val_loss:0.2061_mse:0.1219_2021:09:09:02:33:09',
#                     'rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-020_val_loss:0.2127_mse:0.1679_2021:09:20:02:24:31',
#                     'rev-0_model-lstm_seq-256_opt-adam_lr-0.000900_crop-0.000000_epoch-012_val_loss:0.2495_mse:0.0408_2021:09:14:13:35:09']

test_model_names = ['rev-0_model-ncp_seq-64_opt-adam_lr-0.000900_crop-0.000000_epoch-049_val_loss:0.2629_mse:0.0672_2021:09:24:09:22:31.hdf5']

checkpoints = ['./model-checkpoints/' + n for n in test_model_names]


for (ix, c) in enumerate(checkpoints):
    last_checkpoint = c
    last_model = tf.keras.models.load_model(last_checkpoint)

    model_name = test_model_names[ix]
    #seq_len = int(model_name.split('_')[2].split['-'][1]
    seq_len = None
    for tok in model_name.split('_'):
        if 'seq' in tok:
            seq_len = int(tok.split('-')[1])
    if seq_len is None:
        raise Exception('Cannot find sequence length in filename %s' % model_name)

    #evaluation_model = generate_lstm_model(rnn_sizes, seq_len, IMAGE_SHAPE, False, False, False, False, rnn_stateful=False, batch_size=1)
    #evaluation_model.set_weights(last_model.get_weights())
    evaluation_model = last_model

    output_folder = '/home/ramin/deepdrone/test_plots/%s/test' % model_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected_test'
    test_image_stacks = []
    test_control_stacks = []
    dirs = sorted(os.listdir(test_root))
    dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
    for d in dirs:
        print(d)
        outputs = []
        labels = np.genfromtxt(os.path.join(test_root, d, 'data_out.csv'), delimiter=',', skip_header=1)
        n_frames = len([f for f in os.listdir(os.path.join(test_root, d)) if 'png' in f])
        frame_stack_np = np.zeros((n_frames, 144, 256, 3))
        for ix in range(n_frames):
            frame_stack_np[ix] = Image.open(os.path.join(test_root, d, '%06d.png' % ix))
        for ix in range(seq_len, n_frames):
            raw_outputs = evaluation_model(np.expand_dims(frame_stack_np[ix - seq_len:ix], axis=0))
            scaled_outputs = (raw_outputs[0, -1] * output_std) + output_means
            outputs.append(scaled_outputs)
        
        outputs_flat = np.vstack(outputs)
        print(outputs_flat.shape)
        op = os.path.join(output_folder, d)
        if not os.path.exists(op):
            os.mkdir(op)
        np.savetxt(os.path.join(op, 'reference_data.csv'), labels, delimiter=',', header='vx,vy,vz,omega_z')
        np.savetxt(os.path.join(op, 'prediction_data.csv'), outputs_flat, delimiter=',', header='vx,vy,vz,omega_z')

    output_folder = '/home/ramin/deepdrone/test_plots/%s/train' % model_name
    test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_corrected'
    test_image_stacks = []
    test_control_stacks = []
    dirs = sorted(os.listdir(test_root))
    dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
    for d in dirs[:3]:
        print(d)
        outputs = []
        labels = np.genfromtxt(os.path.join(test_root, d, 'data_out.csv'), delimiter=',', skip_header=1)
        n_frames = len([f for f in os.listdir(os.path.join(test_root, d)) if 'png' in f])
        frame_stack_np = np.zeros((n_frames, 144, 256, 3))
        for ix in range(n_frames):
            frame_stack_np[ix] = Image.open(os.path.join(test_root, d, '%06d.png' % ix))
        for ix in range(seq_len, n_frames):
            raw_outputs = evaluation_model(np.expand_dims(frame_stack_np[ix - seq_len:ix], axis=0))[0, -1]
            scaled_outputs = (raw_outputs * output_std) + output_means
            outputs.append(scaled_outputs)
        
        outputs_flat = np.vstack(outputs)
        print(outputs_flat.shape)
        op = os.path.join(output_folder, d)
        if not os.path.exists(op):
            os.makedirs(op)
        np.savetxt(os.path.join(op, 'reference_data.csv'), labels, delimiter=',', header='vx,vy,vz,omega_z')
        np.savetxt(os.path.join(op, 'prediction_data.csv'), outputs_flat, delimiter=',', header='vx,vy,vz,omega_z')




