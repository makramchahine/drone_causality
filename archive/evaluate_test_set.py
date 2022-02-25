#!/usr/bin/python3

import tensorflow as tf
from tf_data_loader import get_output_normalization
from keras_models import generate_lstm_model
import glob
import os
import numpy as np
from matplotlib.image import imread

IMAGE_SHAPE = (144, 256, 3)

training_root = '/home/ramin/devens_drone_data/devens_2021-08-04'

output_means, output_std = get_output_normalization(training_root)


#lof = glob.glob('./model-checkpoints/*')
#last_checkpoint = max(lof, key=os.path.getmtime)
#last_checkpoint = './model-checkpoints/rev-0_model-lstm_seq-64_opt-adam_lr-0.010000_crop-0.000000_epoch-008_val_loss:0.6405_2021:08:24:16:12:05'
last_checkpoint = './model-checkpoints/rev-0_model-lstm_seq-8_opt-adam_lr-0.000500_crop-0.000000_epoch-077_val_loss:0.1631_2021:08:29:13:15:08'
last_model = tf.keras.models.load_model(last_checkpoint)

seq_len = 1
rnn_sizes = [64]
evaluation_model = generate_lstm_model(rnn_sizes, seq_len, IMAGE_SHAPE, False, False, False, False, rnn_stateful=True, batch_size=1)

evaluation_model.set_weights(last_model.get_weights())


output_folder = '/home/ramin/r_deepdrone/test_plots/test'
test_root = '/home/ramin/devens_drone_data/devens_2021-08-04_testset'
test_image_stacks = []
test_control_stacks = []
dirs = sorted(os.listdir(test_root))
dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
for d in dirs:
    print(d)
    labels = np.genfromtxt(os.path.join(test_root, d, 'data_out.csv'), delimiter=',', skip_header=1)
    n_frames = len([f for f in os.listdir(os.path.join(test_root, d)) if 'png' in f])
    frame_stack_np = np.zeros((n_frames, 144, 256, 3))
    for ix in range(n_frames):
        frame_stack_np[ix] = imread(os.path.join(test_root, d, '%06d.png' % ix))
    raw_outputs = np.array([evaluation_model(np.expand_dims(np.expand_dims(img, axis=0), axis=0)) for img in frame_stack_np])[:,0,0,:]
    scaled_outputs = (raw_outputs * output_std) + output_means
    op = os.path.join(output_folder, d)
    if not os.path.exists(op):
        os.mkdir(op)
    np.savetxt(os.path.join(op, 'reference_data.csv'), labels, delimiter=',', header='vx,vy,vz,omega_z')
    np.savetxt(os.path.join(op, 'prediction_data.csv'), scaled_outputs, delimiter=',', header='vx,vy,vz,omega_z')

output_folder = '/home/ramin/r_deepdrone/test_plots/train'
test_root = '/home/ramin/devens_drone_data/devens_2021-08-04'
test_image_stacks = []
test_control_stacks = []
dirs = sorted(os.listdir(test_root))
dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
for d in dirs:
    print(d)
    labels = np.genfromtxt(os.path.join(test_root, d, 'data_out.csv'), delimiter=',', skip_header=1)
    n_frames = len([f for f in os.listdir(os.path.join(test_root, d)) if 'png' in f])
    frame_stack_np = np.zeros((n_frames, 144, 256, 3))
    for ix in range(n_frames):
        frame_stack_np[ix] = imread(os.path.join(test_root, d, '%06d.png' % ix))
    raw_outputs = np.array([evaluation_model(np.expand_dims(np.expand_dims(img, axis=0), axis=0)) for img in frame_stack_np])[:,0,0,:]
    scaled_outputs = (raw_outputs * output_std) + output_means
    op = os.path.join(output_folder, d)
    if not os.path.exists(op):
        os.mkdir(op)
    np.savetxt(os.path.join(op, 'reference_data.csv'), labels, delimiter=',', header='vx,vy,vz,omega_z')
    np.savetxt(os.path.join(op, 'prediction_data.csv'), scaled_outputs, delimiter=',', header='vx,vy,vz,omega_z')

