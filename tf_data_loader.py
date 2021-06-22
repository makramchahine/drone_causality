import tensorflow as tf
import numpy as np
import os
from matplotlib.image import imread

def process_dataset(root, label_scale):
    run_dirs = os.listdir(root) # should be directories named run%03d
    n = len(run_dirs)
    dataset = np.empty((n, 64, 256, 256, 3), dtype=np.uint8)
    labels = np.empty((n, 64, 4))
    for (dx, d) in enumerate(run_dirs):
        for i in range(len(os.listdir(os.path.join(root, d))) - 1):
            dataset[dx, i] = imread(os.path.join(root, d, '%03d.jpg' % i))
        labels[dx] = np.load(os.path.join(root, d, 'vectors.npy')) * label_scale

    return dataset, labels


def load_dataset(data_root, label_scale=1):
    training_root = os.path.join(data_root, 'training')
    validation_root = os.path.join(data_root, 'validation')

    training_np = process_dataset(training_root, label_scale)
    validation_np = process_dataset(validation_root, label_scale)
    return training_np, validation_np
