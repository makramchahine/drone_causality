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



def load_dataset_multi(root, seq_len, shift, stride, label_scale):

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))
        #return sub.batch(seq_len, drop_remainder=True)

    dirs = os.listdir(root)[:5]
    datasets = []
    for d in dirs:
        labels = np.genfromtxt(os.path.join(root, d, 'data_out.csv'), delimiter=',', skip_header=1)
        labels = labels[:,1:] * label_scale
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        n_images = len(os.listdir(os.path.join(root, d))) - 1
        dataset_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)
        for ix in range(n_images):
            dataset_np[ix] = imread(os.path.join(root, d, '%06d.jpeg' % ix))

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
        dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
        datasets.append(dataset)

    return datasets

def get_dataset_multi(root, seq_len, shift, stride, validation_ratio, label_scale):
    ds = load_dataset_multi(root, seq_len, shift, stride, label_scale)

    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    val_ix = int(len(ds) * 0.2)
    validation_datasets = ds[:val_ix]
    training_datasets = ds[val_ix:]
    
    validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)

    return training, validation

