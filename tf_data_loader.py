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



def get_output_normalization(root):

    training_output_mean_fn = os.path.join(root, 'stats', 'training_output_means.csv')
    if os.path.exists(training_output_mean_fn):
        print('Loading training data output means from: %s' % training_output_mean_fn)
        output_means = np.genfromtxt(training_output_mean_fn, delimiter=',')
    else:
        output_means = np.zeros(4)

    training_output_std_fn = os.path.join(root, 'stats', 'training_output_stds.csv')
    if os.path.exists(training_output_std_fn):
        print('Loading training data output std from: %s' % training_output_std_fn)
        output_stds = np.genfromtxt(training_output_std_fn, delimiter=',')
    else:
        output_stds = np.ones(4)

    return output_means, output_stds



def load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale):
    file_ending = 'png'

    def sub_to_batch(sub_feature, sub_label):
        sfb = sub_feature.batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip((sfb, slb))
        #return sub.batch(seq_len, drop_remainder=True)

    dirs = os.listdir(root)
    dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
    datasets = []

    output_means, output_stds = get_output_normalization(root)

    for (run_number, d) in enumerate(dirs):
        print('Loading Run %d of %d (%s)' % (run_number, len(dirs), d))
        labels = np.genfromtxt(os.path.join(root, d, 'data_out.csv'), delimiter=',', skip_header=1)

        if labels.shape[1] == 4:
            labels = (labels - output_means) / output_stds
            #labels = labels * label_scale
        elif labels.shape[1] == 5:
            labels = (labels[:, 1:] - output_means) / output_stds
            #labels = labels[:,1:] * label_scale
        else:
            raise Exception('Wrong size of input data (expected 4, got %d' % labels.shape[1])
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        #n_images = len(os.listdir(os.path.join(root, d))) - 1
        n_images = len([fn for fn in os.listdir(os.path.join(root, d)) if file_ending in fn])
        #dataset_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)
        dataset_np = np.empty((n_images, *image_size), dtype=np.uint8)

        for ix in range(n_images):
            #dataset_np[ix] = imread(os.path.join(root, d, '%06d.jpeg' % ix))
            img = imread(os.path.join(root, d, '%06d.%s' % (ix, file_ending)))
            dataset_np[ix] = img[img.shape[0] - image_size[0]:, :, :]

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
        dataset = dataset.window(seq_len, shift=shift, stride=stride, drop_remainder=True).flat_map(sub_to_batch)
        datasets.append(dataset)

    return datasets

def get_dataset_multi(root, image_size, seq_len, shift, stride, validation_ratio, label_scale, extra_data_root=None):
    ds = load_dataset_multi(root, image_size, seq_len, shift, stride, label_scale)
    print('n bags: %d' % len(ds))
    cnt = 0
    for d in ds:
        for (ix, _) in enumerate(d):
            pass
        cnt += ix
    print('n windows: %d' % cnt)


    if extra_data_root is not None:
        ds_extra = load_dataset_multi(extra_data_root, image_size, seq_len, shift, stride, label_scale)
        print('\n\n Loaded Extra Dataset! \n\n')
        print('n extra bags: %d' % len(ds_extra))
        cnt = 0
        for d in ds_extra:
            for (ix, _) in enumerate(d):
                pass
            cnt += ix
        print('n extra windows: %d' % cnt)

    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    val_ix = int(len(ds) * validation_ratio)
    validation_datasets = ds[:val_ix]

    if extra_data_root is not None:
        training_datasets = ds[val_ix:] + ds_extra
        print('Total data length: %d' % len(training_datasets))
    else:
        training_datasets = ds[val_ix:]
    
    validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)

    return training, validation

