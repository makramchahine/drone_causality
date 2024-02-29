import os

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.image import imread
from tqdm import tqdm


def process_dataset(root, label_scale):
    run_dirs = os.listdir(root)  # should be directories named run%03d
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
        sfb = sub_feature["input_image"].batch(seq_len, drop_remainder=True)
        std = sub_feature["input_timedelta"].batch(seq_len, drop_remainder=True)
        slb = sub_label.batch(seq_len, drop_remainder=True)
        return tf.data.Dataset.zip(({"input_image":sfb, "input_timedelta":std}, slb))
        # return sub.batch(seq_len, drop_remainder=True)

    dirs = sorted(os.listdir(root))
    dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
    datasets = []

    output_means, output_stds = get_output_normalization(root)

    for (run_number, d) in tqdm(enumerate(dirs)):
        labels = np.genfromtxt(os.path.join(root, d, 'data_out.csv'), delimiter=',', skip_header=1, dtype=np.float32)
        # timedeltas = np.genfromtxt(os.path.join(root, d, 'timedeltas.csv'), delimiter=',', skip_header=1, dtype=np.float32)
        timedeltas = labels[:, 4]
        # timedeltas = np.ones_like((labels.shape[0], 1))

        if labels.shape[1] == 4:
            labels = (labels - output_means) / output_stds
            # labels = labels * label_scale
        elif labels.shape[1] == 5:
            labels = (labels[:, :4] - output_means) / output_stds
            # labels = labels[:,1:] * label_scale
        else:
            raise Exception('Wrong size of input data (expected 4, got %d' % labels.shape[1])
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        # n_images = len(os.listdir(os.path.join(root, d))) - 1
        n_images = len([fn for fn in os.listdir(os.path.join(root, d)) if file_ending in fn])
        # dataset_np = np.empty((n_images, 256, 256, 3), dtype=np.uint8)
        dataset_np = np.empty((n_images, *image_size), dtype=np.uint8)

        for ix in range(n_images):
            # dataset_np[ix] = imread(os.path.join(root, d, '%06d.jpeg' % ix))
            img = Image.open(os.path.join(root, d, '%06da.%s' % (ix, file_ending))).convert('RGB')
            dataset_np[ix] = img

        images_dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        timedeltas_dataset = tf.data.Dataset.from_tensor_slices(timedeltas)
        # dataset = tf.data.Dataset.zip((images_dataset, timedeltas_dataset, labels_dataset))
        dataset = tf.data.Dataset.zip(({"input_image":images_dataset, "input_timedelta":timedeltas_dataset}, labels_dataset))
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

    # indices = np.arange(len(ds))

    # The RNG used to split the validation data is deterministic here to prevent leakage between validation and training between runs
    # rng_val_split = np.random.default_rng(123)
    # rng_val_split.shuffle(indices)

    val_ix = int(len(ds) * validation_ratio)
    print('\nval_ix: %d\n' % val_ix)
    validation_datasets = ds[:val_ix]

    if extra_data_root is not None:
        training_datasets = ds[val_ix:] + ds_extra
        print('Total data length: %d' % len(training_datasets))
    else:
        training_datasets = ds[val_ix:]

    # if either dataset has length 0, trying to call flat map raises error that return type is wrong
    assert len(training_datasets) > 0 and len(validation_datasets) > 0, f"Training or validation dataset has no points!" \
                                                                        f"Train dataset len: {len(training_datasets)}" \
                                                                        f"Val dataset len: {len(validation_datasets)}"
    training = tf.data.Dataset.from_tensor_slices(training_datasets).flat_map(lambda x: x)
    validation = tf.data.Dataset.from_tensor_slices(validation_datasets).flat_map(lambda x: x)

    return training, validation


def frames_to_array_rnn(root, dirs, image_size, seq_len):
    n_runs = len(dirs)
    run_lengths = [len([fn for fn in os.listdir(os.path.join(root, d)) if 'png' in fn]) for d in dirs]
    print('run lengths: ', sorted(run_lengths))
    max_run_length = max(run_lengths)
    print('Longest Run: %d steps' % max_run_length)
    print('Shortest Run: %d steps' % min(run_lengths))
    print(list(zip(run_lengths, dirs)))

    run_len_threshold = 2048

    assert run_len_threshold % seq_len == 0, 'seq_len must divide run_len_threshold'
    max_bins = run_len_threshold // seq_len
    n_runs_over_thresh = len([r for r in run_lengths if r > run_len_threshold])
    print('N runs over length threshold (will be split):', n_runs_over_thresh)
    total_extra_runs = sum(
        [int(np.floor(r / run_len_threshold)) - 1 for r in [f for f in run_lengths if f > run_len_threshold]])
    print('Total extra runs: %d' % total_extra_runs)
    max_frame_index = [min(int(max(np.floor(r / run_len_threshold), 1)) * run_len_threshold, r) for r in run_lengths]
    print('max_frame_index: ', max_frame_index)

    cur_extra_run = 0
    n_batches = min(int(np.ceil(max_run_length / seq_len)), run_len_threshold)
    data = np.zeros((n_batches, n_runs + total_extra_runs, seq_len, *image_size), dtype=np.uint8)
    full_batch_size = n_runs + total_extra_runs
    labels = np.zeros((n_batches, full_batch_size, seq_len, 4))
    print('Data shape: ', data.shape)
    for (ix, dname) in enumerate(dirs):
        print('Loading directory %d of %d (%s)' % (ix, n_runs, dname))
        label_raw = np.genfromtxt(os.path.join(root, dname, 'data_out.csv'), delimiter=',', skip_header=1)
        # for jx in range(run_lengths[ix]):
        for jx in range(max_frame_index[ix]):
            if jx == run_len_threshold:
                cur_extra_run += 1
            img = Image.open(os.path.join(root, dname, '%06d.png' % jx))
            bin_number = int(np.floor(jx / seq_len)) % max_bins
            frame_in_bin = jx % seq_len
            data[bin_number, ix + cur_extra_run, frame_in_bin] = img
            labels[bin_number, ix + cur_extra_run, frame_in_bin] = label_raw[jx]
            # assert len(label_raw)-1 == jx, '%d, %d' % (len(label_raw), jx)
    return (data, labels), full_batch_size


def load_dataset_rnn(root, image_size, seq_len, validation_ratio):
    dirs = sorted(os.listdir(root))
    dirs = [d for d in dirs if 'cached' not in d and 'stats' not in d]
    output_means, output_stds = get_output_normalization(root)

    rng_val_split = np.random.default_rng(5432)
    rng_val_split.shuffle(dirs)
    val_ix = int(len(dirs) * validation_ratio)
    validation_dirs = dirs[:val_ix]
    training_dirs = dirs[val_ix:]

    training_data, batch_size = frames_to_array_rnn(root, training_dirs, image_size, seq_len)
    validation_data, validation_batch_size = frames_to_array_rnn(root, validation_dirs, image_size, seq_len)

    return batch_size, validation_batch_size, training_data, validation_data
