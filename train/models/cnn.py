import tensorflow as tf


def create(num_output):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_output),
    ])

    return model
