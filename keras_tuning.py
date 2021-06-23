#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from tf_data_loader import load_dataset

def build_model(hp):
    model = keras.Sequential()

    model.add(keras.Input(shape=(64, 256, 256, 3)))
    model.add(keras.layers.experimental.preprocessing.Rescaling(1./255))
    normalization_layer = keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.Normalization(mean=0.5, variance=0.03))
    model.add(normalization_layer)

    trans = hp.Float('translation_augmentation', 0.0, 0.2)
    rot = hp.Float('rotation_augmentation', 0.0, 0.2)
    zoom = hp.Float('zoom_augmentation', 0.0, 0.2)
    model.add(keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.RandomTranslation(height_factor=trans, width_factor=trans)))
    model.add(keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.RandomRotation(rot)))
    model.add(keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.RandomZoom(height_factor=zoom, width_factor=zoom)))


    #f1 = hp.Int('filtersize_1', 8, 64, step=4)
    #f2 = hp.Int('filtersize_2', 8, 64, step=4)
    #f3 = hp.Int('filtersize_3', 8, 64, step=4)
    #f4 = hp.Int('filtersize_4', 8, 64, step=4)
    f1 = 24
    f2 = 36
    f3 = 48
    f4 = 64

    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f1, kernel_size=(5,5), strides=(2,2), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f1, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f1, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f1, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f2, kernel_size=(5,5), strides=(2,2), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f2, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f2, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f2, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f3, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f3, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f3, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f3, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f4, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f4, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f4, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=f4, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8,  kernel_size=(3,3), strides=(3,3), activation='relu')))

    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=hp.Float('dropout_rate', 0, 0.5))))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64, activation='linear')))

    lstm_units = hp.Int('rnn_size', 32, 128, step=12)
    model.add(keras.layers.LSTM(units=lstm_units, return_sequences=True))
    model.add(keras.layers.Dense(units=4, activation='linear'))
    
    opt_string = hp.Choice('optimizer', values=['sgd', 'adam'])
    lr = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0005, 0.0001])
    if opt_string == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    elif opt_string == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)


    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

tuner = kt.BayesianOptimization(build_model, objective='val_loss', max_trials=100, executions_per_trial=1, overwrite=True, directory='tuner_tests', project_name='minicity_ncp')
print(tuner.search_space_summary())


training_np, validation_np = load_dataset('/home/aaron/drone_ncp/minicity_bags/videogen_data_test', 5)

training_dataset = tf.data.Dataset.from_tensor_slices(training_np).shuffle(100).batch(8)
validation_dataset = tf.data.Dataset.from_tensor_slices(validation_np).batch(8)

#tuner.search(training_np[0], training_np[1], epochs=5, validation_data=validation_np)
tuner.search(x=training_dataset, epochs=8, validation_data=validation_dataset)
