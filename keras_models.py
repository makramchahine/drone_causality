from tensorflow import keras
from kerasncp.tf import LTCCell
import kerasncp as kncp


def generate_lstm_model(rnn_size, seq_len, image_shape, do_normalization, do_augmentation, data, augmentation_params=None):
    lstm_model = generate_network_trunk(seq_len, image_shape, do_normalization, do_augmentation, data, augmentation_params)
    lstm_model.add(keras.layers.LSTM(units=rnn_size, return_sequences=True))
    #lstm_model.add(keras.layers.LSTM(units=rnn_size, return_sequences=True))
    #lstm_model.add(keras.layers.LSTM(units=rnn_size, return_sequences=True))
    lstm_model.add(keras.layers.Dense(units=4, activation='linear'))
    return lstm_model


def generate_ncp_model(seq_len, image_shape, do_normalization, do_augmentation, data, augmentation_params=None):

    ncp_model = generate_network_trunk(seq_len, image_shape, do_normalization, do_augmentation, data, augmentation_params)

    # Setup the network
    wiring = kncp.wirings.NCP(
        inter_neurons=18,   # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        #motor_neurons=3,    # Number of motor neurons
        motor_neurons=4,    # Number of motor neurons
        sensory_fanout=6,   # How many outgoing synapses has each sensory neuron
        inter_fanout=4,     # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,   # Now many recurrent synapses are in the
                                        # command neuron layer
        motor_fanin=6,      # How many incoming syanpses has each motor neuron
    )

    rnnCell = LTCCell(wiring)

    ncp_model.add(keras.layers.RNN(rnnCell, return_sequences=True))
    return ncp_model
    


def generate_convolutional_layers(model):
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8, kernel_size=(2,2), strides=(2,2), activation='relu')))

    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(2,2), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), activation='relu')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8,  kernel_size=(3,3), strides=(3,3), activation='relu')))

    return model


def generate_normalization_layers(model, data=None):
    model.add(keras.layers.experimental.preprocessing.Rescaling(1./255))

    if data is not None:
        normalization_layer = keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.Normalization())
        normalization_layer.adapt(data) # DATA is all the data after loading into RAM (single array)
    else:
        normalization_layer = keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.Normalization(mean=0.5, variance=0.03))
    model.add(normalization_layer)
    return model


def generate_augmentation_layers(model, augmentation_params):
    # translate -> rotate -> zoom
    trans = augmentation_params['translation']
    rot = augmentation_params['rotation']
    zoom = augmentation_params['zoom']
    model.add(keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.RandomTranslation(height_factor=trans, width_factor=trans)))
    model.add(keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.RandomRotation(rot)))
    model.add(keras.layers.TimeDistributed(keras.layers.experimental.preprocessing.RandomZoom(height_factor=zoom, width_factor=zoom)))
    return model

def generate_network_trunk(seq_len, image_shape, do_normalization, do_augmentation, data=None, augmentation_params=None):

    model = keras.models.Sequential()
    model.add(keras.Input(shape=(seq_len, *image_shape)))

    if do_normalization:
        model = generate_normalization_layers(model)

    if do_augmentation:
        model = generate_augmentation_layers(model, augmentation_params)

    model = generate_convolutional_layers(model)
    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64,   activation='linear')))

    return model

