import os

import kerasncp as kncp
from kerasncp.tf import LTCCell
from tensorflow import keras

from node_cell import *
from tf_cfc import CfcCell, MixedCfcCell

DROPOUT = 0.1

DEFAULT_CONFIG = {
    "clipnorm": 1,
    "size": 128,
    "backbone_activation": "tanh",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 256,
    "backbone_layers": 2,
    "weight_decay": 1e-06,
    "use_mixed": False,
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_lstm_model(rnn_sizes,
                        seq_len,
                        image_shape,
                        do_normalization,
                        do_augmentation,
                        data,
                        dropout=0.1,
                        recurrent_dropout=0.1,
                        augmentation_params=None,
                        rnn_stateful=False,
                        batch_size=None
                        ):
    lstm_model = generate_network_trunk(seq_len,
                                        image_shape,
                                        do_normalization,
                                        do_augmentation,
                                        data,
                                        augmentation_params,
                                        rnn_stateful=rnn_stateful,
                                        batch_size=batch_size
                                        )

    # print(lstm_model.layers[-1].output_shape)
    # print(batch_size, seq_len, lstm_model.layers[-1].output_shape[-1])

    for (ix, s) in enumerate(rnn_sizes):
        lstm_model.add(keras.layers.LSTM(s,
                                         batch_input_shape=(batch_size,
                                                            seq_len,
                                                            lstm_model.layers[-1].output_shape[-1]
                                                            ),
                                         return_sequences=True,
                                         stateful=rnn_stateful,
                                         dropout=dropout,
                                         recurrent_dropout=recurrent_dropout))
        # if ix < len(rnn_sizes) - 1:
        #    lstm_model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.1)))

    # lstm_model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.05)))
    lstm_model.add(keras.layers.Dense(units=4, activation='linear'))
    # print(lstm_model.summary())
    return lstm_model


def generate_ncp_model(seq_len,
                       image_shape,
                       do_normalization,
                       do_augmentation,
                       data,
                       augmentation_params=None,
                       rnn_stateful=False,
                       batch_size=None,
                       seed=2222
                       ):
    ncp_model = generate_network_trunk(seq_len,
                                       image_shape,
                                       do_normalization,
                                       do_augmentation,
                                       data, augmentation_params,
                                       rnn_stateful=rnn_stateful,
                                       batch_size=batch_size)

    # Setup the network
    wiring = kncp.wirings.NCP(
        inter_neurons=18,  # Number of inter neurons
        command_neurons=12,  # Number of command neurons
        motor_neurons=4,  # Number of motor neurons
        sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming syanpses has each motor neuron,
        seed=seed,  # random seed to generate connections between nodes
    )

    rnnCell = LTCCell(wiring)
    ncp_model.add(keras.layers.RNN(rnnCell,
                                   batch_input_shape=(batch_size,
                                                      seq_len,
                                                      ncp_model.layers[-1].output_shape[-1]),
                                   return_sequences=True)
                  )

    return ncp_model


def generate_ctrnn_model(rnn_sizes,
                         seq_len,
                         image_shape,
                         do_normalization,
                         do_augmentation,
                         data,
                         augmentation_params=None,
                         rnn_stateful=False,
                         batch_size=None,
                         ct_network_type='ctrnn',
                         config=DEFAULT_CONFIG
                         ):
    ctrnn_model = generate_network_trunk(seq_len, image_shape,
                                         do_normalization,
                                         do_augmentation,
                                         data,
                                         augmentation_params,
                                         rnn_stateful=rnn_stateful,
                                         batch_size=batch_size
                                         )

    for (ix, s) in enumerate(rnn_sizes):
        if ct_network_type == 'ctrnn':
            Cell = CTRNNCell(units=s, method='dopri5')
        elif ct_network_type == "node":
            Cell = CTRNNCell(units=s, method="dopri5", tau=0)
        elif ct_network_type == "mmrnn":
            Cell = mmRNN(units=s)
        elif ct_network_type == "ctgru":
            Cell = CTGRU(units=s)
        elif ct_network_type == "vanilla":
            Cell = VanillaRNN(units=s)
        elif ct_network_type == "bidirect":
            Cell = BidirectionalRNN(units=s)
        elif ct_network_type == "grud":
            Cell = GRUD(units=s)
        elif ct_network_type == "phased":
            Cell = PhasedLSTM(units=s)
        elif ct_network_type == "gruode":
            Cell = GRUODE(units=s)
        elif ct_network_type == "hawk":
            Cell = HawkLSTMCell(units=s)
        elif ct_network_type == "ltc":
            Cell = LTCCell(units=s)
        elif ct_network_type == "cfc":
            Cell = CfcCell(units=s, hparams=config)
        elif ct_network_type == "mixedcfc":
            Cell = MixedCfcCell(units=s, hparams=config)
        else:
            raise ValueError("Unknown model type '{}'".format(ct_network_type))
        ctrnn_model.add(
            keras.layers.RNN(Cell,
                             batch_input_shape=(batch_size, seq_len,
                                                ctrnn_model.layers[-1].output_shape[-1]),
                             return_sequences=True,
                             stateful=rnn_stateful,
                             time_major=False)
        )

    ctrnn_model.add(keras.layers.Dense(units=4, activation='linear'))

    return ctrnn_model


def generate_convolutional_layers(model):
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu')))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(3, 3), activation='relu')))

    return model


def generate_normalization_layers(model, data=None):
    model.add(keras.layers.experimental.preprocessing.Rescaling(1. / 255))

    if data is not None:

        normalization_layer = keras.layers.TimeDistributed(
            keras.layers.experimental.preprocessing.Normalization())

        normalization_layer.adapt(data)
        # DATA is all the data after loading into RAM (single array)
    else:
        # normalization_layer = keras.layers.TimeDistributed(
        #   keras.layers.experimental.preprocessing.Normalization(mean=0.5, variance=0.03))

        normalization_layer = keras.layers.TimeDistributed(
            keras.layers.experimental.preprocessing.Normalization(
                mean=[0.41718618, 0.48529191, 0.38133072],
                variance=[0.19504249, 0.18745404, 0.20891384]))

    model.add(normalization_layer)

    return model


def generate_augmentation_layers(model, augmentation_params):
    # translate -> rotate -> zoom
    trans = augmentation_params['translation']
    rot = augmentation_params['rotation']
    zoom = augmentation_params['zoom']

    model.add(keras.layers.TimeDistributed(
        keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=trans, width_factor=trans)))

    model.add(keras.layers.TimeDistributed(
        keras.layers.experimental.preprocessing.RandomRotation(rot)))

    model.add(keras.layers.TimeDistributed(
        keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=zoom, width_factor=zoom)))

    return model


def generate_network_trunk(seq_len,
                           image_shape,
                           do_normalization,
                           do_augmentation,
                           data=None,
                           augmentation_params=None,
                           rnn_stateful=False,
                           batch_size=None):
    # model.add(keras.layers.InputLayer(input_shape=(seq_len, *image_shape), batch_size=batch_size))
    # model.add(keras.layers.InputLayer(batch_input_shape=(batch_size, seq_len, *image_shape)))
    # inputs = keras.Input(batch_input_shape=(batch_size,seq_len,*image_shape))
    # rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
    # normalization_layer = keras.layers.experimental.preprocessing.Normalization(mean=[0.41718618, 0.48529191, 0.38133072], variance=[0.19504249, 0.18745404, 0.20891384])
    # inputs = rescaling_layer(inputs)
    # inputs = normalization_layer(inputs)
    # #inputs =keras.layers.experimental.preprocessing.Normalization(mean=[0.41718618, 0.48529191, 0.38133072], variance=[0.19504249, 0.18745404, 0.20891384])(inputs)
    # #model.add(keras.layers.InputLayer(batch_input_shape=(batch_size, seq_len, 144, 256, 3)))
    # #model.add(keras.Input(batch_shape=(6, seq_len, *image_shape)))

    # #if do_normalization:
    # #    model = generate_normalization_layers(model)
    # #
    # # if do_augmentation:
    # #     model = generate_augmentation_layers(model, augmentation_params)

    # model_vgg = keras.applications.VGG16(include_top=False, weights='imagenet',input_shape=image_shape)
    # model_vgg.trainable = False
    # intermediate_model = keras.Model(inputs=model_vgg.input, outputs=model_vgg.get_layer('block5_pool').output)
    # time_dist_layers = keras.layers.TimellayerDistributed(intermediate_model)(inputs)
    # my_time_model = keras.Model(inputs=inputs, outputs=time_dist_layers)

    # model = keras.models.Sequential()

    # model.add(my_time_model)

    # #model = generate_convolutional_layers(model)
    # print(model.summary())
    # model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    # model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1024,   activation='linear')))
    # model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=DROPOUT)))

    # x = model.layers[0].output
    # x = keras.layers.experimental.preprocessing.Rescaling(1./255)(x)

    model_vgg = keras.applications.VGG16(include_top=False,
                                         weights='imagenet',
                                         input_shape=image_shape)

    layers = [l for l in model_vgg.layers]

    inputs = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape))

    rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    # normalization_layer = keras.layers.experimental.preprocessing.Normalization(
    #     mean=[0.41718618, 0.48529191, 0.38133072],
    #     variance=[0.19504249, 0.18745404, 0.20891384])

    normalization_layer = keras.layers.experimental.preprocessing.Normalization(
        mean=[0.41718618, 0.48529191, 0.38133072],
        variance=[.057, .05, .061])

    layers[0] = inputs

    # x = layers[0] # test network without normalization

    x = rescaling_layer(layers[0])
    x = keras.layers.TimeDistributed(normalization_layer)(x)

    my_input_model = keras.Model(inputs=inputs, outputs=x)

    for i in range(len(layers)):
        if i == 0:
            continue
        else:
            layers[i].trainable = False
            layers[i] = keras.layers.TimeDistributed(layers[i])
            x = layers[i](x)

    my_time_model = keras.Model(inputs=inputs, outputs=x)

    model = keras.models.Sequential()

    # model.add(my_time_model)
    model.add(my_input_model)

    # Conv Layers
    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')))

    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu')))

    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')))

    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')))

    model.add(keras.layers.TimeDistributed(
        keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu')))

    # fully connected layers

    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=128, activation='linear')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=DROPOUT)))

    # print(model.summary())

    return model


# function taken from https://github.com/GoldenZephyr/rosetta_drone/blob/main/rnn_control/src/rnn_control_node.py
def load_model_from_weights(model_name: str, checkpoint_name: str):
    # make sure checkpoint includes script dir so script can be run from any file
    checkpoint_path = os.path.join(SCRIPT_DIR, checkpoint_name)
    RNN_SIZE = 128
    IMAGE_SHAPE = (144, 256, 3)
    inputs = keras.Input(shape=IMAGE_SHAPE)
    # normalization layer unssupported by version of tensorflow on drone. Data instead normalized in callback
    if model_name == "ncp_old":
        # old ncp cnn
        x = keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(3, 3), activation='relu')(inputs)
        x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
        x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    else:
        # new ncp cnn
        x = keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(inputs)
        x = keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
        x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    # fully connected layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    DROPOUT = 0.0
    pre_recurrent_layer = keras.layers.Dropout(rate=DROPOUT)(x)

    if model_name.startswith("ncp"):
        assert model_name == "ncp" or model_name == "ncp_old", \
            f"Only legal ncp model names are 'ncp' and 'ncp_old', got {model_name}"
        wiring = kncp.wirings.NCP(
            inter_neurons=18,  # Number of inter neurons
            command_neurons=12,  # Number of command neurons
            motor_neurons=4,  # Number of motor neurons
            sensory_fanout=6,  # How many outgoing synapses has each sensory neuron
            inter_fanout=4,  # How many outgoing synapses has each inter neuron
            recurrent_command_synapses=4,  # Now many recurrent synapses are in the
            # command neuron layer
            motor_fanin=6,  # How many incoming synapses has each motor neuron
        )
        rnn_cell = LTCCell(wiring, ode_unfolds=6)
        inputs_state = tf.keras.Input(shape=(rnn_cell.state_size,))

        motor_out, output_states = rnn_cell(pre_recurrent_layer, inputs_state)
        single_step_model = tf.keras.Model([inputs, inputs_state], [motor_out, output_states])

        single_step_model.load_weights(checkpoint_path)
        hidden_state = (tf.zeros((1, rnn_cell.state_size)))
    elif model_name == 'lstm':
        rnn_cell = tf.keras.layers.LSTMCell(RNN_SIZE)
        c_state = tf.keras.Input(shape=(rnn_cell.state_size[0]))
        h_state = tf.keras.Input(shape=(rnn_cell.state_size[1]))

        output, [next_c, next_h] = rnn_cell(pre_recurrent_layer, [c_state, h_state])
        output = tf.keras.layers.Dense(units=4, activation='linear')(output)
        single_step_model = tf.keras.Model([inputs, c_state, h_state], [next_c, next_h, output])

        single_step_model.load_weights(checkpoint_path)
        # hidden c, hidden h
        hidden_state = (tf.zeros((1, rnn_cell.state_size[0])), tf.zeros((1, rnn_cell.state_size[1])))
    elif model_name == 'mixedcfc':
        CONFIG = {
            "clipnorm": 1,
            "size": 128,
            "backbone_activation": "silu",
            "backbone_dr": 0.1,
            "forget_bias": 1.6,
            "backbone_units": 128,
            "backbone_layers": 1,
            "weight_decay": 1e-06,
            "use_mixed": True,
        }

        rnn_cell = MixedCfcCell(units=RNN_SIZE, hparams=CONFIG)

        c_state = tf.keras.Input(shape=(rnn_cell.state_size[0]))
        h_state = tf.keras.Input(shape=(rnn_cell.state_size[1]))

        output, [next_c, next_h] = rnn_cell(pre_recurrent_layer, [c_state, h_state])
        output = tf.keras.layers.Dense(units=4, activation='linear')(output)
        single_step_model = tf.keras.Model([inputs, c_state, h_state], [next_c, next_h, output])

        single_step_model.load_weights(checkpoint_path)
        # hidden c, hidden h
        hidden_state = (tf.zeros((1, rnn_cell.state_size[0])), tf.zeros((1, rnn_cell.state_size[1])))
    else:
        raise ValueError(f"Illegal model name {model_name}")

    return single_step_model, hidden_state
