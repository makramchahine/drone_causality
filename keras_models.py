import os
from typing import Iterable, Dict

import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow import keras, concat

from node_cell import *
from tf_cfc import CfcCell, MixedCfcCell
from tf_cfc import LTCCell as CFCLTCCell

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

DROPOUT = 0.1
DEFAULT_CFC_CONFIG = {
    "clipnorm": 1,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-06
}
DEFAULT_NCP_SEED = 22222
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Shapes for generate_*_model:
# if single_step, input is tuple of image input (batch [usually 1], h, w, c), and hiddens (batch, hidden_dim)
# if not single step, is just sequence of images with shape (batch, seq_len, h, w, c) otherwise
# output is control output for not single step, for single step it is list of tensors where first element is control
# output and other outputs are any hidden states required
# if single_step, control output is (batch, 4), otherwise (batch, seq_len, 4)
# if single_step, hidden outputs typically have shape (batch, hidden_dimension)

def generate_lstm_model(
        rnn_sizes,
        seq_len,
        image_shape,
        dropout=0.1,
        recurrent_dropout=0.1,
        rnn_stateful=False,
        batch_size=None,
        augmentation_params=None,
        single_step: bool = False,
        no_norm_layer: bool = False,
):
    inputs_image, inputs_value, x = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )

    # vars for single step model
    c_inputs = []
    h_inputs = []
    c_outputs = []
    h_outputs = []
    for (ix, s) in enumerate(rnn_sizes):
        if single_step:
            rnn_cell = tf.keras.layers.LSTMCell(s)
            # keep track of input for each layer of rnn
            c_input = tf.keras.Input(shape=(rnn_cell.state_size[0]))
            h_input = tf.keras.Input(shape=(rnn_cell.state_size[1]))

            x, [c_state, h_state] = rnn_cell(x, [c_input, h_input])
            c_inputs.append(c_input)
            h_inputs.append(h_input)
            c_outputs.append(c_state)
            h_outputs.append(h_state)
        else:
            x = keras.layers.LSTM(
                s,
                batch_input_shape=(
                    batch_size,
                    seq_len,
                    x.shape[-1]
                ),
                return_sequences=True,
                stateful=rnn_stateful,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            )(x)

    x = keras.layers.Dense(units=4, activation='linear')(x)
    if single_step:
        lstm_model = keras.Model([inputs_image, inputs_value, *c_inputs, *h_inputs], [x, *c_outputs, *h_outputs])
    else:
        lstm_model = keras.Model([inputs_image, inputs_value], [x])

    return lstm_model


def generate_ncp_model(seq_len,
                       image_shape,
                       augmentation_params=None,
                       batch_size=None,
                       seed=DEFAULT_NCP_SEED,
                       single_step: bool = False,
                       no_norm_layer: bool = False,
                       ):
    inputs_image, inputs_value, x = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )

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

    rnn_cell = LTCCell(wiring)

    if single_step:
        inputs_state = tf.keras.Input(shape=(rnn_cell.state_size,))
        # wrap output states in list since want output to just be ndarray, not list of 1 el ndarray
        motor_out, [output_states] = rnn_cell(x, inputs_state)
        ncp_model = keras.Model([inputs_image, inputs_value, inputs_state], [motor_out, output_states])
    else:
        x = keras.layers.RNN(rnn_cell,
                             batch_input_shape=(batch_size,
                                                seq_len,
                                                x.shape[-1]),
                             return_sequences=True)(x)

        ncp_model = keras.Model([inputs_image, inputs_value], [x])

    return ncp_model


def generate_ctrnn_model(rnn_sizes,
                         seq_len,
                         image_shape,
                         augmentation_params=None,
                         rnn_stateful=False,
                         batch_size=None,
                         ct_network_type='ctrnn',
                         config=DEFAULT_CFC_CONFIG,
                         single_step: bool = False,
                         no_norm_layer: bool = False,
                         **kwargs,
                         ):
    inputs_image, inputs_image2, inputs_value, x, x2 = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )

    # vars for single step model
    all_hidden_inputs = []  # shape: num layers x num hidden x hidden size
    all_hidden_outputs = []
    for (ix, s) in enumerate(rnn_sizes):
        if ct_network_type == 'ctrnn':
            rnn_cell = CTRNNCell(units=s, method='dopri5')
        elif ct_network_type == "node":
            rnn_cell = CTRNNCell(units=s, method="dopri5", tau=0)
        elif ct_network_type == "mmrnn":
            rnn_cell = mmRNN(units=s)
        elif ct_network_type == "ctgru":
            rnn_cell = CTGRU(units=s)
        elif ct_network_type == "vanilla":
            rnn_cell = VanillaRNN(units=s)
        elif ct_network_type == "bidirect":
            rnn_cell = BidirectionalRNN(units=s)
        elif ct_network_type == "grud":
            rnn_cell = GRUD(units=s)
        elif ct_network_type == "phased":
            rnn_cell = PhasedLSTM(units=s)
        elif ct_network_type == "gruode":
            rnn_cell = GRUODE(units=s)
        elif ct_network_type == "hawk":
            rnn_cell = HawkLSTMCell(units=s)
        elif ct_network_type == "ltc":
            rnn_cell = CFCLTCCell(units=s)
        elif ct_network_type == "cfc":
            rnn_cell = CfcCell(units=s, hparams=config)
        elif ct_network_type == "mixedcfc":
            rnn_cell = MixedCfcCell(units=s, hparams=config)
        elif ct_network_type == "wiredcfccell":
            wiring = kncp.wirings.NCP(
                inter_neurons=18+4,  # Number of inter neurons
                command_neurons=12+4,  # Number of command neurons
                motor_neurons=4+4,  # Number of motor neurons
                sensory_fanout=6+4,  # How many outgoing synapses has each sensory neuron
                inter_fanout=4+4,  # How many outgoing synapses has each inter neuron
                recurrent_command_synapses=4+4,  # Now many recurrent synapses are in the command neuron layer
                motor_fanin=6+4,  # How many incoming syanpses has each motor neuron,
                seed=kwargs.get("wiredcfc_seed", DEFAULT_NCP_SEED),  # random seed to generate connections between nodes
            )
            rnn_cell = WiredCfcCell(wiring=wiring, mode="default")
        else:
            raise ValueError("Unknown model type '{}'".format(ct_network_type))

        if single_step:
            # keep track of input for each layer of rnn
            if isinstance(rnn_cell.state_size, int):
                # only 1 hidden state
                hidden_inputs = [tf.keras.Input(shape=rnn_cell.state_size)]
                x, hidden = rnn_cell(x, hidden_inputs)  # assume hidden is list of length 1 with tensor
                all_hidden_inputs.extend(hidden_inputs)
                all_hidden_outputs.extend(hidden)
            else:
                # multiple hiddens
                hidden_inputs = [tf.keras.Input(shape=size) for size in rnn_cell.state_size]
                x, hidden_outputs = rnn_cell(x, hidden_inputs)
                all_hidden_inputs.extend(hidden_inputs)
                all_hidden_outputs.extend(hidden_outputs)

        else:
            rnn = keras.layers.RNN(rnn_cell,
                                 batch_input_shape=(batch_size, seq_len,
                                                    x.shape[-1]),
                                 return_sequences=True,
                                 stateful=rnn_stateful,
                                 time_major=False)
            x = rnn(x)

    v1 = x[:, 0:4] if single_step else x[:, :, 0:4]
    comm = keras.layers.Dense(units=128, activation='linear')(x)

    rnn2_input = keras.layers.concatenate([comm, x2], axis=-1)
    rnn2_input = keras.layers.Dense(units=128, activation='linear')(rnn2_input)
    
    twornn = True
    if twornn:
        rnn_cell2 = WiredCfcCell(wiring=wiring, mode="default")
        rnn2 = keras.layers.RNN(rnn_cell2,
                batch_input_shape=(batch_size, seq_len,
                                x.shape[-1]),
                return_sequences=True,
                stateful=rnn_stateful,
                time_major=False)
    else:
        rnn_cell2 = rnn_cell
        if not single_step:
            rnn2 = rnn

    if single_step:
        if isinstance(rnn_cell2.state_size, int):
            # only 1 hidden state
            hidden_inputs = [tf.keras.Input(shape=rnn_cell2.state_size)]
            x, hidden = rnn_cell2(rnn2_input, hidden_inputs)  # assume hidden is list of length 1 with tensor
            all_hidden_inputs.extend(hidden_inputs)
            all_hidden_outputs.extend(hidden)
        else:
            # multiple hiddens
            hidden_inputs = [tf.keras.Input(shape=size) for size in rnn_cell2.state_size]
            x, hidden_outputs = rnn_cell2(rnn2_input, hidden_inputs)
            all_hidden_inputs.extend(hidden_inputs)
            all_hidden_outputs.extend(hidden_outputs)
    else:
        x = rnn2(rnn2_input)
        
    v2 = x[:, 0:4] if single_step else x[:, :, 0:4]

    v = keras.layers.concatenate([v1, v2], axis=-1)

    if single_step:
        ctrnn_model = keras.Model([inputs_image, inputs_image2, inputs_value, *all_hidden_inputs], [v, comm, *all_hidden_outputs])
    else:
        ctrnn_model = keras.Model([inputs_image, inputs_image2, inputs_value], [v, comm])

    return ctrnn_model


def generate_tcn_model(
        nb_filters: int,
        kernel_size: int,
        dilations: Iterable[int],
        seq_len,
        image_shape,
        dropout=0.1,
        augmentation_params=None,
        batch_size=None,
        single_step: bool = False,
        no_norm_layer: bool = False,
):
    """
    Temporal Convolutional Network as recurrent architecture
    https://link.springer.com/content/pdf/10.1007/978-3-319-49409-8_7.pdf

    Note that because the TCN has no hidden state (operates on entire sequence), the hidden state returned by this model
    is the entire sequence of CNN embeddings, which is automatically outputted (and truncated to the receptive field
    size of the TCN). The "hidden state" therefore has an additional dimension in single step mode, and is of shape
    (batch, seq, nb_units)
    """
    from tcn import TCN  # only import keras_tcn library if needed

    inputs_image, inputs_value, x = generate_network_trunk(
        seq_len,
        image_shape,
        augmentation_params=augmentation_params,
        batch_size=batch_size,
        single_step=single_step,
        no_norm_layer=no_norm_layer,
    )

    # Setup TCN
    rnn_cell = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        padding="causal",
        use_skip_connections=True,
        dropout_rate=dropout,
        return_sequences=not single_step,
        # following 3 params could be adjustable
        use_batch_norm=False,
        use_layer_norm=True,
        use_weight_norm=False,
    )

    # vars for single step
    inputs_sequence, combined_sequence = None, None
    if single_step:
        # current x shape: (batch [None=1], num_units)
        # append current embedding to previous embedding sequence (embedding dim = x.shape[-1])
        inputs_sequence = tf.keras.Input(shape=(None, x.shape[-1]))  # None is seq len, assume batch size 1 prepended
        x = tf.expand_dims(x, axis=1)  # add seq_len dim to x
        combined_sequence = tf.concat((inputs_sequence, x), axis=1)  # add x to end of sequence, shape: batch, seq, unit
        # don't keep entries farther back than receptive field length
        receptive_slicing_layer = keras.layers.Lambda(lambda y: y[:, :rnn_cell.receptive_field, :],
                                                      name="receptive_slice")
        combined_sequence = receptive_slicing_layer(combined_sequence)
        x = rnn_cell(combined_sequence)
        # output x shape: (batch, nb_filters)
    else:
        # current x shape (batch, seq, num_units)
        x = rnn_cell(x)
        # output x shape: (batch, seq, nb_filters)

    # reduce dims of control signal to size 4
    x = keras.layers.Dense(units=4, activation='linear')(x)

    if single_step:
        tcn_model = keras.Model([inputs_image, inputs_value, inputs_sequence], [x, combined_sequence])
    else:
        tcn_model = keras.Model([inputs_image, inputs_value], [x])

    return tcn_model


def generate_augmentation_layers(x, augmentation_params: Dict, single_step: bool):
    # translate -> rotate -> zoom -> noise
    trans = augmentation_params.get('translation', None)
    rot = augmentation_params.get('rotation', None)
    zoom = augmentation_params.get('zoom', None)
    noise = augmentation_params.get('noise', None)

    if trans is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=trans, width_factor=trans), single_step)(x)

    if rot is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomRotation(rot), single_step)(x)

    if zoom is not None:
        x = wrap_time(keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=zoom, width_factor=zoom), single_step)(x)

    if noise:
        x = wrap_time(keras.layers.GaussianNoise(stddev=noise), single_step)(x)

    return x


def generate_normalization_layers(x, single_step: bool):
    rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    normalization_layer = keras.layers.experimental.preprocessing.Normalization(
        mean=[0.41718618, 0.48529191, 0.38133072],
        variance=[.057, .05, .061])

    x = rescaling_layer(x)
    x = wrap_time(normalization_layer, single_step)(x)
    return x


def wrap_time(layer, single_step: bool):
    """
    Helper function that wraps layer in a timedistributed or not depending on the arguments of this function
    """
    if not single_step:
        return keras.layers.TimeDistributed(layer)
    else:
        return layer


def generate_network_trunk(seq_len,
                           image_shape,
                           augmentation_params: Dict = None,
                           batch_size=None,
                           single_step: bool = False,
                           no_norm_layer: bool = False, ):
    """
    Generates CNN image processing backbone used in all recurrent models. Uses Keras.Functional API

    returns input to be used in Keras.Model and x, a tensor that represents the output of the network that has shape
    (batch [None], seq_len, num_units) if single step is false and (batch [None], num_units) if single step is true.
    Input has shape (batch, h, w, c) if single step is True and (batch, seq, h, w, c) otherwise

    """

    if single_step:
        inputs_image = keras.Input(shape=image_shape, name="input_image")
        inputs_image2 = keras.Input(shape=image_shape, name="input_image2")
        inputs_value = keras.Input(shape=(2,), name="input_vector")
    else:
        inputs_image = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape), name="input_image")
        inputs_image = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape), name="input_image2")
        inputs_value = keras.Input(batch_input_shape=(batch_size, seq_len, 2), name="input_vector")

    xi = inputs_image
    xi2 = inputs_image2
    xp = inputs_value

    if not no_norm_layer:
        xi = generate_normalization_layers(xi, single_step)
        xi2 = generate_normalization_layers(xi2, single_step)

    if augmentation_params is not None:
        xi = generate_augmentation_layers(xi, augmentation_params, single_step)
        xi2 = generate_augmentation_layers(xi2, augmentation_params, single_step)

    # keras sequential to wrap conv layers
    seq = keras.Sequential()

    # Conv Layers
    seq.add(wrap_time(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step))
    seq.add(wrap_time(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step))
    seq.add(wrap_time(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'), single_step))
    seq.add(wrap_time(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'), single_step))
    seq.add(wrap_time(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'), single_step))

    seq.add(wrap_time(keras.layers.Flatten(), single_step))
    seq.add(wrap_time(keras.layers.Dense(units=128, activation='linear'), single_step))
    seq.add(wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step))

    xi = seq(xi)
    xi2 = seq(xi2)

    xp = wrap_time(keras.layers.Dense(units=128, activation='relu'), single_step)(xp)
    xp = wrap_time(keras.layers.Dropout(rate=DROPOUT), single_step)(xp)

    x = wrap_time(keras.layers.Concatenate(axis=-1), single_step)([xi, xp])
    x = wrap_time(keras.layers.Dense(units=128, activation='linear'), single_step)(x)
    # concatenate xi and xp using tf.concat along the last axis
    # x = wrap_time(keras.layers.Lambda(lambda y: tf.concat(y, axis=-1)), single_step)([xi, xp])
    # x = tf.concat([xi, xp], axis=-1)
    
    # x = xi
    x2 = xi2

    return inputs_image, inputs_image2, inputs_value, x, x2
