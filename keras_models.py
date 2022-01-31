import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Tuple, Iterable, Optional, Dict, List

import kerasncp as kncp
from kerasncp.tf import LTCCell
from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, Dense

from node_cell import *
from tf_cfc import CfcCell, MixedCfcCell
from tf_cfc import LTCCell as CFCLTCCell

IMAGE_SHAPE = (144, 256, 3)


# helper classes that contain all the parameters in the generate_*_model functions
@dataclass
class ModelParams:
    # dataclasses can't have non-default follow default
    seq_len: int = field(default=False, init=True)
    image_shape: Tuple[int, int, int] = IMAGE_SHAPE
    do_normalization: bool = False
    do_augmentation: bool = False
    data: Optional[Iterable] = None
    augmentation_params: Dict = None
    rnn_stateful: bool = False
    batch_size: Optional[int] = None
    single_step: bool = False
    no_norm_layer: bool = False


@dataclass
class NCPParams(ModelParams):
    seed: int = 22222


@dataclass
class LSTMParams(ModelParams):
    rnn_sizes: List[int] = field(default=False, init=True)
    dropout: float = 0.1
    recurrent_dropout: float = 0.1


@dataclass
class CTRNNParams(ModelParams):
    rnn_sizes: List[int] = field(default=False, init=True)
    ct_network_type: str = 'ctrnn'
    config: Dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_CFC_CONFIG))


@dataclass
class TCNParams(ModelParams):
    nb_filters: int = field(default=False, init=True)
    kernel_size: int = field(default=False, init=True)
    dilations: Iterable[int] = field(default=False, init=True)
    dropout: float = 0.1


def get_readable_name(params: ModelParams):
    """
    Extracts the model name from the class of params
    """
    class_name = str(params.__class__.__name__)
    return class_name.replace("Params", "").lower()


DROPOUT = 0.1

# TODO: are these params relevant?
# DEFAULT_CONFIG = {
#     "clipnorm": 1,
#     "size": 128,
#     "backbone_activation": "tanh",
#     "backbone_dr": 0.1,
#     "forget_bias": 1.6,
#     "backbone_units": 256,
#     "backbone_layers": 2,
#     "weight_decay": 1e-06,
#     "use_mixed": False,
# }
DEFAULT_CFC_CONFIG = {
    "clipnorm": 1,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-06
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Shapes for generate_*_model:
# if single_step, input is tuple of image input (batch [usually 1], h, w, c), and hiddens (batch, hidden_dim)
# if not single step, is just sequence of images with shape (batch, seq_len, h, w, c) otherwise
# output is control output for not single step, for single step it is list of tensors where first element is control
# output and other outputs are any hidden states required
# if single_step, control output is (batch, 4), otherwise (batch, seq_len, 4)
# if single_step, hidden outputs typically have shape (batch, hidden_dimension)

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
                        batch_size=None,
                        single_step: bool = False,
                        no_norm_layer: bool = False,
                        ):
    inputs_image, x = generate_network_trunk(seq_len,
                                             image_shape,
                                             do_normalization,
                                             do_augmentation,
                                             data,
                                             augmentation_params,
                                             rnn_stateful=rnn_stateful,
                                             batch_size=batch_size,
                                             single_step=single_step,
                                             no_norm_layer=no_norm_layer,
                                             )

    # print(lstm_model.layers[-1].output_shape)
    # print(batch_size, seq_len, lstm_model.layers[-1].output_shape[-1])

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
        # if ix < len(rnn_sizes) - 1:
        #    lstm_model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.1)))

    # lstm_model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.05)))
    x = keras.layers.Dense(units=4, activation='linear')(x)
    if single_step:
        lstm_model = keras.Model([inputs_image, *c_inputs, *h_inputs], [x, *c_outputs, *h_outputs])
    else:
        lstm_model = keras.Model([inputs_image], [x])

    return lstm_model


def generate_ncp_model(seq_len,
                       image_shape,
                       do_normalization,
                       do_augmentation,
                       data,
                       augmentation_params=None,
                       rnn_stateful=False,
                       batch_size=None,
                       seed=22222,
                       single_step: bool = False,
                       no_norm_layer: bool = False,
                       ):
    inputs_image, x = generate_network_trunk(
        seq_len,
        image_shape,
        do_normalization,
        do_augmentation,
        data, augmentation_params,
        rnn_stateful=rnn_stateful,
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
        ncp_model = keras.Model([inputs_image, inputs_state], [motor_out, output_states])
    else:
        x = keras.layers.RNN(rnn_cell,
                             batch_input_shape=(batch_size,
                                                seq_len,
                                                x.shape[-1]),
                             return_sequences=True)(x)

        ncp_model = keras.Model([inputs_image], [x])

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
                         config=DEFAULT_CFC_CONFIG,
                         single_step: bool = False,
                         no_norm_layer: bool = False,
                         ):
    inputs_image, x = generate_network_trunk(
        seq_len, image_shape,
        do_normalization,
        do_augmentation,
        data,
        augmentation_params,
        rnn_stateful=rnn_stateful,
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
            x = keras.layers.RNN(rnn_cell,
                                 batch_input_shape=(batch_size, seq_len,
                                                    x.shape[-1]),
                                 return_sequences=True,
                                 stateful=rnn_stateful,
                                 time_major=False)(x)

    x = keras.layers.Dense(units=4, activation='linear')(x)
    if single_step:
        ctrnn_model = keras.Model([inputs_image, *all_hidden_inputs], [x, *all_hidden_outputs])
    else:
        ctrnn_model = keras.Model([inputs_image], [x])

    return ctrnn_model


def generate_tcn_model(
        nb_filters: int,
        kernel_size: int,
        dilations: Iterable[int],
        seq_len,
        image_shape,
        do_normalization,
        do_augmentation,
        data,
        dropout=0.1,
        augmentation_params=None,
        rnn_stateful=False,
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

    inputs_image, x = generate_network_trunk(
        seq_len,
        image_shape,
        do_normalization,
        do_augmentation,
        data, augmentation_params,
        rnn_stateful=rnn_stateful,
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
        # append current embedding to previous embedding sequence
        inputs_sequence = tf.keras.Input(shape=(None, nb_filters))  # None is seq len, assume batch size 1 prepended
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
        tcn_model = keras.Model([inputs_image, inputs_sequence], [x, combined_sequence])
    else:
        tcn_model = keras.Model([inputs_image], [x])

    return tcn_model


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
                           batch_size=None,
                           single_step: bool = False,
                           no_norm_layer: bool = False, ):
    """
    Generates CNN image processing backbone used in all recurrent models. Uses Keras.Functional API

    returns input to be used in Keras.Model and x, a tensor that represents the output of the network that has shape
    (batch [None], seq_len, num_units) if single step is false and (batch [None], num_units) if single step is true.
    Input has shape (batch, h, w, c) if single step is True and (batch, seq, h, w, c) otherwise

    """

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
    def wrap_time(layer):
        """
        Helper function that wraps layer in a timedistributed or not depending on the arguments of this function
        """
        if not single_step:
            return keras.layers.TimeDistributed(layer)
        else:
            return layer

    model_vgg = keras.applications.VGG16(include_top=False,
                                         weights='imagenet',
                                         input_shape=image_shape)

    layers = [l for l in model_vgg.layers]

    if single_step:
        inputs = keras.Input(shape=image_shape)
    else:
        inputs = keras.Input(batch_input_shape=(batch_size, seq_len, *image_shape))

    layers[0] = inputs

    if not no_norm_layer:
        rescaling_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

        # normalization_layer = keras.layers.experimental.preprocessing.Normalization(
        #     mean=[0.41718618, 0.48529191, 0.38133072],
        #     variance=[0.19504249, 0.18745404, 0.20891384])

        normalization_layer = keras.layers.experimental.preprocessing.Normalization(
            mean=[0.41718618, 0.48529191, 0.38133072],
            variance=[.057, .05, .061])

        # x = layers[0] # test network without normalization

        x = rescaling_layer(layers[0])
        x = wrap_time(normalization_layer)(x)
    else:
        x = layers[0]

    # for i in range(len(layers)):
    #     if i == 0:
    #         continue
    #     else:
    #         layers[i].trainable = False
    #         layers[i] = keras.layers.TimeDistributed(layers[i])
    #         x = layers[i](x)

    # my_time_model = keras.Model(inputs=inputs, outputs=x)

    # Conv Layers
    x = wrap_time(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))(x)

    x = wrap_time(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))(x)

    x = wrap_time(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))(x)

    x = wrap_time(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))(x)

    x = wrap_time(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'))(x)

    # fully connected layers

    x = wrap_time(keras.layers.Flatten())(x)
    x = wrap_time(keras.layers.Dense(units=128, activation='linear'))(x)
    x = wrap_time(keras.layers.Dropout(rate=DROPOUT))(x)

    # print(model.summary())

    return inputs, x


def get_skeleton(params: ModelParams):
    """
    Returns a new model with randomized weights according to the parameters in params
    """
    if isinstance(params, NCPParams):
        model_skeleton = generate_ncp_model(**asdict(params))
    elif isinstance(params, CTRNNParams):
        model_skeleton = generate_ctrnn_model(**asdict(params))
    elif isinstance(params, LSTMParams):
        model_skeleton = generate_lstm_model(**asdict(params))
    elif isinstance(params, TCNParams):
        model_skeleton = generate_tcn_model(**asdict(params))
    else:
        raise ValueError(f"Could not parse param type {params.__class__}")
    return model_skeleton


def load_model_from_weights(params: ModelParams, checkpoint_path: str, load_name_ok: bool = False):
    """
    Convenience function that loads weights from checkpoint_path into model_skeleton
    """
    model_skeleton = get_skeleton(params)
    if load_name_ok:
        try:
            model_skeleton.load_weights(checkpoint_path)
        except TypeError:
            # different number of weights from file and model. Assume normalization layer in model but not file
            # rename conv layers starting at 5
            print("Model had incorrect number of layers. Attempting to load from layer names")
            conv_index = 5
            dense_index = 1
            for layer in model_skeleton.layers:
                if isinstance(layer, Conv2D):
                    layer._name = f"conv2d_{conv_index}"
                    conv_index += 1
                elif isinstance(layer, Dense):
                    layer._name = f"dense_{dense_index}"
                    dense_index += 1
            model_skeleton.load_weights(checkpoint_path, by_name=True)
    else:
        model_skeleton.load_weights(checkpoint_path)

    return model_skeleton


def load_model_no_params(checkpoint_path: str, single_step: bool):
    """
    Convenience function that calls load_model_from weights as above but tries to infer reasonable default params if not
    known
    """
    if 'ncp' in checkpoint_path:
        params = NCPParams(seq_len=64, single_step=single_step)
    elif 'mixedcfc' in checkpoint_path:
        params = CTRNNParams(seq_len=64, rnn_sizes=[128], ct_network_type="mixedcfc", single_step=single_step)
    elif 'lstm' in checkpoint_path:
        params = LSTMParams(seq_len=64, rnn_sizes=[128], single_step=single_step)
    elif "tcn" in checkpoint_path:
        params = TCNParams(seq_len=64, nb_filters=128, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32])
    else:
        raise ValueError(f"Unable to infer model name from path {checkpoint_path}")

    return load_model_from_weights(params, checkpoint_path)
