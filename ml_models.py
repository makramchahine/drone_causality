# ml-models  Copyright (C) 2021  Charles Vorbach, Ramin Hasani, Alexander Amini
from tensorflow import keras
import kerasncp as kncp
from node_cell import *

# Setup the ML Networks
def initializeMLNetwork(config):

    # NCP Model
    wiring = kncp.wirings.NCP(
        inter_neurons=12,   # Number of inter neurons
        command_neurons=32, # Number of command neurons
        motor_neurons=3,    # Number of motor neurons
        sensory_fanout=4,   # How many outgoing synapses has each sensory neuron
        inter_fanout=4,     # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,   # Now many recurrent synapses are in the
                                        # command neuron layer
        motor_fanin=6,      # How many incoming syanpses has each motor neuron
    )

    rnnCell = kncp.LTCCell(wiring)

    ncpModel = keras.models.Sequential()
    ncpModel.add(keras.Input(shape=(config['sequenceLength'], *config['imageShape'])))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8, kernel_size=(2,2), strides=(2,2), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64,   activation='linear')))
    ncpModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

    # LSTM network
    penultimateOutput = ncpModel.layers[-2].output
    lstmOutput        = keras.layers.LSTM(units=config['rnnSize'], return_sequences=True)(penultimateOutput)
    lstmOutput        = keras.layers.Dense(units=3, activation='linear')(lstmOutput)
    lstmModel = keras.models.Model(ncpModel.input, lstmOutput)

    # Vanilla RNN network
    penultimateOutput = ncpModel.layers[-2].output
    rnnOutput         = keras.layers.SimpleRNN(units=config['rnnSize'], return_sequences=True)(penultimateOutput)
    rnnOutput         = keras.layers.Dense(units=3, activation='linear')(rnnOutput)
    rnnModel          = keras.models.Model(ncpModel.input, rnnOutput)

    # GRU network
    penultimateOutput = ncpModel.layers[-2].output
    gruOutput         = keras.layers.GRU(units=config['rnnSize'], return_sequences=True)(penultimateOutput)
    gruOutput         = keras.layers.Dense(units=3, activation='linear')(gruOutput)
    gruModel          = keras.models.Model(ncpModel.input, gruOutput)

    # CT-GRU network
    penultimateOutput  = ncpModel.layers[-2].output
    ctgruCell          = CTGRU(units=config['rnnSize'])
    ctgruOutput        = keras.layers.RNN(ctgruCell, return_sequences=True)(penultimateOutput)
    ctgruOutput        = keras.layers.Dense(units=3, activation='linear')(ctgruOutput)
    ctgruModel         = keras.models.Model(ncpModel.input, ctgruOutput)

    # ODE-RNN network
    penultimateOutput = ncpModel.layers[-2].output
    odernnCell        = CTRNNCell(units=config['rnnSize'], method='dopri5')
    odernnOutput      = keras.layers.RNN(odernnCell, return_sequences=True)(penultimateOutput)
    odernnOutput      = keras.layers.Dense(units=3, activation='linear')(odernnOutput)
    odernnModel       = keras.models.Model(ncpModel.input, odernnOutput)

    # CNN network
    remove_ncp_layer = ncpModel.layers[-3].output
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dense(units=250, activation='relu'))(remove_ncp_layer)
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5))(cnnOutput)
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dense(units=25, activation='relu'))(cnnOutput)
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.3))(cnnOutput)
    cnnOutput = keras.layers.Dense(units=3, activation='linear')(cnnOutput)
    cnnModel  = keras.models.Model(ncpModel.input, cnnOutput)

    # CNN multiple input network
    # TODO(cvorbach) Not sure if this makes sense for a cnn?

    # Select the model we will train
    modelName = config['modelName'] 
    print(f'Loading ML Model [{modelName}]')

    if modelName == "lstm":
        flightModel = lstmModel
    elif modelName == "ncp":
        flightModel = ncpModel
    elif modelName == "cnn":
        flightModel = cnnModel
    elif modelName == "odernn":
        flightModel = odernnModel
    elif modelName == "gru":
        flightModel = gruModel
    elif modelName == "rnn":
        flightModel = rnnModel
    elif modelName == "ctgru":
        flightModel = ctgruModel
    else:
        raise ValueError(f"Unsupported model type: {modelName}")

    flightModel.compile(
        optimizer=keras.optimizers.Adam(0.0005), loss="cosine_similarity",
    )

    # Load weights
    flightModel.load_weights(config['modelWeights'])
    flightModel.summary(line_length=80)

    return flightModel
