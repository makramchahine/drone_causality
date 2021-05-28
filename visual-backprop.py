import tensorflow as tf
from tensorflow import keras
import kerasncp as kncp
from kerasncp.tf import LTCCell
from node_cell import *

import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import os
import datetime

# from mxnet.gluon import nn
# from mxnet import np, npx, init

TRAIN_LSTM = False
MODEL_RECORDING_DIRECTORY = "C:\\Users\\MIT Driverless\\Documents\\deepdrone\\data\\2021-05-20-20-19-28\\images"
IMAGE_OUTPUT_DIRECTORY    = "C:\\Users\\MIT Driverless\\Documents\\deepdrone\\visualbackprop"
WEIGHTS_PATH              = 'C:\\Users\\MIT Driverless\\Documents\\deepdrone\\logs\\target-redwood\\ncp-2021-03-21-13-22-05-rev=13.0-weights.028--0.8371.hdf5'

# Setup the network
SEQUENCE_LENGTH = 32
IMAGE_SHAPE     = (256,256,3)

# wiring = kncp.wirings.NCP(
#     inter_neurons=12,   # Number of inter neurons
#     command_neurons=8,  # Number of command neurons
#     motor_neurons=3,    # Number of motor neurons
#     sensory_fanout=4,   # How many outgoing synapses has each sensory neuron
#     inter_fanout=4,     # How many outgoing synapses has each inter neuron
#     recurrent_command_synapses=4,   # Now many recurrent synapses are in the
#                                     # command neuron layer
#     motor_fanin=6,      # How many incomming syanpses has each motor neuron
# )

# rnnCell = kncp.LTCCell(wiring)

# kernels = [
#     (5,5),
#     (5,5),
#     (3,3),
#     (3,3),
#     (3,3),
# ]

# strides = [
#     (2,2),
#     (2,2),
#     (2,2),
#     (1,1),
#     (1,1),
# ]

# fullModel = keras.models.Sequential()
# fullModel.add(keras.Input(shape=(None, *IMAGE_SHAPE)))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=kernels[0], strides=strides[0], activation='relu')))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=kernels[1], strides=strides[1], activation='relu')))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=kernels[2], strides=strides[2], activation='relu')))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=kernels[3], strides=strides[3], activation='relu')))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=kernels[4], strides=strides[4], activation='relu')))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1000, activation='relu')))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=100,  activation='relu')))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.3)))
# fullModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=24,   activation='relu')))
# fullModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

# fullModel.compile(
#     optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
# )

# # LSTM network
# penultimateOutput = fullModel.layers[-2].output
# lstmOutput        = keras.layers.SimpleRNN(units=3, return_sequences=True, activation='relu')(penultimateOutput)
# lstmModel = keras.models.Model(fullModel.input, lstmOutput)

# # Configure the model we will train
# if TRAIN_LSTM:
#     visualizeModel = lstmModel
# else:
#     visualizeModel = fullModel

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

print(dir(kncp))

rnnCell = LTCCell(wiring)

kernels = [
    (5,5),
    (3,3),
    (2,2),
    (2,2)
] 

strides = [ 
    (3,3),
    (2,2),
    (2,2),
    (2,2),
]


ncpModel = keras.models.Sequential()
ncpModel.add(keras.Input(shape=(SEQUENCE_LENGTH, *IMAGE_SHAPE)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8, kernel_size= (2,2), strides=(2,2), activation='relu')))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64,   activation='linear')))
ncpModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

ncpModel.compile(
    optimizer=keras.optimizers.Adam(0.0005), loss="cosine_similarity",
)

# Load weights
visualizationModel = ncpModel


# Load weights
visualizationModel.load_weights(WEIGHTS_PATH)
visualizationModel.summary(line_length=80)

convolutionalLayers = visualizationModel.layers[:4]

visualizationModel.summary()
# print([l.output_shape for l in convolutionalLayers])

# sys.exit()


# Separate the covolutional and dense outputs for individual inspection
# convModel = keras.models.Model(visualizationModel.input, outputs=[visualizationModel.layers[4].output])
# denseModel = keras.models.Model(visualizationModel.input, outputs=[visualizationModel.layers[11].output])
# rnnModel  = keras.models.Model(fullModel.input, outputs=[keras.])

# Visual saliancy

# convolutionalOutput = model.layers[4].output
# saliencyDetector = ConvolutionHead(filters=convolutionalOutput.shape[-1], features_per_filter=(convolutionalOutput.shape[-3]*convolutionalOutput.shape[-2]))

activationModel = keras.models.Model(inputs=visualizationModel.input, outputs=[layer.output for layer in convolutionalLayers])


# Show LTC Model
# plt.figure()
# legend_handles = rnnCell.draw_graph(draw_labels=True)
# plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
# # sns.despine(left=True, bottom=True)
# plt.tight_layout()
# plt.show()

FEATURES_PER_ROW = 12

layer_names = []
for layer in visualizationModel.layers:
    layer_names.append(layer.name)

images = np.zeros((1, SEQUENCE_LENGTH, *IMAGE_SHAPE))
for i, imageFile in enumerate(os.listdir(MODEL_RECORDING_DIRECTORY)):
    print("Creating Image: ", i)
    try:
        image = np.array(PIL.Image.open(MODEL_RECORDING_DIRECTORY + '\\' + imageFile).convert('RGB'), dtype=np.float32) / 255
    except PIL.UnidentifiedImageError:
        print("Image: ", imageFile, " is corrupt.")

    # Correctly add image to sliding window
    if i < SEQUENCE_LENGTH:
        images[0, i] = image
    else:
        images[0] = np.roll(images[0], -1, axis=0)
        images[0, -1] = image

    # Visualization
    #print("Predicting")
    activations = activationModel.predict(images)
    #print("Done Predicting")
    average_layer_maps = []
    for layer_activation in activations: # Only the convolutional layers 
        feature_maps = layer_activation[0, min(i, SEQUENCE_LENGTH-1)]
        n_features   = feature_maps.shape[-1]
        average_feature_map = np.sum(feature_maps, axis=-1) / n_features
        average_layer_maps.append(average_feature_map)
        #print(average_feature_map.shape)

    # print(average_layer_maps)
    # print(average_layer_maps[0].shape)
    average_layer_maps = [fm[np.newaxis, :, :, np.newaxis] for fm in average_layer_maps]
    saliency_mask = tf.constant(average_layer_maps[-1])
    for l in reversed(range(0, len(average_layer_maps))):
        kernel = np.ones((*kernels[l], 1,1))

        if l > 0:
            output_shape = average_layer_maps[l-1].shape
        else:
            output_shape = (1, *(IMAGE_SHAPE[:2]), 1)

        #print('---')
        #print(saliency_mask.shape)
        #print(average_layer_maps[l].shape)
        #print(average_layer_maps[l-1].shape)
        #print(kernel.shape)
        #print(output_shape)

        saliency_mask = tf.nn.conv2d_transpose(saliency_mask, kernel, output_shape, strides[l], padding='VALID')
        if l > 0:
            saliency_mask = saliency_mask * average_layer_maps[l-1]

        saliency_mask      /= np.max(saliency_mask)

        #print('Saliency: ', saliency.shape)
    saliency_mask = saliency_mask[0]

    plt.imshow(saliency_mask)
    plt.savefig(IMAGE_OUTPUT_DIRECTORY + f'//saliency_mask_{i:05}.png')
    

        # tconv = nn.Conv2DTranspose(1, kernel_size=kernel.shape, strides=strides[l])
        # tconv.initialize(init.Constant(kernel))
        # saliency = tconv(saliency)
        # saliency = saliency * average_layer_maps[i-1]
        # print("Done")
        # tf.nn.conv2d_transpose(input=tf.convert_to_tensor(saliency), filters=np.ones((*kernels[l], 1, 1)), output_shape=average_layer_maps[l-1], strides=strides[l], padding='SAME', name=None)

    

    # TODO (cvorbach) the scale to image

    # RNN input
    # denseOutput = denseModel.predict(images)
    # print(denseOutput.shape)

    # sensoryNeuronActivation = denseOutput[0, min(i, SEQUENCE_LENGTH-1)]





    # # The first Convolutional Feature
    # convolutionalOutput = convModel.predict(images)
    # print(convolutionalOutput.shape)
    # plt.imshow(convolutionalOutput[0, i, :, :, 0])
    # plt.show()

    # # direction = fullModel.predict(images)[0][0]

    # activations = activation_model.predict(images)
    # first_layer_activation = activations[0]


    # # plt.imshow(image)

    # for layer_name, layer_activation in zip(layer_names, activations):
    #     n_features   = layer_activation.shape[-1]
    #     size         = layer_activation.shape[2]  # feature map is (size, size, n_features)
    #     n_cols       = n_features // FEATURES_PER_ROW
    #     display_grid = np.zeros((size * n_cols, FEATURES_PER_ROW * size))
    #     for col in range(n_cols):
    #         for row in range(FEATURES_PER_ROW):
    #             print(layer_activation.shape)
    #             channel_image = layer_activation[0, 0, :, :, col * FEATURES_PER_ROW + row] # TODO(cvorbach) check this index

    #             # Normalize and convert image
    #             channel_image -= channel_image.mean()
    #             channel_image /= channel_image.std()
    #             channel_image *= 64
    #             channel_image += 128
    #             channel_image = np.clip(channel_image, 0, 255).astype('uint8')

    #             print(channel_image.shape)

    #             display_grid[col * size : (col + 1) * size, 
    #                          row * size : (row + 1) * size] = channel_image

    #     scale = 1. / size
    #     plt.figure(figsize = (scale  * display_grid.shape[1], scale * display_grid.shape[0]))
    #     plt.title(layer_name)
    #     plt.grid(False)
    #     plt.imshow(display_grid, aspect='auto', cmap='viridis')
    #     plt.show()

    # plt.matshow(first_layer_activation[0, 0, :, :, 4], cmap='viridis')
    # plt.show()
