import numpy as np
import os
# from tensorflow import keras
# import kerasncp as kncp
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import PIL.Image
from tensorflow import keras
import kerasncp as kncp
from datetime import datetime

# Load the data
dataRuns = []
dataDirectory = os.getcwd() + '\\training-data'
imageOdometryType = [('timestamp', np.uint64), ('x', np.float32), ('y', np.float32), ('z', np.float32), ('qw', np.float32), ('qx', np.float32), ('qy', np.float32), ('qz', np.float32), ('imagefile', 'U32')]

# for runDirectory in os.listdir(dataDirectory):
#     imageDirectory = dataDirectory + '\\' + runDirectory + '\\images'
# 
#     if len(os.listdir(imageDirectory)) <= 1:
#         print(imageDirectory)


for n, runDirectory in enumerate(os.listdir(dataDirectory)):
    if n > 10:
        break
    imageDirectory = dataDirectory + '\\' + runDirectory + '\\images'
    odometryFile   = dataDirectory + '\\' + runDirectory + '\\airsim_rec.txt'

    odometry = np.array(np.genfromtxt(fname=odometryFile, dtype=imageOdometryType, skip_header=1))
    validImages = np.full((len(odometry),), fill_value=False)

    imageMap = dict()
    for i, record in enumerate(odometry):
        imageFile = str(record["imagefile"])
        try:
            imageMap[imageFile] = np.array(PIL.Image.open(imageDirectory + '\\' + imageFile).convert('RGB')) 
            validImages[i]     = True
        except PIL.UnidentifiedImageError:
            pass

    odometry = odometry[validImages]

    # print(imageMap[imageFile].shape)

    images = np.zeros((len(odometry), *(imageMap[imageFile].shape)))

    for i, record in enumerate(odometry):
        images[i] = imageMap[record['imagefile']]

    dataRuns.append((odometry, images))

# Setup the network
wiring = kncp.wirings.NCP(
    inter_neurons=12,   # Number of inter neurons
    command_neurons=8,  # Number of command neurons
    motor_neurons=3,    # Number of motor neurons
    sensory_fanout=4,   # How many outgoing synapses has each sensory neuron
    inter_fanout=4,     # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=4,   # Now many recurrent synapses are in the
                                    # command neuron layer
    motor_fanin=6,      # How many incomming syanpses has each motor neuron
)
rnnCell = kncp.LTCCell(wiring)

imageShape = dataRuns[0][1].shape # (Sample Count, 144, 256, 3)

model = keras.models.Sequential()
model.add(keras.Input(shape=(None, *imageShape[1:])))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2), strides=2)))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.MaxPool2D(pool_size=(2,2))))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=24, activation='relu')))
model.add(keras.layers.RNN(rnnCell, return_sequences=True))

model.compile(
    optimizer=keras.optimizers.Adam(0.01), loss="mean_squared_error",
)

# print("Output: ", model.output)
# print("Input: ", model.input)

# model.summary(line_length=100)

# Plot NCP wiring
# sns.set_style("white")
# plt.figure(figsize=(12, 12))
# legend_handles = rnnCell.draw_graph(layout='spiral',neuron_colors={"command": "tab:cyan"})
# plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
# sns.despine(left=True, bottom=True)
# plt.tight_layout()
# plt.show()

# Train

for i in range(10):
    for run in dataRuns:
        xTrain = np.array([run[1][:-1]])
        yTrain = np.array([[[odometry['x'], odometry['y'], odometry['z']] for odometry in run[0][1:]]])

        # print("x: ", xTrain.shape)
        # print("y: ", yTrain.shape)

        model.fit(
            x=xTrain, y=yTrain, epochs=5
        )

        model.save('model-checkpoints/' + datetime.now() + '-model.p')

model.evaluate(xTrain, yTrain)

# Offload 

# Visualize?
