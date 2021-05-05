# drone-flight  Copyright (C) 2020  Charles Vorbach
import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr, YawMode

import sys 
import time 
import threading
import random 
import numpy as np
import pprint
import heapq
import pickle
import matplotlib.pyplot as plt
import cv2
import os
import csv
import re
import argparse
from collections import OrderedDict
from enum import Enum

from scipy.spatial.transform import Rotation as R

# Start up
client = airsim.MultirotorClient() 
client.confirmConnection() 
client.enableApiControl(True) 

# Weather
client.simEnableWeather(True)
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.0)
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0)

# Operating Modes
class Task: 
    TARGET = 'target'
    FOLLOWING = 'following'
    MAZE = 'maze'
    HIKING = 'hiking'

# Parameters
parser = argparse.ArgumentParser(description='Fly the deepdrone agent in the Airsim simulator')
parser.add_argument('--task',               type=str,   default='target', help='Task to attempt')
parser.add_argument('--endpoint_tolerance', type=float, default=5.0,      help='The distance tolerance on reaching the endpoint marker')
parser.add_argument('--near_task_radius',   type=float, default=15.0,     help='The max distance of endpoints in the near planning task')
parser.add_argument('--far_task_radius',    type=float, default=50.0,     help='The max distance of endpoints in the far planning task')
parser.add_argument('--min_blaze_gap',      type=float, default=10.0,     help='The minimum distance between hiking task blazes')
parser.add_argument('--plot_period',        type=float, default=0.5,      help='The time between updates of debug plotting information')
parser.add_argument('--control_period',     type=float, default=0.7,      help='Update frequency of the pure pursuit controller')
parser.add_argument('--speed',              type=float, default=0.5,      help='Drone flying speed')
parser.add_argument('--voxel_size',         type=float, default=1.0,      help='The size of voxels in the occupancy map cache')
parser.add_argument('--cache_size',         type=float, default=100000,   help='The number of entries in the local occupancy cache')
parser.add_argument('--lookahead_distance', type=float, default=0.75,      help='Pure pursuit lookahead distance')
parser.add_argument('--bogo_attempts',      type=int,   default=5000,     help='Number of attempts to make in generate and test algorithms')
parser.add_argument('--n_runs',             type=int,   default=30,       help='Number of repetitions of the task to attempt')
parser.add_argument("--plot_debug", dest="plot_debug", action="store_true")
parser.set_defaults(gps_signal=False)
parser.add_argument('--record', dest='record', action='store_true')
parser.set_defaults(record=False)
parser.add_argument('--model_weights', type=str, default=None, help='Model weights to load and fly with')
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--rnn_size', type=int, default=32, help='Select the size of RNN network you would like to train')
parser.add_argument('--timeout', type=int, default=30)
args = parser.parse_args()

RECORDING_DIRECTORY    = 'C:/Users/MIT Driverless/Documents/AirSim'
RECORDING_NAME_REGEX   = re.compile(r'^[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+$')

CAMERA_FOV           = np.pi / 8
RADIANS_2_DEGREES    = 180 / np.pi
CAMERA_OFFSET        = np.array([0.5, 0, -0.5])
ENDPOINT_OFFSET      = np.array([0, -0.03, 0.025])
MAX_INCLINATION      = 0.3
DRONE_START          = np.array([-32295.757812, 2246.772705, 1894.547119])
WORLD_2_UNREAL_SCALE = 100

#TODO(cvorbach) CAMERA_HORIZONTAL_OFFSET

# Setup the network
IMAGE_SHAPE     = (256,256,3)

flightModel = None
if args.model_weights is not None:
    from tensorflow import keras
    import kerasncp as kncp
    from node_cell import *

    # Parse out the model info from file path
    weightsFile = args.model_weights.split('/')[-1]
    modelName   = weightsFile[:weightsFile.index('-')]
    # modelName   = 'ncp'

    # Setup the network
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
    ncpModel.add(keras.Input(shape=(args.seq_len, *IMAGE_SHAPE)))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(2,2), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=8, kernel_size=(2,2), strides=(2,2), activation='relu')))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
    ncpModel.add(keras.layers.TimeDistributed(keras.layers.Dense(units=64,   activation='linear')))
    ncpModel.add(keras.layers.RNN(rnnCell, return_sequences=True))

    # NCP network with multiple input (Requires the Functional API)
    imageInput        = ncpModel.layers[0].input
    penultimateOutput = ncpModel.layers[-2].output
    imageFeatures     = keras.layers.Dense(units=48, activation="linear")(penultimateOutput)

    gpsInput    = keras.Input(shape = (args.seq_len, 3))
    gpsFeatures = keras.layers.Dense(units=16, activation='linear')(gpsInput)

    multiFeatures = keras.layers.concatenate([imageFeatures, gpsFeatures])

    rnn, state = keras.layers.RNN(rnnCell, return_state=True)(multiFeatures)
    npcMultiModel = keras.models.Model(inputs=[imageInput, gpsInput], outputs = [rnn])

    # LSTM network
    penultimateOutput = ncpModel.layers[-2].output
    lstmOutput        = keras.layers.LSTM(units=args.rnn_size, return_sequences=True)(penultimateOutput)
    lstmOutput        = keras.layers.Dense(units=3, activation='linear')(lstmOutput)
    lstmModel = keras.models.Model(ncpModel.input, lstmOutput)

    # LSTM multiple input network
    lstmMultiOutput        = keras.layers.LSTM(units=args.rnn_size, return_sequences=True)(multiFeatures)
    lstmMultiOutput        = keras.layers.Dense(units=3, activation='linear')(lstmMultiOutput)
    lstmMultiModel = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[lstmMultiOutput])

    # for x in trainData:
    #     (x1, x2), y = x
    #     print(x1.shape)
    #     print(x2.shape)
    #     print(y.shape)
    #     print(lstmMultiOutput.shape)
    #     sys.exit()

    # Vanilla RNN network
    penultimateOutput = ncpModel.layers[-2].output
    rnnOutput         = keras.layers.SimpleRNN(units=args.rnn_size, return_sequences=True)(penultimateOutput)
    rnnOutput         = keras.layers.Dense(units=3, activation='linear')(rnnOutput)
    rnnModel          = keras.models.Model(ncpModel.input, rnnOutput)

    # Vanilla RNN multiple input network
    rnnMultiOutput = keras.layers.SimpleRNN(units=args.rnn_size, return_sequences=True)(multiFeatures)
    rnnMultiOutput = keras.layers.Dense(units=3, activation='linear')(rnnMultiOutput)
    rnnMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[rnnMultiOutput])

    # GRU network
    penultimateOutput = ncpModel.layers[-2].output
    gruOutput         = keras.layers.GRU(units=args.rnn_size, return_sequences=True)(penultimateOutput)
    gruOutput         = keras.layers.Dense(units=3, activation='linear')(gruOutput)
    gruModel          = keras.models.Model(ncpModel.input, gruOutput)

    # GRU multiple input network
    gruMultiOutput = keras.layers.GRU(units=args.rnn_size, return_sequences=True)(multiFeatures)
    gruMultiOutput = keras.layers.Dense(units=3, activation='linear')(gruMultiOutput)
    gruMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[gruMultiOutput])

    # CT-GRU network
    penultimateOutput  = ncpModel.layers[-2].output
    ctgruCell          = CTGRU(units=args.rnn_size)
    ctgruOutput        = keras.layers.RNN(ctgruCell, return_sequences=True)(penultimateOutput)
    ctgruOutput        = keras.layers.Dense(units=3, activation='linear')(ctgruOutput)
    ctgruModel         = keras.models.Model(ncpModel.input, ctgruOutput)

    # CT-GRU multiple input network
    ctgruMultiCell   = CTGRU(units=args.rnn_size)
    ctgruMultiOutput = keras.layers.RNN(ctgruMultiCell, return_sequences=True)(multiFeatures)
    ctgruMultiOutput = keras.layers.Dense(units=3, activation="linear")(ctgruMultiOutput)
    ctgruMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[ctgruMultiOutput])

    # ODE-RNN network
    penultimateOutput = ncpModel.layers[-2].output
    odernnCell        = CTRNNCell(units=args.rnn_size, method='dopri5')
    odernnOutput      = keras.layers.RNN(odernnCell, return_sequences=True)(penultimateOutput)
    odernnOutput      = keras.layers.Dense(units=3, activation='linear')(odernnOutput)
    odernnModel       = keras.models.Model(ncpModel.input, odernnOutput)

    # ODE-RNN multiple input network
    odernnMultiCell   = CTRNNCell(units=args.rnn_size, method='dopri5')
    odernnMultiOutput = keras.layers.RNN(odernnMultiCell, return_sequences=True)(multiFeatures)
    odernnMultiOutput = keras.layers.Dense(units=3, activation='linear')(odernnMultiOutput)
    odernnMultiModel  = keras.models.Model(inputs=[imageInput, gpsInput], outputs=[odernnMultiOutput])

    # CNN network
    # Revision 2: 1000 and 100 units to 500 and 50 units
    remove_ncp_layer = ncpModel.layers[-3].output
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dense(units=250, activation='relu'))(remove_ncp_layer)
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5))(cnnOutput)
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dense(units=25, activation='relu'))(cnnOutput)
    cnnOutput = keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.3))(cnnOutput)
    cnnOutput = keras.layers.Dense(units=3, activation='linear')(cnnOutput)
    cnnModel  = keras.models.Model(ncpModel.input, cnnOutput)

    # CNN multiple input network
    # TODO(cvorbach) Not sure if this makes sense for a cnn?

    # Configure the model we will train
    print(modelName)
    if not args.task == Task.MAZE:
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
    else:
        if modelName == "lstm":
            flightModel = lstmMultiModel
        elif modelName == "ncp":
            flightModel = npcMultiModel
        elif modelName == "cnn":
            raise ValueError(f"Unsupported model type: {modelName}")
        elif modelName == "odernn":
            flightModel = odernnMultiModel
        elif modelName == "gru":
            flightModel = gruMultiModel
        elif modelName == "rnn":
            flightModel = rnnMultiModel
        elif modelName == "ctgru":
            flightModel = ctgruMultiModel
        else:
            raise ValueError(f"Unsupported model type: {modelName}")

    flightModel.compile(
        optimizer=keras.optimizers.Adam(0.0005), loss="cosine_similarity",
    )

    # Load weights
    flightModel.load_weights(args.model_weights)
    flightModel.summary(line_length=80)

# Utilities

def normalize(vector):
    if np.linalg.norm(vector) == 0:
        raise ZeroDivisionError()
    return vector / np.linalg.norm(vector)


def distance(p1, p2):
    return np.linalg.norm(p2 - p1)


class CatmullRomSegment:
    def __init__(self, p, alpha=0.5):
        if len(p) != 4:
            raise ValueError('Catmull Rom Segment requires 4 points')

        self.p = p

        t0 = 0
        t1 = t0 + distance(p[0], p[1])**alpha
        t2 = t1 + distance(p[1], p[2])**alpha
        t3 = t2 + distance(p[2], p[3])**alpha

        self.t = [t0, t1, t2, t3]

        self.coeff = self.getCoeff()
        
        # print('t:', t)
        # print('c:', self.coeff)

    def __call__(self, t):
        if t < 0 or t > 1:
            raise ValueError('Catmull Rom Segment cannot extrapolate t:', t) 

        # print('t', t)

        a, b, c, d = self.coeff
        value = a*t**3 + b*t**2 + c*t + d

        return value

    def ddt(self, t):
        if t < 0 or t > 1:
            raise ValueError('Catmull Rom Segment cannot extrapolate') 

        a, b, c, d = self.coeff
        derivative = 3*a*t**2 + 2*b*t + c
        return derivative

    def getCoeff(self):
        t0, t1, t2, t3 = self.t
        p0, p1, p2, p3 = self.p

        m1 = (t2-t1) * ( (p1-p0)/(t1-t0) - (p2-p0)/(t2-t0) + (p2-p1)/(t2-t1) )
        m2 = (t2-t1) * ( (p2-p1)/(t2-t1) - (p3-p1)/(t3-t1) + (p3-p2)/(t3-t2) )

        print('t', self.t)

        a =  2*self.p[1] - 2*self.p[2] +  m1  + m2
        b = -3*self.p[1] + 3*self.p[2] - 2*m1 - m2
        c = m1
        d = self.p[1]

        return a, b, c, d

class CatmullRomSpline:
    def __init__(self, p):
        if len(p) < 4:
            raise ValueError('Catmull Rom Spline requires at least 4 points')

        self.segments = []

        # Create the initial segment
        self.p = list(p[:4])
        self.extendSegment(p[:4])

        # Create additional segments
        self.extend(p[4:])

    def extend(self, newPoints):
        if len(newPoints) == 0: # If nothing to add
            return 

        # Extend each new segment from the previous last points
        for i, point in enumerate(newPoints):
            self.extendSegment([*self.p[-3:], point])
            self.p.append(point)

    def extendSegment(self, segmentPoints):
        if len(segmentPoints) > 4:
            raise ValueError('Too many points to extend Catmull Rom Spline')

        if len(segmentPoints) < 4:
            raise ValueError('Not enough points to extend Catmull Rom Spline')

        self.segments.append(CatmullRomSegment(segmentPoints))

    def __call__(self, t):
        '''
        Returns the value of the spline at s in [0, len(segments)+1]
        where s is the spline parameterization
        '''
        if t < 0 or t > 1:
            raise Exception(f"Catmull Rom Spline cannot extrapolate")

        s = t*len(self)

        # Index of segment to use
        idx = min(int(s), len(self.segments)-1)
        ds  = s - idx

        value = self.segments[idx](ds)
        return value

    def ddt(self, t):
        '''
        Returns the value of the spline at s in [0, len(segments)+1]
        where s is the spline parameter
        '''
        if t < 0 or t > 1:
            raise Exception(f"Catmull Rom Spline cannot extrapolate")

        s = t*len(self)

        # Index of segment to use
        idx = min(int(s), len(self.segments)-1)
        ds  = s - idx

        derivative = self.segments[idx].ddt(ds)

    def end(self):
        return self(len(segments))

    def pop(self, s):
        self.segments.pop()
        self.p.pop()

    def __len__(self):
      return len(self.segments)


class ExtendablePath:
    def __init__(self, knotPoints):
        if len(knotPoints) < 4:
            raise ValueError('Extendable Path needs at least 4 knots')

        self.knots   = np.array(knotPoints)
        self.xSpline = CatmullRomSpline(list(self.knots[:, 0]))
        self.ySpline = CatmullRomSpline(list(self.knots[:, 1]))
        self.zSpline = CatmullRomSpline(list(self.knots[:, 2]))

    def extend(self, newPoints):
        self.xSpline.extend(list(newPoints[:, 0]))
        self.ySpline.extend(list(newPoints[:, 1]))
        self.zSpline.extend(list(newPoints[:, 2]))

    def __call__(self, t):
        return np.array([
            self.xSpline(t),
            self.ySpline(t),
            self.zSpline(t)
        ])

    def tangent(self, t):
        return np.array([
            self.xSpline.ddt(t),
            self.ySpline.ddt(t),
            self.zSpline.ddt(t)
        ])

    def project(self, point):
        position, _ = getPose()
        tSamples    = np.linspace(0, 1, num=1000)
        nearestT    = tSamples[np.argmin([np.linalg.norm(position - self(t)) for t in tSamples])]

        return nearestT
    
    def end(self):
        return self.knots[-2]

    def __len__(self):
        return len(self.xSpline.segments)


class CubicSpline:
    def __init__(self, x, y, tol=1e-10):
        self.x = x
        self.y = y
        self.coeff = self.fit(x, y, tol)

    def fit(self, x, y, tol=1e-10):
        """
        Interpolate using natural cubic splines.
    
        Generates a strictly diagonal dominant matrix then solves.
    
        Returns coefficients:
        b, coefficient of x of degree 1
        c, coefficient of x of degree 2
        d, coefficient of x of degree 3
        """ 
    
        x = np.array(x)
        y = np.array(y)
    
        # check if sorted
        if np.any(np.diff(x) < 0):
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

        size = len(x)
        delta_x = np.diff(x)
        delta_y = np.diff(y)
    
        # Initialize to solve Ac = k
        A = np.zeros(shape = (size,size))
        k = np.zeros(shape=(size,1))
        A[0,0] = 1
        A[-1,-1] = 1
    
        for i in range(1,size-1):
            A[i, i-1] = delta_x[i-1]
            A[i, i+1] = delta_x[i]
            A[i,i] = 2*(delta_x[i-1]+delta_x[i])

            k[i,0] = 3*(delta_y[i]/delta_x[i] - delta_y[i-1]/delta_x[i-1])
    
        # Solves for c in Ac = k
        c = np.linalg.solve(A, k)
    
        # Solves for d and b
        d = np.zeros(shape = (size-1,1))
        b = np.zeros(shape = (size-1,1))
        for i in range(0,len(d)):
            d[i] = (c[i+1] - c[i]) / (3*delta_x[i])
            b[i] = (delta_y[i]/delta_x[i]) - (delta_x[i]/3)*(2*c[i] + c[i+1])    
    
        return b.squeeze(), c.squeeze(), d.squeeze()

    def __call__(self, t):
        '''
        Returns the value of the spline at t in [x[0], x[-1]]
        '''

        x = self.x
        y = self.y
        b, c, d = self.coeff

        # TODO(cvorbach) allow extrapolation
        if t < x[0] or t > x[-1]:
            raise Exception("Can't extrapolate")

        # Index of segment to use
        idx = np.argmax(x > t) - 1
                
        dx = t - x[idx]
        value = y[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
        return value

    def ddt(self, t):
        '''
        Returns the derivative of the spline at t in [x[0], x[-1]]
        '''

        x = self.x
        y = self.y
        b, c, d = self.coeff

        # TODO(cvorbach) allow extrapolation
        if t < x[0] or t > x[-1]:
            raise Exception("Can't extrapolate")

        # Index of segment to use
        idx = np.argmax(x > t) - 1

        dx         = t - x[idx]
        derivative = b[idx] + 2*c[idx]*dx + 3*d[idx]*dx**2
        return derivative

    def d2dt2(self, t):
        '''
        Returns the second derivative of the spline at t in [x[0], x[-1]]
        '''

        x = self.x
        y = self.y
        b, c, d = self.coeff

        # TODO(cvorbach) allow extrapolation
        if t < x[0] or t > x[-1]:
            raise Exception("Can't extrapolate")

        # Index of segment to use
        idx = np.argmax(x > t) - 1
        secondDerivative = 2*c[idx] + 6*d[idx]*dx
        return secondDerivative

class Path:
    def __init__(self, knotPoints):
        self.knotPoints = knotPoints
        self.fit(knotPoints)

    def fit(self, knotPoints):
        knots = np.array(knotPoints)
        t = np.linspace(0, 1, knots.shape[0])
        self.xSpline = CubicSpline(t, knots[:, 0])
        self.ySpline = CubicSpline(t, knots[:, 1])
        self.zSpline = CubicSpline(t, knots[:, 2])

    def __call__(self, t):
        return np.array([
            self.xSpline(t),
            self.ySpline(t),
            self.zSpline(t)
        ])

    def tangent(self, t):
        return np.array([
            self.xSpline.ddt(t),
            self.ySpline.ddt(t),
            self.zSpline.ddt(t)
        ])

    def normal(self, t):
        tangentDerivative = np.array([
            self.xSpline.d2dt2(t),
            self.ySpline.d2dt2(t),
            self.zSpline.d2dt2(t)
        ])
        return normalize(tangentDerivative)

    def project(self, point):
        position, _ = getPose()
        tSamples = np.linspace(0, 1, num=1000)
        nearstT  = tSamples[np.argmin([np.linalg.norm(position - self(t)) for t in tSamples])]
        return nearstT
    
    def end(self):
        return self.knotPoints[-1]

# claInfinitePath:
#     def __init__(self, start, occupancyMap, momentumWeight=0.9, stepSize=10, inclinationLimit=0.1, zLimit=(-20, -10)):
#         self.occupancyMap     = occupancyMap
#         self.momentumWeight   = momentumWeight
#         self.stepSize         = stepSize
#         self.inclinationLimit = inclinationLimit
# 
#         self.x                = start
#         self.momentum         = normalize(np.array([random.random(), random.random(), 0]))
# 
#         self.generateInitialPath()
# 
#     def getStep(self):
#         perturbance   = normalize(np.random.random_sample(3))
#         stepDirection = self.momentumWeight*self.momentum + (1-self.momentumWeight)*perturbance
#         stepDirection = normalize(stepDirection)
# 
#         self.momentum = stepDirection
#         step = self.stepSize * stepDirection
# 
#         return self.x + step
# 
#     def generateInitialPath(self, start):
#         isCollisionFree = False
#         while not isCollisionFree:
# 
# 
#         initialSteps = [self.getStep()]
# 
#     def ignite(self):
#         p = []
#         for i in range(4):
#             x, 
#             p.append() 
# 
#     # def update(self, nextX, nextMomentum):
#     #     self.x        = nextX
#     #     self.momentum = nextMomentum

def randomWalk(start, momentumWeight=0.5, stepSize=3, gradientLimit=np.pi/12, zLimit=(-20, -10), pathLength=5, occupancyMap=None, retryLimit = 10):
    normalDistribution = np.random.default_rng().normal 
    momentum = normalize(np.array([normalDistribution(), normalDistribution(), normalDistribution()]))
    path = [start]

    for i in range(retryLimit):
        for _ in range(pathLength):

            # Generate an unoccupied next step in the random walk
            isUnoccupiedNextStep = False
            stuckSteps = 0
            while not isUnoccupiedNextStep:
                rotation = R.from_matrix([[momentum[0], 0, 0], [0, momentum[1], 0], [0, 0, momentum[2]]])
                perturbance = rotation.apply(normalize(np.array((0, normalDistribution(), normalDistribution()))))

                # the continue in previous direction with a random perturbance left/right and up/down
                stepDirection = normalize(momentumWeight * momentum + (1 - momentumWeight) * perturbance)

                # apply gradient limit
                heading = rotation.apply([1, 0, 0])
                stepDirection  = np.array([min(max(u - heading[i], -gradientLimit), gradientLimit) for i, u in enumerate(stepDirection)]) 
                stepDirection += heading

                step          = stepSize * stepDirection

                nextStep = path[-1] + step

                # apply altitude limits
                nextStep[2] = min(max(nextStep[2], zLimit[0]), zLimit[1])

                isUnoccupiedNextStep = occupancyMap is None or nextStep not in occupancyMap 

                stuckSteps += int(not isUnoccupiedNextStep)
                if stuckSteps > retryLimit:
                    break 

            path.append(nextStep)

        if stuckSteps < retryLimit:
            break

    if i > retryLimit:
        raise Exception('Couldn\'t find free random walk')

    return path

# Utilities
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __contains__(self, key):
        if key in self.cache:
            self.cache.move_to_end(key) # Move to front of LRU cache
            return True
        return False

    def add(self, key):
        self.cache[key] = None          # Don't care about the dict's value, just its set of keys
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def discard(self, key):
        self.cache.pop(key, None)

    def keys(self):
        return list(self.cache.keys())

class VoxelOccupancyCache:

    def __init__(self, voxelSize: float, capacity: int):
        self.voxelSize  = voxelSize
        self.cache      = LRUCache(capacity)
    
    def addPoint(self, point):
        voxel = self.point2Voxel(point)

        self.cache.add(voxel)
        for v in self.getAdjacentVoxels(voxel):
            self.cache.add(v)

    def __contains__(self, point):
        voxel = self.point2Voxel(point)
        return voxel in self.cache

    def point2Voxel(self, point):
        return tuple(self.voxelSize * int(round(v / self.voxelSize)) for v in point)

    def getAdjacentVoxels(self, voxel):
        adjacentVoxels = [
            (voxel[0] - 1, voxel[1] - 1, voxel[2] - 1),
            (voxel[0],     voxel[1] - 1, voxel[2] - 1),
            (voxel[0] + 1, voxel[1] - 1, voxel[2] - 1),

            (voxel[0] - 1, voxel[1], voxel[2] - 1),
            (voxel[0],     voxel[1], voxel[2] - 1),
            (voxel[0] + 1, voxel[1], voxel[2] - 1),

            (voxel[0] - 1, voxel[1] + 1, voxel[2] - 1),
            (voxel[0],     voxel[1] + 1, voxel[2] - 1),
            (voxel[0] + 1, voxel[1] + 1, voxel[2] - 1),

            (voxel[0] - 1, voxel[1] - 1, voxel[2]),
            (voxel[0],     voxel[1] - 1, voxel[2]),
            (voxel[0] + 1, voxel[1] - 1, voxel[2]),

            (voxel[0] - 1, voxel[1], voxel[2]),
            (voxel[0],     voxel[1], voxel[2]),
            (voxel[0] + 1, voxel[1], voxel[2]),

            (voxel[0] - 1, voxel[1] + 1, voxel[2]),
            (voxel[0],     voxel[1] + 1, voxel[2]),
            (voxel[0] + 1, voxel[1] + 1, voxel[2]),

            (voxel[0] - 1, voxel[1] - 1, voxel[2] + 1),
            (voxel[0],     voxel[1] - 1, voxel[2] + 1),
            (voxel[0] + 1, voxel[1] - 1, voxel[2] + 1),

            (voxel[0] - 1, voxel[1], voxel[2] + 1),
            (voxel[0],     voxel[1], voxel[2] + 1),
            (voxel[0] + 1, voxel[1], voxel[2] + 1),

            (voxel[0] - 1, voxel[1] + 1, voxel[2] + 1),
            (voxel[0],     voxel[1] + 1, voxel[2] + 1),
            (voxel[0] + 1, voxel[1] + 1, voxel[2] + 1),
        ]

        return adjacentVoxels

    def getNextSteps(self, voxel, endpoint):
        neighbors = []
        possibleNeighbors = self.getAdjacentVoxels(voxel)

        for v in possibleNeighbors:
            if v not in self.cache or distance(np.array(v), np.array(endpoint)) < args.endpoint_tolerance:
                neighbors.append(v) 

        return neighbors

    def plotOccupancies(self):
        occupiedPoints = [Vector3r(float(v[0]), float(v[1]), float(v[2])) for v in self.cache.keys()]
        client.simPlotPoints(occupiedPoints, color_rgba = [0.0, 0.0, 1.0, 1.0], duration=args.plot_period/2.0) 


def world2UnrealCoordinates(vector):
    return (vector + DRONE_START) * WORLD_2_UNREAL_SCALE


def unreal2WorldCoordinates(vector):
    return (vector - DRONE_START) / WORLD_2_UNREAL_SCALE


def isVisible(point, position, orientation):
    # Edge case point == position
    if distance(point, position) < 0.05:
        return True

    # Check if endpoint is in frustrum
    xUnit = np.array([1, 0, 0])
    cameraDirection = R.from_quat(orientation).apply(xUnit)

    endpointDirection = normalize(point - position)

    # TODO(cvorbach) check square, not circle
    angle = np.arccos(np.dot(cameraDirection, endpointDirection)) 

    if abs(angle) > CAMERA_FOV:
        return False

    # TODO(cvorbach) Check for occlusions with ray-tracing

    return True


def orientationAt(endpoint, position):
    # Get the drone orientation that faces towards the endpoint at position
    displacement = np.array(endpoint) - np.array(position)
    endpointYaw = np.arctan2(displacement[1], displacement[0])
    orientation = R.from_euler('xyz', [0, 0, endpointYaw]).as_quat()

    return orientation

def euclidean(voxel1, voxel2):
    return distance(np.array(voxel1), np.array(voxel2))

def greedy(voxel1, voxel2):
    return 100*euclidean(voxel1, voxel2)

# A* Path finding 
def findPath(startpoint, endpoint, occupancyMap, h=greedy, d=euclidean):
    start = occupancyMap.point2Voxel(startpoint)
    end   = occupancyMap.point2Voxel(endpoint)

    cameFrom = dict()

    gScore = dict()
    gScore[start] = 0

    fScore = dict()
    fScore[start] = h(start, endpoint)

    openSet = [(fScore[start], start)]

    while openSet:
        current = heapq.heappop(openSet)[1]

        # client.simPlotPoints([Vector3r(*current)], duration = 60)

        if current == end:
            path = [current]
            while path[-1] != start:
                current = cameFrom[current]
                path.append(current)
            
            return list(reversed(path))

        for neighbor in occupancyMap.getNextSteps(current, end):

            # # skip neighbors from which the endpoint isn't visible
            # neighborOrientation = orientationAt(endpoint, neighbor)
            # if not isVisible(np.array(end), np.array(neighbor), neighborOrientation):
            #     continue

            tentativeGScore = gScore.get(current, float("inf")) + d(current, neighbor)

            if tentativeGScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = current
                gScore[neighbor]   = tentativeGScore

                if neighbor in fScore:
                    try:
                        openSet.remove((fScore[neighbor], neighbor))
                    except:
                        pass
                fScore[neighbor]   = gScore.get(neighbor, float('inf')) + h(neighbor, endpoint)

                heapq.heappush(openSet, (fScore[neighbor], neighbor))
        
    raise ValueError("Couldn't find a path")


def getTime():
    return 1e-9 * client.getMultirotorState().timestamp


def getPose():
    pose        = client.simGetVehiclePose()
    position    = pose.position.to_numpy_array() - CAMERA_OFFSET
    orientation = pose.orientation.to_numpy_array()
    return position, orientation


def isValidEndpoint(endpoint, occupancyMap):
    if endpoint in occupancyMap:
        return False

    position, orientation = getPose()
    if not isVisible(endpoint, position, orientation):
        return False

    # TODO(cvorbach) Check there is a valid path

    return True


def generateMazeTarget(occupancyMap, radius=50, zLimit=[-30, -10]):
    isValid = False
    attempts = 0
    while not isValid:
        endpoint = np.array([
            2 * radius * (random.random() - 0.5), 
            2 * radius * (random.random() - 0.5), 
            (zLimit[0] - zLimit[1]) * random.random() + zLimit[1]])

        # yawRotation = R.from_euler('xyz', [0, 0, R.from_quat(orientation).as_euler('xyz')[2]])
        # endpoint = position + yawRotation.apply(occupancyMap.point2Voxel(radius * normalize(np.array([random.random(), random.random(), -random.random()]) - 0.5)))
        # endpoint[2] = min(max(endpoint[2], zLimit[0]), zLimit[1])

        isValid = endpoint not in occupancyMap
        isValid = isValid and isValidEndpoint(endpoint, occupancyMap)

        attempts += 1
        if attempts > args.bogo_attempts:
            return None

    return endpoint


def generateTarget(occupancyMap, radius=10, zLimit=(-float('inf'), float('inf'))):
    isValid = False
    attempts = 0
    while not isValid:
        # TODO(cvorbach) smarter generation without creating points under terrain
        position, orientation = getPose()
        yawRotation = R.from_euler('xyz', [0, 0, R.from_quat(orientation).as_euler('xyz')[2]])

        endpoint = position + yawRotation.apply(occupancyMap.point2Voxel(radius * normalize(np.array([random.random(), 0.1*random.random(), -random.random()]))))

        # Altitude limit
        endpoint[2] = min(max(endpoint[2], zLimit[0]), zLimit[1])

        endpoint = occupancyMap.point2Voxel(endpoint)
        isValid = isValidEndpoint(endpoint, occupancyMap)

        attempts += 1
        if attempts > args.bogo_attempts:
            return None

    return np.array(endpoint)


def turnTowardEndpoint(endpoint, timeout=0.01):
    position, _ = getPose()
    displacement = endpoint - position

    endpointYaw = np.arctan2(displacement[1], displacement[0]) * RADIANS_2_DEGREES
    client.rotateToYawAsync(endpointYaw, timeout_sec = timeout).join()
    print("Turned toward endpoint")


def tryPlotting(lastPlotTime, occupancyMap):
    if getTime() < args.plot_period + lastPlotTime:
        return lastPlotTime

    # occupancyMap.plotOccupancies()

    print("Replotted :)")
    return getTime()


def updateOccupancies(occupancyMap):
    lidarData = client.getLidarData()
    lidarPoints = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
    if len(lidarPoints) >=3:
        lidarPoints = np.reshape(lidarPoints, (lidarPoints.shape[0] // 3, 3))

        for p in lidarPoints:
            occupancyMap.addPoint(p)

    # print("Lidar data added")


def getNearestPoint(trajectory, position):
    closestDist = None
    for i in range(len(trajectory)):
        point = trajectory[i]
        dist  = distance(point, position)

        if closestDist is None or dist < closestDist:
            closestDist = dist
            closestIdx  = i

    return closestIdx


def getProgress(trajectory, currentIdx, position):
    if len(trajectory) <= 1:
        raise Exception("Trajectory is too short")

    progress = 0
    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]
        p2 = trajectory[i+1]

        if i < currentIdx:
            progress += distance(p2, p1)
        else:
            partialProgress = (position - p1).dot(p2 - p1) / np.linalg.norm(p2 - p1)
            progress += partialProgress
            break

    return progress


def pursuitVelocity(trajectory):
    '''
    This function is kinda of a mess, but basically it implements 
    carrot following of a lookahead.

    The three steps are:
    1. Find the point on the path nearest to the drone as start of carrot
    2. Find points in front of and behind the look ahead point by arc length
    3. Linearly interpolate to find the lookahead point and chase it.
    '''

    position, _ = getPose()
    startIdx = getNearestPoint(trajectory, position)
    progress = getProgress(trajectory, startIdx, position)

    arcLength   = -progress
    pointBehind = trajectory[0]
    for i in range(1, len(trajectory)):
        pointAhead = trajectory[i]

        arcLength += distance(pointAhead, pointBehind)
        if arcLength > args.lookahead_distance:
            break

        pointBehind = pointAhead

    # if look ahead is past the end of the trajectory
    if np.array_equal(pointAhead, pointBehind): 
        lookAheadPoint = pointAhead

    else:
        behindWeight = (arcLength - args.lookahead_distance) / distance(pointAhead, pointBehind)
        aheadWeight = 1.0 - behindWeight

        # sanity check
        if not (0 <= aheadWeight <= 1 or 0 <= behindWeight <= 1):
            raise Exception("Invalid Interpolation Weights")

        lookAheadPoint = aheadWeight * pointAhead + behindWeight * pointBehind

    # Compute velocity to pursue lookahead point
    pursuitVector = lookAheadPoint - position
    pursuitVector = args.speed * normalize(pursuitVector)
    return pursuitVector


def getLookAhead(path, t, position, lookAhead, dt=1e-4):
    lookAheadPoint        = position  # TODO(cvorbach) increments always?
    lookAheadDisplacement = 0

    while np.linalg.norm(lookAheadDisplacement) < lookAhead:
        t += dt

        if t < 1:
            lookAheadDisplacement = lookAheadPoint - position
            lookAheadPoint = path(t)
        else:
            t = 1
            lookAheadPoint = path(t)
            break
            # TODO(cvorbach) unnecessary with extrapolation
        
    return t, lookAheadPoint

class RunFailure(Exception):
    pass

def followPath(path, lookAhead = 2, dt = 1e-4, marker=None, earlyStopDistance=None, planningWrapper=None, planningKnots=None, recordingEndpoint=None, model=None):
    startTime       = getTime()
    position, _     = getPose()
    t               = path.project(position) # find the new nearest path(t)
    lookAheadPoint  = path(t)
    reachedEnd      = False

    lastPlotTime    = getTime()
    markerPose      = airsim.Pose()

    planningThread  = None
    controlThread   = None

    print('started following path')

    if args.plot_debug:
        client.simPlotPoints([Vector3r(*path(t)) for t in np.linspace(0, 1, 1000)], color_rgba = [0.0, 0.0, 1.0, 1.0], duration = 60)

    if args.record and args.task != Task.HIKING:
        client.startRecording()

    endpointDirections = []
    imagesBuffer       = np.zeros((1, args.seq_len, *IMAGE_SHAPE))
    gpsBuffer          = np.zeros((1, args.seq_len, 3))
    numBufferEntries   = 0

    # control loop
    lastVelocity = None
    alpha        = 1.0
    while not reachedEnd:
        position, orientation = getPose()
        updateOccupancies(occupancyMap)

        # handle timeout and collision check
        if startTime + args.timeout < getTime():
            raise RunFailure('Run timed out.')

        if client.simGetCollisionInfo().has_collided:
            raise RunFailure('Crashed.')

        # handle planning thread if needed
        if planningWrapper is not None:

            # if we have finished planning
            if planningThread is None or not planningThread.is_alive():

                # update the spline path
                if np.any(planningKnots != path.knotPoints):
                    path.fit(planningKnots.copy())
                    t = path.project(position) # find the new nearest path(t)

                # restart planning
                planningThread = threading.Thread(target=planningWrapper, args=(planningKnots,))
                planningThread.start()

            planningThread.join(timeout=args.control_period)

        # place marker if passed
        if marker is not None:
            markerT, markerPosition = getLookAhead(path, t, position, lookAhead)

           #  tangent  = normalize(path.tangent(markerT))
           #  normal   = normalize(np.cross(tangent, (0, 0, 1)))

           #  inclinationAngle = np.pi + MAX_INCLINATION * (1 - np.abs(tangent[2])) # pi is b/c the drone is upside down at 0 inclination
           #  inclination = R.from_rotvec(inclinationAngle*normal)
           #  tangent     = inclination.apply([tangent[0], tangent[1], 0])

           #  binormal = normalize(np.cross(tangent, normal))

           #  markerOrientation = R.from_matrix(np.array([tangent, normal, binormal]).T)

           #  markerPose.orientation = Quaternionr(*markerOrientation.as_quat())
            markerPose.position    = Vector3r(*markerPosition)
            client.simSetObjectPose(marker, markerPose)

        # advance the pursuit point if needed
        # TODO(cvorbach) move to its own thread
        t, lookAheadPoint = getLookAhead(path, t, position, lookAhead)
        lookAheadDisplacement = lookAheadPoint - position
        if t > 1:
            reachedEnd = True
            break

        if model is None:
            # optionally stop within distance of path end
            if earlyStopDistance is not None:
                if distance(path(1.0), position) < earlyStopDistance:
                    reachedEnd = True

            if reachedEnd:
                break

            # get yaw angle and pursuit velocity to the lookAheadPoint
            endpointDisplacement = path(1.0) - position
            yawAngle = np.arctan2(endpointDisplacement[1], endpointDisplacement[0]) * RADIANS_2_DEGREES

            if lastVelocity is None:
                velocity = args.speed * normalize(lookAheadDisplacement)
            else:
                velocity = args.speed * normalize(lookAheadDisplacement)
                velocity = alpha * velocity + (1-alpha)*lastVelocity
            lastVelocity = velocity

            # plot
            if args.plot_debug:
                lastPlotTime = tryPlotting(lastPlotTime, occupancyMap)

            # record direction vector to endpoint if needed 
            if args.record and recordingEndpoint is not None:
                endpointDirections.append((time.time(), *normalize(recordingEndpoint - position)))

            # start control thread
            if controlThread is not None:
                controlThread.join()
            controlThread = client.moveByVelocityAsync(float(velocity[0]), float(velocity[1]), float(velocity[2]), args.control_period, yaw_mode=YawMode(is_rate = False, yaw_or_rate = yawAngle))

        # If we are flying by a model
        else:
            # place marker if passed
            if marker is not None:
                markerPose.position = Vector3r(*lookAheadPoint)
                client.simSetObjectPose(marker, markerPose)

            # get and format an image
            image = None
            while image is None or len(image) == 1:
                image = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
                image = np.fromstring(image.image_data_uint8, dtype=np.uint8).astype(np.float32) / 255
            image = np.reshape(image, IMAGE_SHAPE)
            image = image[:, :, ::-1]                 # Required since order is BGR instead of RGB by default

            # add the image to the sliding window
            if numBufferEntries < args.seq_len:
                imagesBuffer[0, numBufferEntries] = image
                # gpsBuffer[0, numBufferEntries]    = gpsDirection
                numBufferEntries += 1
            else:
                imagesBuffer[0]     = np.roll(imagesBuffer[0], -1, axis=0)
                imagesBuffer[0][-1] = image

            # compute a velocity vector from the model
            prediction = model.predict(imagesBuffer)[0][numBufferEntries-1]
            direction  = normalize(prediction)
            direction  = R.from_quat(orientation).apply(direction) # Transform from the drone camera's reference frame to static coordinates
            velocity   = args.speed * direction

            yawRotation = R.from_euler('xyz', [0, 0, R.from_quat(orientation).as_euler('xyz')[2]])

            # get yaw angle to the endpoint
            lookAheadDisplacement = lookAheadPoint - position
            yawAngle = np.arctan2(lookAheadDisplacement[1], lookAheadDisplacement[0]) * RADIANS_2_DEGREES

            endpointDisplacement = path(1.0) - position
            yawAngle = np.arctan2(endpointDisplacement[1], endpointDisplacement[0]) * RADIANS_2_DEGREES

            # check if we've reached the endpoint
            if np.linalg.norm(endpointDisplacement) < args.endpoint_tolerance:
                reachedEnd = True
                break

            # start control thread
            if controlThread is not None:
                controlThread.join()
            controlThread = client.moveByVelocityAsync(float(velocity[0]), float(velocity[1]), float(velocity[2]), args.control_period, yaw_mode=YawMode(is_rate = False, yaw_or_rate = yawAngle))

    # hide the marker
    if marker is not None:
        markerPose.position = Vector3r(0,0,100)
        client.simSetObjectPose(marker, markerPose)

    if args.record and args.task != Task.HIKING:
        client.stopRecording()

    # Write out the direction vectors if needed
    if args.record and recordingEndpoint is not None:
        recordingDir = sorted([d for d in os.listdir(RECORDING_DIRECTORY) if RECORDING_NAME_REGEX.match(d)])[-1]
        with open(RECORDING_DIRECTORY + '/' + recordingDir + '/endpoint_directions.txt', 'w') as f:
            endpointFileWriter = csv.writer(f) 
            endpointFileWriter.writerows(endpointDirections)


def moveToEndpoint(endpoint, occupancyMap, recordEndpointDirection=False, model=None):
    updateOccupancies(occupancyMap)

    position, _  = getPose()
    print('first planning')
    pathKnots    = findPath(position, endpoint, occupancyMap)
    print('finshed first planning')
    pathToEndpoint = Path(pathKnots.copy())

    def planningWrapper(knots):
        # run planning
        newKnots = findPath(position, endpoint, occupancyMap)

        # replace the old knots
        knots.clear()
        for k in newKnots:
            knots.append(k)
            
        # print('Finished Planning')

    if recordEndpointDirection:
        recordingEndpoint = endpoint
    else:
        recordingEndpoint = None

    followPath(pathToEndpoint, earlyStopDistance=args.endpoint_tolerance, planningWrapper=planningWrapper, planningKnots=pathKnots, recordingEndpoint=recordingEndpoint, model=model)
    print('Reached Endpoint')


def checkHalfSpace(testPoint, p):
    '''
    Checks that testPoint is in the half space defined by
    point p and normal vector (p - x) / ||p - x||
    where x is the current posistion of the drone
    '''
    
    x, _ = getPose()

    return (testPoint - x).dot(normalize(p - x)) > np.linalg.norm(p - x)


def checkBand(k, blazes, blazeStart, zLimit):

    band = []

    # check sides of each square of radius k between zLimits
    for z in reversed(range(zLimit[0], zLimit[1])):
        corners = np.array([
            (blazeStart[0] - k, blazeStart[1] - k, z),
            (blazeStart[0] + k, blazeStart[1] - k, z),
            (blazeStart[0] + k, blazeStart[1] + k, z),
            (blazeStart[0] - k, blazeStart[1] + k, z)
        ])

        tangentVectors = np.array([
            (1, 0, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (0, -1, 0)
        ])

        sideLength = 2*k-1

        # check each side
        for corner, tangent in zip(corners, tangentVectors):
            for i in range(sideLength):
                voxel = occupancyMap.point2Voxel(corner + i * tangent)
                band.append(voxel)

                isOccupied    = voxel in occupancyMap
                isSpacedOut   = not np.any([distance(np.array(b), voxel) < args.min_blaze_gap for b in blazes])
                isInHalfSpace = checkHalfSpace(np.array(voxel), np.array(blazeStart))

                if isOccupied and isSpacedOut and isInHalfSpace:
                    return voxel

    # client.simPlotPoints([Vector3r(*v) for v in band])
    return None


def generateHikingBlazes(start, occupancyMap, numBlazes = 2, zLimit=(-15, -1), maxSearchDepth=50):
    blazes = []
    blazeStart = occupancyMap.point2Voxel(start)

    # lastPlotTime = tryPlotting(-float('inf'), occupancyMap)

    for _ in range(numBlazes):
        nextBlaze   = None
        
        k = 5
        while nextBlaze is None and k < maxSearchDepth: 
            nextBlaze = checkBand(k, blazes, blazeStart, zLimit)
            k += 1

        if k == maxSearchDepth:
            raise Exception('Could not find a tree to blaze')

        blazes.append(nextBlaze)
        blazeStart = blazes[-1]

    return blazes


# -----------------------------
# MAIN
# -----------------------------

# Takeoff
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-10, 1).join()
# print("Taken off")

occupancyMap = VoxelOccupancyCache(args.voxel_size, args.cache_size)

# get the markers
markers = client.simListSceneObjects('Red_Cube.*') 

quadcopterLeader = client.simListSceneObjects('QuadcopterLeader.*')[0]

if len(markers) < 1:
    raise Exception('Didn\'t find any endpoint markers. Check there is a Red_Cube is in the scene')

# start the markers out of the way
markerPose = airsim.Pose()
markerPose.position = Vector3r(0, 0, 100)
for marker in markers:
    client.simSetObjectPose(marker, markerPose)

# The starting pose of each run
startingPose = client.simGetVehiclePose()

# Collect data runs
successes = 0
for i in range(args.n_runs):
    client.simSetVehiclePose(            
        startingPose,
        True
    )
    position, orientation = getPose()
    time.sleep(2)
    
    try:
        if args.task == Task.TARGET:
            marker = markers[0]

            print('here 1')

            # Random rotation
            client.rotateToYawAsync(random.random() * 2.0 * np.pi * RADIANS_2_DEGREES).join()

            print('here 2')

            # Set up
            endpoint = generateMazeTarget(occupancyMap, radius=args.near_task_radius, zLimit=(-5, -15))
            if endpoint is None:
                continue

            print('here 3')
                
            # place endpoint marker
            endpointPose = airsim.Pose()
            endpointPose.position = Vector3r(*endpoint)
            client.simSetObjectPose(marker, endpointPose)

            turnTowardEndpoint(endpoint, timeout=10)

            moveToEndpoint(endpoint, occupancyMap, model=flightModel)

        if args.task == Task.FOLLOWING:

            marker = markers[0]

            updateOccupancies(occupancyMap)
            print('updated occupancies')

            # TODO(cvorbach) Online path construction with collision checking along each spline length
            walk = randomWalk(position, stepSize=5, occupancyMap=occupancyMap)
            path = Path(walk)
            # path = ExtendablePath(walk)

            # t = np.linspace(0, 1, 1000) 
            # client.simPlotPoints([Vector3r(*path(t_i)) for t_i in t], color_rgba = [0.0, 0.0, 1.0, 1.0], duration = 60)
            # sys.exit()

            print('got path')

            followPath(path, marker=marker, model=flightModel, earlyStopDistance=args.endpoint_tolerance)
            print('reached path end')

        if args.task == Task.HIKING:
            updateOccupancies(occupancyMap)
            print('updated occupancies')

            newStart = generateMazeTarget(occupancyMap, radius=args.near_task_radius)
            moveToEndpoint(newStart, occupancyMap)

            print('move to z-level')
            zLimit = (-10, -5)
            client.moveToZAsync((zLimit[0] + zLimit[1])/2, 1).join()
            print('reached z-level')

            print('getting blazes')
            hikingBlazes = generateHikingBlazes(position, occupancyMap, zLimit=zLimit)
            turnTowardEndpoint(hikingBlazes[0], timeout=10)
            print('Got blazes')

            if len(hikingBlazes) > len(markers):
                raise Exception('Not enough markers for each blaze to get one')

            # place makers
            markerPose = airsim.Pose()
            for i, blaze in enumerate(hikingBlazes):
                print('placed blaze', i)
                markerPose.position = Vector3r(*blaze)
                client.simSetObjectPose(markers[i], markerPose)

            if args.record:
                client.startRecording()

            for blaze in hikingBlazes:
                print('moving to blaze')
                moveToEndpoint(blaze, occupancyMap, model=flightModel)
                # TODO(cvorbach) rotate 360
                print('moved to blaze')

            if args.record:
                client.stopRecording()

        if args.task == Task.MAZE:
            # TODO(cvorbach) reimplement me
            # record the direction vector
            marker = markers[0]

            # Set up
            endpoint = generateMazeTarget(occupancyMap, radius=args.far_task_radius, zLimit=(-5, -15))
            if endpoint is None:
                continue
                
            # place endpoint marker
            endpointPose = airsim.Pose()
            endpointPose.position = Vector3r(*endpoint)
            client.simSetObjectPose(marker, endpointPose)

            turnTowardEndpoint(endpoint, timeout=10)

            moveToEndpoint(endpoint, occupancyMap, recordEndpointDirection=True, model=flightModel)

        successes += 1
        print(f'Run {i} succeeded: {successes}/{i+1} success rate.')
    except RunFailure:
        print(f'Run {i} failed:    {successes}/{i+1} success rate.')


print('Finished Data Runs')