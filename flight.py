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
import pickle
import matplotlib.pyplot as plt
import cv2
import os
import csv
import re
import argparse
from enum import Enum

from scipy.spatial.transform import Rotation as R

from planning import *

# Start up
client = airsim.MultirotorClient() 
client.confirmConnection() 
client.enableApiControl(True) 

# Weather
client.simEnableWeather(True)
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0)
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.25)

# Operating Modes
class Task: 
    TARGET = 'target'
    FOLLOWING = 'following'
    MAZE = 'maze'
    HIKING = 'hiking'
    DEMO = 'demo'

# Parameters
parser = argparse.ArgumentParser(description='Fly the deepdrone agent in the Airsim simulator')
parser.add_argument('--task',               type=str,   default='target', help='Task to attempt')
parser.add_argument('--endpoint_tolerance', type=float, default=2.0,      help='The distance tolerance on reaching the endpoint marker')
parser.add_argument('--near_task_radius',   type=float, default=10.0,     help='The max distance of endpoints in the near planning task')
parser.add_argument('--far_task_radius',    type=float, default=50.0,     help='The max distance of endpoints in the far planning task')
parser.add_argument('--min_blaze_gap',      type=float, default=10.0,     help='The minimum distance between hiking task blazes')
parser.add_argument('--plot_period',        type=float, default=0.5,      help='The time between updates of debug plotting information')
parser.add_argument('--control_period',     type=float, default=1.5,      help='Update frequency of the pure pursuit controller')
parser.add_argument('--speed',              type=float, default=0.5,      help='Drone flying speed')
parser.add_argument('--voxel_size',         type=float, default=1.0,      help='The size of voxels in the occupancy map cache')
parser.add_argument('--cache_size',         type=float, default=100000,   help='The number of entries in the local occupancy cache')
parser.add_argument('--lookahead_distance', type=float, default=0.75,      help='Pure pursuit lookahead distance')
parser.add_argument('--bogo_attempts',      type=int,   default=5000,     help='Number of attempts to make in generate and test algorithms')
parser.add_argument('--n_runs',             type=int,   default=1000,       help='Number of repetitions of the task to attempt')
parser.add_argument("--plot_debug", dest="plot_debug", action="store_true")
parser.set_defaults(gps_signal=False)
parser.add_argument('--record', dest='record', action='store_true')
parser.set_defaults(record=False)
parser.add_argument('--autoKill', dest='record', action='store_true')
parser.set_defaults(autoKill=False)
parser.add_argument('--model_weights', type=str, default=None, help='Model weights to load and fly with')
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--rnn_size', type=int, default=32, help='Select the size of RNN network you would like to train')
parser.add_argument('--timeout', type=int, default=30)
args = parser.parse_args()

RECORDING_DIRECTORY    = 'C:/Users/MIT Driverless/Documents/AirSim'
RECORDING_NAME_REGEX   = re.compile(r'^[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+$')

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
        if args.autoKill and startTime + args.timeout < getTime():
            raise RunFailure('Run timed out.')

        if args.autoKill and client.simGetCollisionInfo().has_collided:
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
        # print('t:', t)
        if t >= 1:
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
    pathKnots    = findPath(position, endpoint, occupancyMap, args.endpoint_tolerance)
    print('finshed first planning')
    pathToEndpoint = Path(pathKnots.copy())

    def planningWrapper(knots):
        # run planning
        newKnots = findPath(position, endpoint, occupancyMap, args.endpoint_tolerance)

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
    #print('S:', [startingPose.position.x_val, startingPose.position.y_val, startingPose.position.z_val])
    # client.simSetVehiclePose(            
    #     startingPose,
    #     True
    # )
    time.sleep(5)
    position, orientation = getPose()
    
    try:

        if args.task == Task.DEMO:
            loop = generateLoop()

            t = np.linspace(0, 1, 1000) 
            if args.plot_debug:
                client.simPlotPoints([Vector3r(*loop(t_i)) for t_i in t], color_rgba = [0.0, 0.0, 1.0, 1.0], duration = 60)

            # Move camera
            rotation = R.from_rotvec(3*np.pi/2 * np.array([0,0,1]))  
            camera_pose = airsim.Pose(Vector3r(0,8,0), Quaternionr(*rotation.as_quat())) 
            client.simSetCameraPose('0', camera_pose)

            moveToEndpoint(loop(0), occupancyMap)

            if args.record:
                client.startRecording()

            followPath(loop)

            if args.record:
                client.stopRecording()

        if args.task == Task.TARGET:
            marker = markers[0]

            # print('here 1')

            # Random rotation
            client.rotateToYawAsync(random.random() * 2.0 * np.pi * RADIANS_2_DEGREES).join()

            # print('here 2')

            # Set up
            endpoint = generateMazeTarget(occupancyMap, radius=args.near_task_radius, zLimit=(-5, -15))
            if endpoint is None:
                continue

            # print('here 3')
                
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
    #except RunFailure:
    except Exception as e:
        raise e
        print(f'Run {i} failed:    {successes}/{i+1} success rate.')


print('Finished Data Runs')