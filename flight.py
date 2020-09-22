# drone-flight  Copyright (C) 2020  Charles Vorbach
import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr, YawMode

import sys 
import time 
import random 
import numpy as np
import pprint
import heapq
import pickle
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from tensorflow import keras
import kerasncp as kncp

# Start up
client = airsim.MultirotorClient() 
client.confirmConnection() 
client.enableApiControl(True) 

# Constants
ENDPOINT_TOLERANCE    = 1.5       # m
ENDPOINT_RADIUS       = 10        # m
PLOT_PERIOD           = 10.0      # s
PLOT_DELAY            = 3.0       # s
CONTROL_PERIOD        = 1.0       # s
SPEED                 = 0.5       # m/s
YAW_TIMEOUT           = 0.1       # s
VOXEL_SIZE            = 1.0       # m
LOOK_AHEAD_DIST       = 1.0       # m
MAX_ENDPOINT_ATTEMPTS = 50
N_RUNS                = 1000
ENABLE_PLOTTING       = True
ENABLE_RECORDING      = False
FLY_BY_MODEL          = False

CAMERA_FOV = np.pi / 6
RADIANS_2_DEGREES = 180 / np.pi
CAMERA_OFFSET = np.array([0.5, 0, -0.5])

ENDPOINT_OFFSET = np.array([0, -0.5, 0.5])
#TODO(cvorbach) CAMERA_HORIZONTAL_OFFSET

# Setup the network
TRAINING_SEQUENCE_LENGTH = 1
IMAGE_SHAPE              = (256,256,3)

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

model = keras.models.Sequential()
model.add(keras.Input(shape=(None, *IMAGE_SHAPE)))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=48, kernel_size=(3,3), strides=(2,2), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1000, activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.5)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=100,  activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(rate=0.3)))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=24,   activation='relu')))
model.add(keras.layers.RNN(rnnCell, return_sequences=True))

model.compile(
    optimizer=keras.optimizers.Adam(0.00005), loss="cosine_similarity",
)

# Load weights
if FLY_BY_MODEL:
    model.load_weights('model-checkpoints/weights.132--0.91.hdf5')

# Utilities
class SparseVoxelOccupancyMap:
    def __init__(self, voxelSize):
        self.voxelSize = voxelSize
        self.occupiedVoxels = set()

    def addPoint(self, point):
        voxel = self.point2Voxel(point)
        self.occupiedVoxels.add(voxel)

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1] - 1, voxel[2] - 1))
        self.occupiedVoxels.add((voxel[0],     voxel[1] - 1, voxel[2] - 1))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1] - 1, voxel[2] - 1))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1], voxel[2] - 1))
        self.occupiedVoxels.add((voxel[0],     voxel[1], voxel[2] - 1))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1], voxel[2] - 1))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1] + 1, voxel[2] - 1))
        self.occupiedVoxels.add((voxel[0],     voxel[1] + 1, voxel[2] - 1))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1] + 1, voxel[2] - 1))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1] - 1, voxel[2]))
        self.occupiedVoxels.add((voxel[0],     voxel[1] - 1, voxel[2]))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1] - 1, voxel[2]))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1], voxel[2]))
        self.occupiedVoxels.add((voxel[0],     voxel[1], voxel[2]))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1], voxel[2]))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1] + 1, voxel[2]))
        self.occupiedVoxels.add((voxel[0],     voxel[1] + 1, voxel[2]))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1] + 1, voxel[2]))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1] - 1, voxel[2] + 1))
        self.occupiedVoxels.add((voxel[0],     voxel[1] - 1, voxel[2] + 1))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1] - 1, voxel[2] + 1))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1], voxel[2] + 1))
        self.occupiedVoxels.add((voxel[0],     voxel[1], voxel[2] + 1))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1], voxel[2] + 1))

        self.occupiedVoxels.add((voxel[0] - 1, voxel[1] + 1, voxel[2] + 1))
        self.occupiedVoxels.add((voxel[0],     voxel[1] + 1, voxel[2] + 1))
        self.occupiedVoxels.add((voxel[0] + 1, voxel[1] + 1, voxel[2] + 1))
        
    def isOccupied(self, point):
        voxel = self.point2Voxel(point)
        return voxel in self.occupiedVoxels
    
    def point2Voxel(self, point):
        return tuple(self.voxelSize * int(round(v / self.voxelSize)) for v in point)

    def getNeighbors(self, voxel):
        neighbors = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == 0 and j == 0 and k == 0:
                        continue

                    neighboringVoxel = tuple(self.voxelSize * np.array([i, j, k])  + voxel)
                    if neighboringVoxel not in self.occupiedVoxels:
                        neighbors.append(neighboringVoxel)

        return neighbors
    
    def plotOccupancies(self):
        occupiedPoints = [Vector3r(float(v[0]), float(v[1]), float(v[2])) for v in self.occupiedVoxels]
        client.simPlotPoints(occupiedPoints, color_rgba = [0.0, 0.0, 1.0, 1.0], duration=PLOT_PERIOD-PLOT_DELAY) 


def isVisible(point, position, orientation):
    # Edge case point == position
    if np.linalg.norm(point - position) < 0.05:
        return True

    # Check if endpoint is in frustrum
    xUnit = np.array([1, 0, 0])
    cameraDirection = R.from_quat(orientation).apply(xUnit)

    endpointDirection = point - position
    endpointDirection /= np.linalg.norm(endpointDirection)

    # TODO(cvorbach) check square, not circle
    angle = np.arccos(np.dot(cameraDirection, endpointDirection)) 

    if abs(angle) > CAMERA_FOV:
        return False

    # TODO(cvorbach) Check for occlusions with ray-tracing

    return True


def orientationAt(endpoint, position):
    # Get the drone orientation that faces towards the endpoint at position
    displacement = endpoint - position
    endpointYaw = np.arctan2(displacement[1], displacement[0])
    orientation = R.from_euler('xyz', [0, 0, endpointYaw]).as_quat()

    return orientation


def h(voxel, map, endpoint):
    return np.linalg.norm(endpoint - np.array(voxel))

def d(voxel1, voxel2, map):
    return np.linalg.norm(np.array(voxel2) - np.array(voxel1))

# A* Path finding 
def findPath(startpoint, endpoint, map):
    start = map.point2Voxel(startpoint)
    end   = map.point2Voxel(endpoint)

    cameFrom = dict()

    gScore = dict()
    gScore[start] = 0

    fScore = dict()
    fScore[start] = h(start, map, endpoint)

    openSet = [(fScore[start], start)]

    while openSet:
        current = heapq.heappop(openSet)[1]

        if current == end:
            path = [current]
            while path[-1] != start:
                current = cameFrom[current]
                path.append(current)
            
            return list(reversed(path))

        for neighbor in map.getNeighbors(current):
            # skip neighbors from which the endpoint isn't visible
            neighborOrientation = orientationAt(endpoint, neighbor)
            if not isVisible(np.array(end), np.array(neighbor), neighborOrientation):
                continue

            tentativeGScore = gScore.get(current, float("inf")) + d(current, neighbor, map)

            if tentativeGScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = current
                gScore[neighbor]   = tentativeGScore

                if neighbor in fScore:
                    try:
                        openSet.remove((fScore[neighbor], neighbor))
                    except:
                        pass
                fScore[neighbor]   = gScore.get(neighbor, float('inf')) + h(neighbor, map, endpoint)

                heapq.heappush(openSet, (fScore[neighbor], neighbor))
        
    raise ValueError("Couldn't find a path")


def getTime():
    return 1e-9 * client.getMultirotorState().timestamp


def getPose():
    pose        = client.simGetVehiclePose()
    position    = pose.position.to_numpy_array() - CAMERA_OFFSET
    orientation = pose.orientation.to_numpy_array()
    return position, orientation


def isValidEndpoint(endpoint, map):
    if map.isOccupied(endpoint):
        return False

    position, orientation = getPose()
    if not isVisible(endpoint, position, orientation):
        return False

    # TODO(cvorbach) Check there is a valid path

    return True


def generateEndpoint(map):
    isValid = False
    attempts = 0
    while not isValid:
        endpoint = np.array([1, random.random() * np.cos(CAMERA_FOV), random.random() * np.cos(CAMERA_FOV)])        # generate point in field of view
        endpoint = ENDPOINT_RADIUS * random.random() * R.from_quat(getPose()[1]).apply(endpoint) - ENDPOINT_OFFSET  # place out in world NED coordinates
        endpoint = map.point2Voxel(endpoint)
        isValid = isValidEndpoint(endpoint, map)

        attempts += 1
        if attempts > MAX_ENDPOINT_ATTEMPTS:
            return None

    return np.array(endpoint)


def turnTowardEndpoint(endpoint, timeout=0.01):
    position, _ = getPose()
    displacement = endpoint - position

    endpointYaw = np.arctan2(displacement[1], displacement[0]) * RADIANS_2_DEGREES
    client.rotateToYawAsync(endpointYaw, timeout_sec = timeout).join()
    print("Turned toward endpoint")


def tryPlotting(lastPlotTime, trajectory, map):
    if getTime() < PLOT_PERIOD + lastPlotTime:
        return lastPlotTime

    plottableTrajectory = [Vector3r(float(trajectory[i][0]), float(trajectory[i][1]), float(trajectory[i][2])) for i in range(len(trajectory))]
    client.simPlotPoints(plottableTrajectory, color_rgba = [0.0, 1.0, 0.0, 1.0], duration=PLOT_PERIOD-PLOT_DELAY) 

    map.plotOccupancies()

    print("Replotted :)")
    return getTime()


def updateOccupancies(map):
    lidarData = client.getLidarData()
    lidarPoints = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
    if len(lidarPoints) >=3:
        lidarPoints = np.reshape(lidarPoints, (lidarPoints.shape[0] // 3, 3))

        for p in lidarPoints:
            map.addPoint(p)

    print("Lidar data added")


def pursuitVelocity(trajectory):
    # Find lookahead point
    position, _ = getPose()
    lookAheadPoint = trajectory[-1]
    for i in range(1, len(trajectory)):
        lookAheadPoint = trajectory[i]
        if np.linalg.norm(lookAheadPoint - position) > LOOK_AHEAD_DIST: # TODO(cvorbach) interpolate properly
            break

    # Compute velocity to pursue lookahead point
    pursuitVector = lookAheadPoint - position
    pursuitVector = SPEED / np.linalg.norm(pursuitVector) * pursuitVector

    return pursuitVector


def moveToEndpoint(endpoint, model = None):
    controlThread   = None
    reachedEndpoint = False
    lastPlotTime    = 0

    images = np.zeros((1, TRAINING_SEQUENCE_LENGTH, *IMAGE_SHAPE))

    i = 0
    while not reachedEndpoint:
        position, _ = getPose()

        if map.isOccupied(endpoint):
            print("Endpoint is occupied")
            break

        if map.isOccupied(position):
            print("Drone in occupied position")

        updateOccupancies(map)

        if model is None:
            trajectory = findPath(position, endpoint, map)
            velocity   = pursuitVelocity(trajectory)
        else:
            # get an image
            image = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            image = np.fromstring(image.image_data_uint8, dtype=np.uint8).astype(np.float32) / 255
            image = np.reshape(image, IMAGE_SHAPE)
            image = image[:, :, ::-1]                 # Required since order is BGR instead of RGB by default

            images[0, i % TRAINING_SEQUENCE_LENGTH] = image

            direction = model.predict(images)[0][0]
            direction = direction / np.linalg.norm(direction)
            print(direction)
            velocity = SPEED * direction

        if ENABLE_PLOTTING:
            lastPlotTime = tryPlotting(lastPlotTime, trajectory, map)

        displacement = endpoint - position
        endpointYaw = np.arctan2(displacement[1], displacement[0]) * RADIANS_2_DEGREES

        if controlThread is not None:
            controlThread.join()
        controlThread = client.moveByVelocityAsync(float(velocity[0]), float(velocity[1]), float(velocity[2]), CONTROL_PERIOD, yaw_mode=YawMode(is_rate = False, yaw_or_rate = endpointYaw))
        print("Moving")
        
        reachedEndpoint = np.linalg.norm(endpoint - position) <= ENDPOINT_TOLERANCE

        i += 1

    # Finish moving
    if controlThread is not None:
        controlThread.join()


# -----------------------------
# MAIN
# -----------------------------

# Takeoff
client.armDisarm(True)
client.takeoffAsync().join()
client.takeoffAsync().join()
print("Taken off")

# Try to load saved occupancy map
try:
    with open("occupancy_map.p", 'rb') as f:
        map = pickle.load(f)
except:
    map = SparseVoxelOccupancyMap(VOXEL_SIZE)
 
# Collect data runs
for i in range(N_RUNS):
    # Random rotation
    client.rotateToYawAsync(random.random() * 2.0 * np.pi * RADIANS_2_DEGREES).join()

    # Set up
    endpoint = generateEndpoint(map)
    if endpoint is None:
        continue
    client.simPlotPoints([Vector3r(endpoint[0], endpoint[1], endpoint[2])], size=100, is_persistent = True) 

    turnTowardEndpoint(endpoint, timeout=10.0)

    if ENABLE_RECORDING:
        client.startRecording()

    # Control loop
    try:
        if FLY_BY_MODEL:
            moveToEndpoint(endpoint, model=model)
        else:
            moveToEndpoint(endpoint)

    # Clean up
    finally:
        if ENABLE_RECORDING:
            client.stopRecording()
            time.sleep(0.2) # wait a short period for the camera to really turn off

        client.simFlushPersistentMarkers()

# Dump saved map
with open('occupancy_map.p', 'wb') as f:
    pickle.dump(map, f)
