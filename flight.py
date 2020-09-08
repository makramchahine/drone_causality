import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr

import sys 
import time 
import random 
import numpy as np
import pprint
import heapq
import pickle

from scipy.spatial.transform import Rotation as R

# Images to collect
imagesRequests = [
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanner, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_right",  airsim.ImageType.DepthPlanner, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_left",   airsim.ImageType.DepthPlanner, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("fpv",          airsim.ImageType.DepthPlanner, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("back_center",  airsim.ImageType.DepthPlanner, pixels_as_float = False, compress = True), 

    airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_right",  airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_left",   airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("fpv",          airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("back_center",  airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True),

    airsim.ImageRequest("front_center", airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_right",  airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_left",   airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("fpv",          airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("back_center",  airsim.ImageType.DepthVis, pixels_as_float = False, compress = True),

    airsim.ImageRequest("front_center", airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_right",  airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_left",   airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("fpv",          airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("back_center",  airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True),

    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_right",  airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_left",   airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("fpv",          airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("back_center",  airsim.ImageType.Segmentation, pixels_as_float = False, compress = True),

    airsim.ImageRequest("front_center", airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_right",  airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_left",   airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("fpv",          airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("back_center",  airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True),

    airsim.ImageRequest("front_center", airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_right",  airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("front_left",   airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("fpv",          airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
    airsim.ImageRequest("back_center",  airsim.ImageType.Infrared, pixels_as_float = False, compress = True),  
]

# Start up
client = airsim.MultirotorClient() 
client.confirmConnection() 
client.enableApiControl(True) 

# Constants
ENDPOINT_TOLERANCE    = 2.5       # m
ENDPOINT_RADIUS       = 15        # m
PLOT_PERIOD           = 10.0      # s
PLOT_DELAY            = 3.0       # s
CONTROL_PERIOD        = 0.75      # s
SPEED                 = 0.5       # m/s
YAW_TIMEOUT           = 0.1       # s
VOXEL_SIZE            = 1.0       # m
LOOK_AHEAD_DIST       = 1.0       # m
MAX_ENDPOINT_ATTEMPTS = 50
N_RUNS                = 1000
ENABLE_PLOTTING       = False
ENABLE_RECORDING      = True

CAMERA_FOV = np.pi / 6
RADIANS_2_DEGREES = 180 / np.pi
CAMERA_VERTICAL_OFFSET = np.array([0, 0, -0.5])
#TODO(cvorbach) CAMERA_HORIZONTAL_OFFSET

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
    position    = pose.position.to_numpy_array() - CAMERA_VERTICAL_OFFSET
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
        # TODO(cvorbach) generate points already in the camera frustrum
        endpoint = map.point2Voxel(ENDPOINT_RADIUS * np.array([random.random(), random.random(), -random.random()]))
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


def moveToEndpoint(endpoint):
    controlThread   = None
    reachedEndpoint = False
    lastPlotTime    = 0

    while not reachedEndpoint:
        position, _ = getPose()

        if map.isOccupied(endpoint):
            print("Endpoint is occupied")
            break

        if map.isOccupied(position):
            print("Drone in occupied position")

        updateOccupancies(map)

        trajectory = findPath(position, endpoint, map)
        velocity   = pursuitVelocity(trajectory)

        if ENABLE_PLOTTING:
            lastPlotTime = tryPlotting(lastPlotTime, trajectory, map)

        if controlThread is not None:
            controlThread.join()

        turnTowardEndpoint(endpoint)

        controlThread = client.moveByVelocityAsync(float(velocity[0]), float(velocity[1]), float(velocity[2]), CONTROL_PERIOD)
        print("Moving")
        
        reachedEndpoint = np.linalg.norm(endpoint - position) <= ENDPOINT_TOLERANCE

    # Finish moving
    if controlThread is not None:
        controlThread.join()

# -----------------------------
# MAIN
# -----------------------------

# Takeoff
client.armDisarm(True)
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
    moveToEndpoint(endpoint)

    # Clean up
    if ENABLE_RECORDING:
        client.stopRecording()

    with open('occupancy_map.p', 'wb') as f:
        pickle.dump(map, f)

    client.simFlushPersistentMarkers()

#     time.sleep(0.1)
#   
# client. rmDisarm(False)
# 
# print(type(images[0]))

# for i,  mg in enumerate(images):
# 
#     views = [
#         '_front_center_Scene',
#         '_front_right_Scene',
#         '_front_left_Scene',
#         '_fpv_Scene',
#         '_back_center_Scene',
#         '_front_center_DepthPlanner',
#         '_front_right_DepthPlanner',
#         '_front_left_DepthPlanner',
#         '_fpv_DepthPlanner',
#         '_back_center_DepthPlanner',
#         '_front_center_DepthPerspective',
#         '_front_right_DepthPerspective',
#         '_front_left_DepthPerspective',
#         '_fpv_DepthPerspective',
#         '_back_center_DepthPerspective',
#         '_front_center_DepthVis',
#         '_front_right_DepthVis',
#         '_front_left_DepthVis',
#         '_fpv_DepthVis',
#         '_back_center_DepthVis',
#         '_front_center_DisparityNormalized',
#         '_front_right_DisparityNormalized',
#         '_front_left_DisparityNormalized',
#         '_fpv_DisparityNormalized',
#         '_back_center_DisparityNormalized',
#         '_front_center_Segmentation',
#         '_front_right_Segmentation',
#         '_front_left_Segmentation',
#         '_fpv_Segmentation',
#         '_back_center_Segmentation',
#         '_front_center_SurfaceNormals',
#         '_front_right_SurfaceNormals',
#         '_front_left_SurfaceNormals',
#         '_fpv_SurfaceNormals',
#         '_back_center_SurfaceNormals',
#         '_front_center_Infrared',
#         '_front_right_Infrared',
#         '_front_left_Infrared',
#         '_fpv_Infrared',
#         '_back_center_Infrared',
#     ] 
# 
#     for j, view in enumerate(views):
#         with open('data/img_' + str(i) + view + '.png', 'wb') as f:
#             f.write(img[j].image_data_uint8 )
# # # # # # # # # # # # 