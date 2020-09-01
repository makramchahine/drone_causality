import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr

import sys 
import time 
import random 
import numpy as np
import pprint
import heapq

# from scipy.interpolate import UnivariateSpline

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
ENDPOINT_TOLERANCE = 2.5
ENDPOINT_RADIUS = 15
PLOT_PERIOD = 5.0
CONTROL_PERIOD = 0.5
SPEED = 0.5 # m/s
# BOUNDING_RADIUS = 2.5
VOXEL_SIZE = 3.0

# Utilities
class SparseVoxelOccupancyMap:
    def __init__(self, voxelSize):
        self.voxelSize = voxelSize
        self.occupiedVoxels = set()

    def addPoint(self, point):
        voxel = self.point2Voxel(point)
        self.occupiedVoxels.add(voxel)
        # voxelRadius = int(BOUNDING_RADIUS // self.voxelSize)

        # for i in range(-voxelRadius - 1, voxelRadius):
        #     for j in range(-voxelRadius - 1, voxelRadius):
        #         for k in range(-voxelRadius - 1, voxelRadius):
        #             neighboringVoxel = tuple(self.voxelSize * np.array([i, j, k])  + voxel)
        #             self.occupiedVoxels.add(neighboringVoxel)
        
    def isOccupied(self, point):
        voxel = self.point2Voxel(point)
        return voxel in self.occupiedVoxels
    
    def point2Voxel(self, point):
        return tuple(self.voxelSize * (v // self.voxelSize) for v in point)

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
        client.simPlotPoints(occupiedPoints, color_rgba = [0.0, 0.0, 1.0, 1.0], duration=PLOT_PERIOD-1.0) 

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
        # TODO(cvorbach) use a heap here for big performace gains
        current = heapq.heappop(openSet)[1]

        if current == end:
            path = [current]
            while path[-1] != start:
                current = cameFrom[current]
                path.append(current)
            
            return list(reversed(path))

        for neighbor in map.getNeighbors(current):
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

# Takeoff
client.armDisarm(True)
client.takeoffAsync().join()
print("Taken off")

map = SparseVoxelOccupancyMap(VOXEL_SIZE)

for i in range(100):
    controlThread = None

    endpoint = map.point2Voxel(ENDPOINT_RADIUS * np.array([random.random(), random.random(), -random.random()]))
    client.simPlotPoints([Vector3r(endpoint[0], endpoint[1], endpoint[2])], is_persistent = True) 

    pose = client.simGetVehiclePose()
    position = pose.position.to_numpy_array()
    displacement = endpoint - position

    endpointYaw = np.arctan2(displacement[1], displacement[0]) * 180 / 3.14
    client.rotateToYawAsync(endpointYaw).join()
    print("Pointed to endpoint")

    lastPlotTime = 0

    # client.startRecording()
    while np.linalg.norm(endpoint - position) > ENDPOINT_TOLERANCE:
        time = 1e-9 * client.getMultirotorState().timestamp

        pose = client.simGetVehiclePose()
        position = pose.position.to_numpy_array()
        displacement = endpoint - position

        if map.isOccupied(endpoint):
            break

        if map.isOccupied(position):
            raise ValueError("Drone in occupied position")

        endpointYaw = np.arctan2(displacement[0], displacement[1])

        lidarData = client.getLidarData()
        lidarPoints = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
        if len(lidarPoints) >=3:
            lidarPoints = np.reshape(lidarPoints, (lidarPoints.shape[0] // 3, 3))

            for p in lidarPoints:
                map.addPoint(p)

        print("Lidar data added")

        trajectory = findPath(position, endpoint, map)
        trajectoryLine = [Vector3r(float(trajectory[i][0]), float(trajectory[i][1]), float(trajectory[i][2])) for i in range(len(trajectory))]

        if time >= PLOT_PERIOD + lastPlotTime:
            lastPlotTime = time
            client.simPlotPoints(trajectoryLine, color_rgba = [0.0, 1.0, 0.0, 1.0], duration=PLOT_PERIOD-1.0) 
            map.plotOccupancies()
            print("Replotted :)")

        nextStep = trajectory[1] - position
        nextStep = SPEED / np.linalg.norm(nextStep) * nextStep

        if controlThread is not None:
            controlThread.join()

        endpointYaw = np.arctan2(displacement[1], displacement[0]) * 180 / 3.14
        client.rotateToYawAsync(endpointYaw, timeout_sec = 0.1).join()

        controlThread = client.moveByVelocityAsync(float(nextStep[0]), float(nextStep[1]), float(nextStep[2]), CONTROL_PERIOD)
        print("Moving")

        print("Turning")

    # client.stopRecording()

    if controlThread is not None:
        controlThread.join()
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