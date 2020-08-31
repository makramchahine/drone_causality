import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr

import sys 
import time 
import random 
import numpy as np
import pprint
import heapq

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
ENDPOINT_TOLERANCE = 1.5
ENDPOINT_RADIUS = 25
PLOT_PERIOD = 4.0
CONTROL_PERIOD = 0.5
SPEED = 1.0 # m/s

# Utilities
class SparseVoxelOccupancyMap:
    def __init__(self, voxelSize):
        self.voxelSize = voxelSize
        self.occupiedVoxels = set()

    def addPoint(self, point):
        voxel = self.point2Voxel(point)
        self.occupiedVoxels.add(voxel)

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
        client.simPlotPoints(occupiedPoints, color_rgba = [0.0, 0.0, 1.0, 1.0], duration=PLOT_PERIOD-0.5) 

class Heap:
    def __init__(self, map=min):
        self.data = []
        self.map  = map

    def parent(self, i):
        if i == 0:
            return None
        else:
            return (i - 1) // 2

    def left(self, i):
        ret = 2*i + 1
        if ret >= len(self.data):
            return None
        else:
            return ret
    
    def right(self, i):
        ret = 2*i + 2
        if ret >= len(self.data):
            return None
        else:
            return ret
    
    def heapify(self, i):
        l = self.left(i)
        r = self.right(i)
        top = i

        if l is not None and self.map(self.data[l]) > self.map(self.data[i]):
            top = l

        if  r is not None and self.map(self.data[top]) < self.map(self.data[r]):
            top = r
        
        if top != i:
            swp = self.data[i]
            self.data[i] = self.data[top]
            self.data[top] = swp
            self.heapify(top)

    def pop(self):
        # Replace top element with bottom leaf
        top = self.data[0]
        self.data[0] = self.data[-1]
        del self.data[-1]

        # Fix the heap
        self.heapify(0)
        return top

    def add(self, key):
        self.data.append(-float('inf'))
        self.increaseKey(len(self.data), key)

    def increaseKey(self, i, key):
        if key < A[i]:
            raise("New key " + str(key) + " is not larger than previous key " + str(self.data[i]) + " at location i=" + str(i))

        self.data[i] = key
        while self.parent(i) is not None and self.data[self.parent(i)] < self.data[i]:
            swp = self.data[i]
            self.data[i] = self.data[self.parent(i)]
            self.data[self.parent(i)] = swp
            i = self.parent(i)

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

    openSet = Heap(map=fScore.get)
    

    while openSet:
        # TODO(cvorbach) use a heap here for big performace gains
        current =  

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
                fScore[neighbor]   = gScore.get(neighbor, float('inf')) + h(neighbor, map, endpoint)

                heapq.heappush(openSet, (fScore[neighbor], neighbor))
        
    raise ValueError("Couldn't find a path")

# Takeoff
# client.armDisarm(True)
# client.takeoffAsync().join()

endpoint = ENDPOINT_RADIUS * np.array([random.random(), random.random(), -random.random()])
client.simPlotPoints([Vector3r(endpoint[0], endpoint[1], endpoint[2])], is_persistent = True) 

map = SparseVoxelOccupancyMap(1)
steps = 0
controlThread = None

position = np.zeros((3,))
while np.linalg.norm(endpoint - position) > ENDPOINT_TOLERANCE:
    pose = client.simGetVehiclePose()
    position = pose.position.to_numpy_array()

    lidarData = client.getLidarData()
    lidarPoints = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
    if len(lidarPoints) >=3:
        lidarPoints = np.reshape(lidarPoints, (lidarPoints.shape[0] // 3, 3))

        for p in lidarPoints:
            map.addPoint(p)

    trajectory = findPath(position, endpoint, map)
    trajectoryLine = [Vector3r(float(trajectory[i][0]), float(trajectory[i][1]), float(trajectory[i][2])) for i in range(len(trajectory))]

    if (steps % int(PLOT_PERIOD / CONTROL_PERIOD) == 0):
        print("Replotted :)")
        client.simPlotPoints(trajectoryLine, color_rgba = [0.0, 1.0, 0.0, 1.0], duration=PLOT_PERIOD-0.5) 
        map.plotOccupancies()

    nextStep = trajectory[1] - position
    nextStep = SPEED / np.linalg.norm(nextStep) * nextStep

    if controlThread is not None:
        controlThread.join()
    controlThread = client.moveByVelocityAsync(float(nextStep[0]), float(nextStep[1]), float(nextStep[2]), CONTROL_PERIOD)
    
    steps += 1

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