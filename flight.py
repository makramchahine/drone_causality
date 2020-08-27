import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr

import sys 
import time 
import random 
import numpy as np
import pprint

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

state = client.getMultirotorState()
s = pprint.pformat(state)

radius = 25
toleranace = 1.0
endpoint = radius * np.array([random.random(), random.random(), -random.random()])
client.simPlotPoints([Vector3r(endpoint[0], endpoint[1], endpoint[2])], is_persistent = True) 
 
class VoxelOccupancyMap:
    def __init__(self, radius, cellSize):
        self.cellSize = cellSize
        self.radius   = radius

        sideLength = int(2*radius / cellSize + 0.5)
        self.sideLength = sideLength

        self.data = np.full((sideLength, sideLength, sideLength), False)
        self.occupiedVoxels = set()
    
    def addPoint(self, point):
        i, j, k = self.indexOf(point)
        self.data[i, j, k] = True
        self.occupiedVoxels.add((i,j,k))

    def isOccupied(self, point):
        i, j, k = self.indexOf(point)
        return self.data[i, j, k]

    def indexOf(self, point):
        i = int((self.radius + point[0]) / self.cellSize)
        j = int((self.radius + point[1]) / self.cellSize)
        k = int((self.radius + point[2]) / self.cellSize)

        if any([v < 0 for v in (i, j, k)]) or any([v >= self.sideLength for v in (i, j, k)]):
            raise ValueError("Passed point " + str(point) + " is outside voxel map bounds.")

        return i, j, k

    def getAdjacent(self, idx):
        neighbors = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == 0 and j == 0 and k == 0:
                        continue

                    newIdx = (idx[0] + i, idx[1] + j, idx[2] + k)

                    if newIdx[0] < 0 or newIdx[1] < 0 or newIdx[2] < 0:
                        continue

                    if newIdx[0] >= self.sideLength or newIdx[1] >= self.sideLength or newIdx[2] >= self.sideLength:
                        continue
                    
                    if self.data[newIdx[0], newIdx[1], newIdx[2]]:
                        continue
                    
                    neighbors.append(newIdx)

        return neighbors


    def idx2Point(self, idx):
        return np.array((self.cellSize * idx[0] - self.radius, self.cellSize * idx[1] - self.radius, self.cellSize * idx[2] - self.radius))

    def plotOccupancies(self):
        occupiedPoints = [self.idx2Point(idx) for idx in self.occupiedVoxels]
        occupiedPoints = [Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in occupiedPoints]
        client.simPlotPoints(occupiedPoints, color_rgba = [0.0, 0.0, 1.0, 1.0], duration=1.5) 

    def h(self, idx):
        return np.linalg.norm(endpoint - self.idx2Point(idx))

    def d(self, idx1, idx2):
        return np.linalg.norm(self.idx2Point(idx2) - self.idx2Point(idx1))



# A* Path finding 
def findPath(startpoint, endpoint, map):
    start = map.indexOf(startpoint)
    end   = map.indexOf(endpoint)

    openSet = {start}
    cameFrom = dict()

    gScore = dict()
    gScore[start] = 0

    fScore = dict()
    fScore[start] = map.h(start)

    while openSet:
        current = min(openSet, key = lambda n: fScore.get(n, float("inf")))

        if current == end:
            path = [current]
            while path[-1] != start:
                current = cameFrom[current]
                path.append(current)
            
            return [map.idx2Point(idx) for idx in reversed(path)]

        openSet.remove(current)

        for neighbor in map.getAdjacent(current):
            tentativeGScore = gScore.get(current, float("inf")) + map.d(current, neighbor)

            if tentativeGScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = current
                gScore[neighbor]   = tentativeGScore
                fScore[neighbor]   = gScore.get(neighbor, float('inf')) + map.h(neighbor)

                if neighbor not in openSet:
                    openSet.add(neighbor)
    
    raise ValueError("Couldn't find a path")

map = VoxelOccupancyMap(100, 1)

client.armDisarm(True)
client.takeoffAsync()

speed = 2.0 # m/s

steps = 0

controlPeriod = 0.5
plotPeriod = 4.0

position = np.zeros((3,))
while np.linalg.norm(endpoint - position) > toleranace:
    pose = client.simGetVehiclePose()
    position = pose.position.to_numpy_array()

    lidarData = client.getLidarData()
    lidarPoints = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
    if len(lidarPoints) >=3:
        lidarPoints = np.reshape(lidarPoints, (lidarPoints.shape[0] // 3, 3))

        for p in lidarPoints:
            map.addPoint(p + position)

    trajectory = findPath(position, endpoint, map)
    trajectoryLine = [Vector3r(float(trajectory[i][0]), float(trajectory[i][1]), float(trajectory[i][2])) for i in range(len(trajectory))]

    if (steps % int(plotPeriod / controlPeriod + 0.5)):
        client.simPlotPoints(trajectoryLine, color_rgba = [0.0, 1.0, 0.0, 1.0], duration=1.5) 
        map.plotOccupancies()

    if len(trajectory) > 2:
        nextStep = (float(trajectory[2][0]), float(trajectory[2][1]), float(trajectory[1][2]))
    else:
        nextStep = endpoint

    print("p: " + str(position))
    print("n: " + str(nextStep))

    client.moveToPositionAsync(*nextStep, speed)
    
    time.sleep(controlPeriod)

    steps += 1

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