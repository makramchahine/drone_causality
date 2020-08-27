import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr

import sys 
import time 
import random 
import numpy as np

client = airsim.MultirotorClient() 
client.confirmConnection() 
client.enableApiControl(True) 

def rotate(v, quaternion): 
    u = np.array([quaternion.x_val, quaternion.y_val, quaternion.z_val])
    s = np.array([quaternion.w_val])

    return 2 * np.dot(u, v) * u + (s*s - np.dot(u,u)) * v + 2 * s * np.cross(u, v)

def getCentroid(face):
    return (face[0] + face[1] + face[2]) / 3

def distancePoint2Face(point, face):
    faceNormal = np.cross(face[0] - face[1], face[1] - face[2])
    faceNormal = faceNormal / np.linalg.norm(faceNormal)
    pass

# IMPORTANT! This needs to be the starting location of the drone in UE Coordinates
NED_OFFSET = np.array((-44187.0, -74102.0, 22867.0))
CENTIMETERS_TO_METERS = 0.01

# List of returned meshes are received via this function
meshes=client.simGetMeshPositionVertexBuffers()
faces = []
centroids = []
for m in meshes:

    # Convert the lists to numpy arrays
    vertices = np.array(m.vertices, dtype=np.float32)
    indices  = np.array(m.indices, dtype=np.uint32)
    numVertices = len(vertices) // 3
    numIndices  = len(indices)
    
    vertices = vertices.reshape((numVertices, 3))
    indices  = indices.reshape((numIndices // 3, 3))

    for i in indices:
        faces.append(np.array((
            CENTIMETERS_TO_METERS*(vertices[i[0]] - NED_OFFSET), 
            CENTIMETERS_TO_METERS*(vertices[i[1]] - NED_OFFSET), 
            CENTIMETERS_TO_METERS*(vertices[i[2]] - NED_OFFSET))))
        centroids.append(getCentroid(faces[-1]))

MIN_CLEARANCE = 1.0
def checkOccupied(point):
    for c in centroids:
        if np.dot(point - c, point - c) < MIN_CLEARANCE**2:
            return True
    return False

radius = 5.0
endpoint = Vector3r( radius*random.random(),  radius*random.random(), - radius*random.random())
speed    = 1 # m/s

# Mark the endpoint
client.simPlotPoints([endpoint], size = 10, is_persistent = True)

# Build voxel map
# Player start = (-44187.0, -74102.0, 22867.0)
# def convert2UECoordinates(point):
#     return meters2Centimeters * Vector3r(point.x, point.y, -point.z) + nedOffset

class voxelOccupancyMap:

    def __init__(self, diameter=20, cellWidth=1):
        sideLength = diameter / cellWidth
        self.data = [[[]]]

        for x in range(sideLength):
            for y in range (sideLength):
                for z in range(sideLength):
                    pointNED = Vector3r(x, y, z)


 
# # Find a collision-free path to the endpoint
# gridCellLength = 1
# grid = np.zeros((20,20,20))
# 
# def getDroneWorldPosition():
#     return client.simGetVehiclePose().position +
# 
# def checkIsOccupied(i, j, k):
#     dronePosition = getDroneWorldPosition()
#     cellPosition = dronePosition + (i, j, k)
# 
#     for vertex in meshes.vertices:
#         if abs(cellPosition - vertex) < gridCellLength / 2.0:
#             return True
# 
#     return False
# 
# def fillOccupied(g):
#     for i in len(g):
#         for j in len(g):
#             for k in len(g):
#                 g[i, j, k] = checkIsOccupied(i, j, k)
# 
# print(fillOccupied(grid))

# client.armDisarm(True)
# client.takeoffAsync().join()

# print("Taken off")

# images = []
#         airsim.ImageRequest("fpv",          airsim.ImageType.DepthPlanner, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("back_center",  airsim.ImageType.DepthPlanner, pixels_as_float = False, compress = True), 
#       
#         airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_right",  airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_left",   airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("fpv",          airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("back_center",  airsim.ImageType.DepthPerspective, pixels_as_float = False, compress = True),
#       
#         airsim.ImageRequest("front_center", airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_right",  airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_left",   airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("fpv",          airsim.ImageType.DepthVis, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("back_center",  airsim.ImageType.DepthVis, pixels_as_float = False, compress = True),
#       
#         airsim.ImageRequest("front_center", airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_right",  airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_left",   airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("fpv",          airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("back_center",  airsim.ImageType.DisparityNormalized, pixels_as_float = False, compress = True),
#        
#         airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_right",  airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_left",   airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("fpv",          airsim.ImageType.Segmentation, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("back_center",  airsim.ImageType.Segmentation, pixels_as_float = False, compress = True),
#       
#         airsim.ImageRequest("front_center", airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_right",  airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_left",   airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("fpv",          airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("back_center",  airsim.ImageType.SurfaceNormals, pixels_as_float = False, compress = True),
#       
#         airsim.ImageRequest("front_center", airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_right",  airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("front_left",   airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("fpv",          airsim.ImageType.Infrared, pixels_as_float = False, compress = True), 
#         airsim.ImageRequest("back_center",  airsim.ImageType.Infrared, pixels_as_float = False, compress = True),  
#     ])) 

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