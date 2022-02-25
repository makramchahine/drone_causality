# flight-control  Copyright (C) 2021  Charles Vorbach
import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr, YawMode

from ml_models import initializeMLNetwork
from flight_utils import getLookAhead, VoxelOccupancyCache

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
import toml
from enum import Enum

from scipy.spatial.transform import Rotation as R

# Operating Modes
class TaskType: 
    TARGET = 'target'
    FOLLOWING = 'following'
    MAZE = 'maze'
    HIKING = 'hiking'

class TaskInfo:
    def __init__(self, taskType ):
        self.taskType = taskType

class TargetTaskInfo(TaskInfo):
    def __init__(self, taskType, initialPose, endpoint, endpointPose):
        super().__init__(taskType)
        self.initialPose = initialPose
        self.endpoint    = endpoint
        self.endpointPose = endpointPose

class FlightConfig:
    def __init__(self, mission, tomlRep):
        if mission != tomlRep['mission']:
            raise ValueError('Passed mission doesn\'t match the toml representation')
        self.configData = tomlRep

    def __getitem__(self, item):
        try:
            return self.configData[item]
        except:
            raise ValueError(f'Configuration toml is missing field {item} or otherwise malformed.')


# Flight Controller
class FlightController:
    def __init__(self, mission='target'):
        self.client = airsim.MultirotorClient()
        self.config = self.loadMission(mission)

        print(self.config['useModel'])

        if self.config['useModel']:
            self.mlModel = initializeMLNetwork(self.config)

        self.occupancyMap = VoxelOccupancyCache(self.config['voxel_size'], self.config['cache_size'], self.client)

        self.arm()

    def generateTask(self, taskType):
        if taskType == TaskType.TARGET:
            return self.generateTargetTask()
        else:
            raise ValueError(f'Unknown task type {taskType}')

    def generateTargetTask(self):

        # Random rotation
        position, orientation = self.getPose()
        

        # Get endpoint


        endpointPose = airsim.Pose()
        endpointPose.position = Vector3r(*endpoint)

    def getTarget(self, origin, radius, orientation):
        haveTarget = False
        attempts   = 0
        while not haveTarget:
            step = radius*(np.random.rand((3,)) - 0.5) 
            target = origin + step

            unoccupied = target not in self.occupancyMap
            visible    = isVisible(origin, target, orientation)
            haveTarget = unoccupied and visible

            attempts += 1
            if attempts > self.client.bogo_attempts:
                raise RuntimeError('Exceeded bogo limit while getting a target')



    def executeTask(self, taskInfo):
        if taskInfo.taskType == TaskType.TARGET:
            self.executeTargetTask(taskInfo)
        else:
            raise ValueError(f'Unknown task type {taskInfo.taskType}')

    def executeTargetTask(self, taskInfo):
        marker = self.getMarkers(hide=True)[0]

        self.client.simSetVehiclePose(taskInfo.initialPose)
        self.client.simSetObjectPose(marker, taskInfo.endpointPose)

        self.flyToPoint(taskInfo.endpoint)

    def flyToPoint(self, point):
        self.updateOccupancies()
        position, _  = getPose()

        # Plan initial path
        pathKnots      = findPath(position, endpoint, occupancyMap)
        pathToPoint = Path(pathKnots.copy())

        # Callback to update the path in planning thread
        def planningWrapper(knots):
            newKnots = findPath(position, endpoint, occupancyMap)

            # Return the new knots to the main thread
            # TODO(cvorbach) Technically, I don't think this threadsafe
            knots.clear()
            for k in newKnots:
                knots.append(k)

        # Call the main flight controller
        flyPath(
            pathToPoint, 
            earlyStopDistance   = self.config['earlyStopDistance'],
            planningWrapper     = planningWrapper,
            planningKnotsBuffer = pathKnots, 
            mlModel             = self.mlModel,
            lookAtPoint         = point
        )

    def joinPlanningThread(self, path, planningThread, planningWrapper, planningKnotsBuffer):
        # Start planning thread if it doesn't exist or finished completed
        if planningThread is None or not planningThread.is_alive():

            # update the spline path
            if np.any(planningKnotsBuffer != path.knotPoints):
                path.fit(planningKnotsBuffer.copy())
                t = path.project(position) # find the new nearest path(t)

            # restart planning
            planningThread = threading.Thread(target=planningWrapper, args=(planningKnots,))
            planningThread.start()

        planningThread.join(timeout=self.config['controllerUpdatePeriod'])

        return planningThread, t

    def flyPath(self, path, earlyStopDistance=0, planningWrapper=None, planningKnotsBuffer=None, mlModel=None, lookAtPoint=None):
        '''
        Workhorse function which contains the control loop for moving the drone along a parameterized path.
        '''
        position, _       = getPose()
        t                 = path.project(position) # find the path(t) nearest the starting position
        t, lookAheadPoint = getLookAhead(path, t, position, self.config['lookAheadDistance'])
        finishedPath      = False

        planningThread    = None
        controlThread     = None

        lastPlotTime      = getTime()

        if self.config['plotDebug']:
            client.simPlotPoints([Vector3r(*path(t)) for t in np.linspace(0, 1, 1000)], color_rgba = [0.0, 0.0, 1.0, 1.0], duration = 60)

        imagesBuffer    = np.zeros((1, self.config['sequenceLength'], *IMAGE_SHAPE))
        imagesBufferIdx = 0

        while not finishedPath:
            position, orientation = self.getPose()
            self.updateOccupancies()
                
            # Switch to plannning between control updates
            if planningWrapper is not None:
                planningThread, t = self.joinPlanningThread(path, planningThread, planningWrapper, planningKnotsBuffer)

            # TODO(cvorbach) Move marker                

            # Get the pure pursuit point
            t, lookAheadPoint = getLookAhead(path, t, position, self.config['lookAheadDistance'])
            lookAheadDisplacement = lookAheadPoint - position

            # Compute the drone's next yaw angle
            if lookAtEndpoint:
                # get yaw angle and pursuit velocity to the lookAtPoint
                endpointDisplacement = path(1.0) - position
                yawAngle = np.arctan2(endpointDisplacement[1], endpointDisplacement[0])
            else:
                # just look at the pursuit point
                yawAngle = np.arctan2(lookAheadDisplacement[1], lookAheadDisplacement[0])
            
            yawAngle = np.rad2deg(yawAngle)

            # Check if we've finished
            if distance(path(1.0), position) < np.max((earlyStopDistance, self.config['endpointTolerance'])):
                finishedPath = True
                break

            # Get a velocity vector from pure pursuit or the ml model
            if mlModel is None:
                velocity = self.config['droneSpeed'] * normalize(lookAheadDisplacement)

                if self.config['plotDebug']:
                    lastPlotTime = self.tryPlottingDebug()

            else:
                # get and format an image
                # TODO(cvorbach) Why is there a while loop?
                image = None
                while image is None or len(image) == 1:
                    image = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
                    image = np.fromstring(image.image_data_uint8, dtype=np.uint8).astype(np.float32) / 255
                image = np.reshape(image, IMAGE_SHAPE)
                image = image[:, :, ::-1]                 # Required since order is BGR instead of RGB by default

                # add the image to the sliding window
                if imagesBufferIdx < self.config['sequenceLength']:
                    imagesBuffer[0, imagesBufferIdx] = image
                    imagesBufferIdx += 1
                else:
                    imagesBuffer[0]     = np.roll(imagesBuffer[0], -1, axis=0)
                    imagesBuffer[0][-1] = image
                
                # compute a velocity vector from the model
                prediction = model.predict(imagesBuffer)[0][numBufferEntries-1]
                direction  = normalize(prediction)
                direction  = R.from_quat(orientation).apply(direction) # Transform from the drone camera's reference frame to static coordinates
                velocity   = self.config['droneSpeed'] * direction

            # run control update
            if controlThread is not None:
                controlThread.join()
            controlThread = client.moveByVelocityAsync(
                float(velocity[0]),
                float(velocity[1]),
                float(velocity[2]),
                self.config['controlUpdatePeriod'],
                yaw_mode=YawMode(is_rate = False, yaw_or_rate = yawAngle)
            )

            # TODO(cvorbach) Run failure check




                
                
                






        









        
            


    def getMarkers(self, hide=False):
        markers = self.client.simListSceneObjects('Red_Cube.*') 
        if len(markers) < 1:
            raise Exception('Didn\'t find any endpoint markers. Check there is a Red_Cube is in the scene')

        if hide: # move the markers out of view
            markerPose = airsim.Pose()
            markerPose.position = Vector3r(0, 0, 100)
            for marker in markers:
                self.client.simSetObjectPose(marker, markerPose)

        return markers
        

    @classmethod
    def loadMission(cls, mission, missionDir='missions'):
        with open(f'{missionDir}/{mission}.toml') as fp:
            print(f'{missionDir}/{mission}.toml')
            tomlRep = toml.load(fp)

        return FlightConfig(mission, tomlRep)

    def arm(self):
        self.client.confirmConnection() 
        self.client.enableApiControl(True) 

    def weather(self, fog=0, rain=0):
        self.client.simEnableWeather(True)
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog)
        self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain)


    