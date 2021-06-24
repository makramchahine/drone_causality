# drone-flight  Copyright (C) 2020  Charles Vorbach
import setup_path
import airsim
from airsim import Vector3r, Pose, Quaternionr, YawMode

import sys 
import time 
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
import toml
from enum import Enum
import traceback

from scipy.spatial.transform import Rotation as R

from planning import *
from ml_models import *
from tasks import *

class Empty:
    def __repr__(self):
        return 'Empty()'

    def __str__(self):
        return self.__repr__()

# Parameters
parser = argparse.ArgumentParser(description='Fly the deepdrone agent in the Airsim simulator')
parser.add_argument('--mission',               type=str,   default='target', help='Task to attempt')
parser.add_argument('--task',                  type=str,   default=Empty(), help='Task to attempt')
parser.add_argument('--endpoint_tolerance',    type=float, default=Empty(),      help='The distance tolerance on reaching the endpoint marker')
parser.add_argument('--max_endpoint_radius',   type=float, default=Empty(),     help='The max distance of endpoints in the near planning task')
parser.add_argument('--min_blaze_gap',         type=float, default=Empty(),     help='The minimum distance between hiking task blazes')
parser.add_argument('--plot_update_period',    type=float, default=Empty(),      help='The time between updates of debug plotting information')
parser.add_argument('--control_update_period', type=float, default=Empty(),      help='Update frequency of the pure pursuit controller')
parser.add_argument('--drone_speed',           type=float, default=Empty(),      help='Drone flying drone_speed')
parser.add_argument('--voxel_size',            type=float, default=Empty(),      help='The size of voxels in the occupancy map cache')
parser.add_argument('--cache_size',            type=float, default=Empty(),   help='The number of entries in the local occupancy cache')
parser.add_argument('--lookahead_distance',    type=float, default=Empty(),     help='Pure pursuit lookahead distance')
parser.add_argument('--bogo_attempts',         type=int,   default=Empty(),     help='Number of attempts to make in generate and test algorithms')
parser.add_argument('--num_repetitions',       type=int,   default=Empty(),     help='Number of repetitions of the task to attempt')
parser.add_argument("--plot_debug", dest="plot_debug",     action="store_true")
parser.set_defaults(gps_signal=Empty())
parser.add_argument('--record',     dest='record',         action='store_true')
parser.set_defaults(record=Empty())
parser.add_argument('--autoKill',   dest='record',         action='store_true')
parser.set_defaults(autoKill=Empty())
parser.add_argument('--model_weights',         type=str,   default=Empty(), help='Model weights to load and fly with')
parser.add_argument('--seq_len',               type=int,   default=Empty())
parser.add_argument('--batch_size',            type=int,   default=Empty())
parser.add_argument('--rnn_size',              type=int,   default=Empty(), help='Select the size of RNN network you would like to train')
parser.add_argument('--timeout',               type=int,   default=Empty())
args = parser.parse_args()

class FlightConfig:
    def __init__(self, mission, args, missionDir='missions'):
        with open(f'{missionDir}/{mission}.toml') as fp:
            print(f'{missionDir}/{mission}.toml')
            tomlRep = toml.load(fp)

        if mission != tomlRep['mission']:
            raise ValueError('Passed mission doesn\'t match the toml representation')

        self.fileConfig = tomlRep
        self.cmlConfig  = vars(args)

        # print('cml:', self.cmlConfig)
        # print('file:', self.fileConfig)


    def __getitem__(self, item):
        try:
            if item in self.cmlConfig and not isinstance(self.cmlConfig[item], Empty):
                return self.cmlConfig[item]
            else:
                return self.fileConfig[item]
        except:
            raise ValueError(f'{item} was not passed on command line and configuration toml is missing field {item} (or toml is malformed).')

config = FlightConfig(args.mission, args)

# Start up
client = airsim.MultirotorClient() 
client.confirmConnection() 
client.enableApiControl(True) 

# Weather
client.simEnableWeather(True)
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, config['fog'])
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, config['rain'])

RECORDING_DIRECTORY    = 'C:/Users/MIT Driverless/Documents/AirSim'
RECORDING_NAME_REGEX   = re.compile(r'^[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+$')

#TODO(cvorbach) CAMERA_HORIZONTAL_OFFSET

# Setup the network
flightModel = None
if config['use_model']:
    from ml_models import *
    flightModel = initializeMLNetwork(config)

# ---------------ght--------------
# MAIN
# -----------------------------

# Takeoff
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-10, 1).join()
print("Taken off")

# Create flight controller
flight_control = FlightController(client, config)

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
for i in range(config['num_repetitions']):
    # TODO(cvorbach) figure out why this doesn't work?
    #print('S:', [startingPose.position.x_val, startingPose.position.y_val, startingPose.position.z_val])
    # client.simSetVehiclePose(            
    #     startingPose,
    #     True
    # )
    # time.sleep(5)
    
    try:
        if config['task'] == Task.OCCLUSION:
            marker = markers[0]

            newStart = flight_control.generateMazeTarget(radius=config['max_endpoint_radius'])
            flight_control.moveToEndpoint(newStart)

            # Random rotation
            client.rotateToYawAsync(random.random() * 2.0 * np.pi * RADIANS_2_DEGREES).join()

            # print('here 2')

            # Set up
            endpoint = flight_control.generateMazeTarget(radius=config['max_endpoint_radius'], zLimit=(-5, -15), visibility=Visibility.NOT_VISIBLE)
            if endpoint is None:
                continue

            # print('here 3')
            position, _ = flight_control.getPose()
            print(distance(endpoint, position))
                
            # place endpoint marker
            endpointPose = airsim.Pose()
            endpointPose.position = Vector3r(*endpoint)
            client.simSetObjectPose(marker, endpointPose)

            flight_control.turnTowardEndpoint(endpoint, timeout=10)

            flight_control.moveToEndpoint(endpoint, model=flightModel)

        if config['task'] == Task.INTERACTIVE:
            print('Dropping to prompt')
            sys.exit()
            
        if config['task'] == Task.MULTITRACK:
            marker1, marker2 = markers[:2]
            markers = [marker1, marker2]

            flight_control.updateOccupancies()

            start1 = flight_control.generateMazeTarget(radius=config['max_endpoint_radius'], zLimit=(-5, -15), visibility=Visibility.VISIBLE)
            start2 = flight_control.generateMazeTarget(radius=config['max_endpoint_radius'], zLimit=(-5, -15), visibility=Visibility.VISIBLE)

            end1 = flight_control.generateMazeTarget(radius=2*config['max_endpoint_radius'], zLimit=(-5, -15))
            end2 = flight_control.generateMazeTarget(radius=2*config['max_endpoint_radius'], zLimit=(-5, -15))

            pathVoxels1 = findPath(start1, end1, flight_control.occupancy_cache, config['endpoint_tolerance'])
            pathVoxels2 = findPath(start2, end2, flight_control.occupancy_cache, config['endpoint_tolerance'])

            trajectoryTime = np.linspace(0, config['target_move_time'], len(pathVoxels1))

            knots1 = np.array([[*pathVoxels1[i], t] for i, t in zip(range(len(pathVoxels1)), trajectoryTime)])
            knots2 = np.array([[*pathVoxels2[i], t] for i, t in zip(range(len(pathVoxels2)), trajectoryTime)])

            trajectory1 = Trajectory(knots1)
            trajectory2 = Trajectory(knots2)
            targetTrajectories = [trajectory1, trajectory2]

            position, _ = flight_control.getPose()
            center, radius = walzBoundingSphere([traj(0) for traj in targetTrajectories])
            yawAngle = np.arctan2((center - position)[1], (center - position)[0])
            orientation = R.from_euler('xyz', [0,0,yawAngle])
            for traj in targetTrajectories:
                if not isVisible(traj(0), position, orientation, config):
                    raise RuntimeError('Target starts already out-of-view!')

            for i in range(len(targetTrajectories)):
                flight_control.animateTrajectory(markers[i], targetTrajectories[i], 0)

            trackingTrajectory = Trajectory(findTrackingKnots(position, targetTrajectories, flight_control.occupancy_cache, config))

            print('Starting at ', time.time())
            startTime = time.time()
            while time.time() - startTime < trajectoryTime[-1]:
                position, _ = flight_control.getPose()
                t = time.time() - startTime
                print('t', t)

                for i in range(len(targetTrajectories)):
                    flight_control.animateTrajectory(markers[i], targetTrajectories[i], t)

                lookAheadPoint        = trackingTrajectory(t + 0.1)
                lookAheadDisplacement = lookAheadPoint - position

                speed = min(np.linalg.norm(lookAheadDisplacement), config['max_speed'])
                print('speed', speed)
                try:
                    velocity = speed*normalize(lookAheadDisplacement)
                except ZeroDivisionError:
                    velocity = np.zeros((3,1))
                print('velocity', velocity)


                center, radius = walzBoundingSphere([traj(t) for traj in targetTrajectories])
                yawAngle = np.arctan2((center - position)[1], (center - position)[0])

                client.moveByVelocityAsync(
                    float(velocity[0]), float(velocity[1]), float(velocity[2]), 
                    config['control_update_period'], 
                    # yaw_mode=YawMode(is_rate = False, yaw_or_rate = yawAngle)
                ).join()


        if config['task'] == Task.DEMO:
            loop = generateLoop()

            t = np.linspace(0, 1, 1000) 
            if config['plot_debug']:
                client.simPlotPoints([Vector3r(*loop(t_i)) for t_i in t], color_rgba = [0.0, 0.0, 1.0, 1.0], duration = 60)

            # Move camera
            rotation = R.from_rotvec(3*np.pi/2 * np.array([0,0,1]))  
            camera_pose = airsim.Pose(Vector3r(0,3,0), Quaternionr(*rotation.as_quat())) 
            client.simSetCameraPose('0', camera_pose)

            flight_control.moveToEndpoint(loop(0))

            # if config['record']:
            #     client.startRecording()

            flight_control.followPath(loop)

            # if config['record']:
            #     client.stopRecording()

        if config['task'] == Task.TARGET:
            marker = markers[0]

            # print('here 1')

            # Random rotation
            client.rotateToYawAsync(random.random() * 2.0 * np.pi * RADIANS_2_DEGREES).join()

            # print('here 2')

            # Set up
            endpoint = flight_control.generateMazeTarget(radius=config['max_endpoint_radius'], zLimit=(-5, -15))
            if endpoint is None:
                continue

            # print('here 3')
            position, _ = flight_control.getPose()
            print(distance(endpoint, position))
                
            # place endpoint marker
            endpointPose = airsim.Pose()
            endpointPose.position = Vector3r(*endpoint)
            client.simSetObjectPose(marker, endpointPose)

            flight_control.turnTowardEndpoint(endpoint, timeout=10)

            flight_control.moveToEndpoint(endpoint, model=flightModel)

        if config['task'] == Task.FOLLOWING:

            marker = markers[0]
            flight_control.updateOccupancies()
            print('updated occupancies')

            position, _ = flight_control.getPose()

            # TODO(cvorbach) Online path construction with collision checking along each spline length
            walk = randomWalk(position, stepSize=1.5, occupancyMap=flight_control.occupancy_cache)
            path = Path(walk)
            # path = ExtendablePath(walk)

            # t = np.linspace(0, 1, 1000) 
            # client.simPlotPoints([Vector3r(*path(t_i)) for t_i in t], color_rgba = [0.0, 0.0, 1.0, 1.0], duration = 60)
            # sys.exit()

            print('got path')

            flight_control.followPath(path, marker=marker, model=flightModel, earlyStopDistance=config['endpoint_tolerance'])
            print('reached path end')

        if config['task'] == Task.HIKING:
            flight_control.updateOccupancies()
            print('updated occupancies')

            newStart = flight_control.generateMazeTarget(radius=config['max_endpoint_radius'])
            print(newStart)
            flight_control.moveToEndpoint(newStart)

            print('move to z-level')
            zLimit = (-10, -5)
            client.moveToZAsync((zLimit[0] + zLimit[1])/2, 1).join()
            print('reached z-level')

            print('getting blazes')
            position, _ = flight_control.getPose()
            hikingBlazes = flight_control.generateHikingBlazes(position, zLimit=zLimit)
            flight_control.turnTowardEndpoint(hikingBlazes[0], timeout=10)
            print('Got blazes')

            if len(hikingBlazes) > len(markers):
                raise Exception('Not enough markers for each blaze to get one')

            # place makers
            markerPose = airsim.Pose()
            for i, blaze in enumerate(hikingBlazes):
                print('placed blaze', i)
                markerPose.position = Vector3r(*blaze)
                client.simSetObjectPose(markers[i], markerPose)

            if config['record']:
                client.startRecording()

            for blaze in hikingBlazes:
                print('moving to blaze')
                flight_control.moveToEndpoint(blaze, model=flightModel)
                # TODO(cvorbach) rotate 360
                print('moved to blaze')

            if config['record']:
                client.stopRecording()

        if config['task'] == Task.MAZE:
            # Set up
            marker = markers[0]

            endpoint = flight_control.generateMazeTarget(radius=config['max_endpoint_radius'], zLimit=(-5, -15))
            if endpoint is None:
                continue
                
            # place endpoint marker
            endpointPose = airsim.Pose()
            endpointPose.position = Vector3r(*endpoint)
            client.simSetObjectPose(marker, endpointPose)

            flight_control.turnTowardEndpoint(endpoint, timeout=10)

            flight_control.moveToEndpoint(endpoint, recordEndpointDirection=True, model=flightModel)

        successes += 1
        print(f'Run {i} succeeded: {successes}/{i+1} success rate.')
    #except RunFailure:
    except Exception as e:
        print(f'Run {i} failed:    {successes}/{i+1} success rate.')
        print(e)
        traceback.print_exc()


print('Finished Data Runs')