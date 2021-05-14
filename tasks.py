## Copyright 2021 Charlie Vorbach
# Utilities for creating and executing drone tasks in airsim

import airsim
from airsim import Vector3r, Pose, Quaternionr, YawMode
import threading

from planning import *

# Operating Modes
class Task: 
    TARGET = 'target'
    FOLLOWING = 'following'
    MAZE = 'maze'
    HIKING = 'hiking'
    DEMO = 'demo'
    MULTITRACK = 'multitrack'

class FlightController:
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.occupancy_cache = VoxelOccupancyCache(config['voxel_size'], config['occupancy_cache_size'])


    def getTime(self):
        return 1e-9 * self.client.getMultirotorState().timestamp


    def getPose(self):
        pose        = self.client.simGetVehiclePose()
        position    = pose.position.to_numpy_array() - CAMERA_OFFSET
        orientation = pose.orientation.to_numpy_array()
        return position, orientation


    def isValidEndpoint(self, endpoint, requireVisible=False):
        if endpoint in self.occupancy_cache:
            return False

        position, orientation = self.getPose()
        if requireVisible and not isVisible(endpoint, position, R.from_quat(orientation), self.config):
            return False

        # TODO(cvorbach) Check there is a valid path

        return True


    def generateMazeTarget(self, radius=50, zLimit=[-30, -10], requireVisible=False):
        isValid = False
        attempts = 0
        while not isValid:
            endpoint = np.array([
                2 * radius * (random.random() - 0.5), 
                2 * radius * (random.random() - 0.5), 
                (zLimit[0] - zLimit[1]) * random.random() + zLimit[1]])

            isValid = self.isValidEndpoint(endpoint, requireVisible=requireVisible)

            attempts += 1
            if attempts > self.config['bogo_attempts']:
                print(attempts)
                raise RuntimeError('Exceeded bogo limit in Maze Target Generation')

        return endpoint


    def generateTarget(self, radius=10, zLimit=(-float('inf'), float('inf'))):
        isValid = False
        attempts = 0
        while not isValid:
            # TODO(cvorbach) smarter generation without creating points under terrain
            position, orientation = self.getPose()
            yawRotation = R.from_euler('xyz', [0, 0, R.from_quat(orientation).as_euler('xyz')[2]])

            endpoint = position + yawRotation.apply(self.occupancy_cache.point2Voxel(radius * normalize(np.array([random.random(), 0.1*random.random(), -random.random()]))))

            # Altitude limit
            endpoint[2] = min(max(endpoint[2], zLimit[0]), zLimit[1])

            endpoint = self.occupancy_cache.point2Voxel(endpoint)
            isValid = self.isValidEndpoint(endpoint, self.occupancy_cache)

            attempts += 1
            if attempts > self.config['bogo_attempts']:
                return None

        return np.array(endpoint)


    def turnTowardEndpoint(self, endpoint, timeout=0.01):
        position, _ = self.getPose()
        displacement = endpoint - position

        endpointYaw = np.arctan2(displacement[1], displacement[0]) * RADIANS_2_DEGREES
        self.client.rotateToYawAsync(endpointYaw, timeout_sec = timeout).join()
        print("Turned toward endpoint")


    def tryPlotting(self, lastPlotTime):
        if self.getTime() < self.config['plot_update_period'] + lastPlotTime:
            return lastPlotTime

        # occupancyMap.plotOccupancies()

        print("Replotted :)")
        return self.getTime()


    def updateOccupancies(self):
        lidarData = self.client.getLidarData()
        lidarPoints = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
        if len(lidarPoints) >=3:
            lidarPoints = np.reshape(lidarPoints, (lidarPoints.shape[0] // 3, 3))

            for p in lidarPoints:
                self.occupancy_cache.addPoint(p)

        # print("Lidar data added")


    def getNearestPoint(self, trajectory, position):
        closestDist = None
        for i in range(len(trajectory)):
            point = trajectory[i]
            dist  = distance(point, position)

            if closestDist is None or dist < closestDist:
                closestDist = dist
                closestIdx  = i

        return closestIdx


    def getProgress(self, trajectory, currentIdx, position):
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


    def pursuitVelocity(self, trajectory):
        '''
        This function is kinda of a mess, but basically it implements 
        carrot following of a lookahead.

        The three steps are:
        1. Find the point on the path nearest to the drone as start of carrot
        2. Find points in front of and behind the look ahead point by arc length
        3. Linearly interpolate to find the lookahead point and chase it.
        '''

        position, _ = self.getPose()
        startIdx = self.getNearestPoint(trajectory, position)
        progress = self.getProgress(trajectory, startIdx, position)

        arcLength   = -progress
        pointBehind = trajectory[0]
        for i in range(1, len(trajectory)):
            pointAhead = trajectory[i]

            arcLength += distance(pointAhead, pointBehind)
            if arcLength > self.config['lookahead_distance']:
                break

            pointBehind = pointAhead

        # if look ahead is past the end of the trajectory
        if np.array_equal(pointAhead, pointBehind): 
            lookAheadPoint = pointAhead

        else:
            behindWeight = (arcLength - self.config['lookahead_distance']) / distance(pointAhead, pointBehind)
            aheadWeight = 1.0 - behindWeight

            # sanity check
            if not (0 <= aheadWeight <= 1 or 0 <= behindWeight <= 1):
                raise Exception("Invalid Interpolation Weights")

            lookAheadPoint = aheadWeight * pointAhead + behindWeight * pointBehind

        # Compute velocity to pursue lookahead point
        pursuitVector = lookAheadPoint - position
        pursuitVector = self.config['drone_speed'] * normalize(pursuitVector)
        return pursuitVector


    def getLookAhead(self, path, t, position, lookAhead, endT=1, dt=1e-1):
        lookAheadPoint        = position  # TODO(cvorbach) increments always?
        lookAheadDisplacement = 0

        while np.linalg.norm(lookAheadDisplacement) < lookAhead:
            t += dt

            if t < endT:
                lookAheadDisplacement = lookAheadPoint - position
                lookAheadPoint = path(t)[:3]
            else:
                t = endT
                lookAheadPoint = path(t)[:3]
                break
                # TODO(cvorbach) unnecessary with extrapolation
            
        return t, lookAheadPoint

    class RunFailure(Exception):
        pass

    def animateTrajectory(self, marker, trajectory, t):
        markerPose = airsim.Pose()
        position   = trajectory(t)

        tangent  = normalize(trajectory.tangent(t))
        normal   = normalize(np.cross(tangent, (0,0,1))) # Note, we use world z-axis for up direction
        binormal = normalize(np.cross(tangent, normal))  # This fails if tangent == z-axis == (0,0,1)

        orientation = R.from_matrix(np.array([tangent, normal, binormal]).T)

        markerPose.position    = Vector3r(*position)
        markerPose.orientation = Quaternionr(*orientation.as_quat())

        self.client.simSetObjectPose(marker, markerPose)

    def followTrajectory(self, trajectory, lookAhead = 2, kp=1):
        raise NotImplementedError


    def followPath(self, path, lookAhead = 2, dt = 1e-4, marker=None, earlyStopDistance=None, planningWrapper=None, planningKnots=None, recordingEndpoint=None, model=None):
        startTime       = self.getTime()
        position, _     = self.getPose()
        t               = path.project(position) # find the new nearest path(t)
        lookAheadPoint  = path(t)
        reachedEnd      = False

        lastPlotTime    = self.getTime()
        markerPose      = airsim.Pose()

        planningThread  = None
        controlThread   = None

        print('started following path')

        if self.config['plot_debug']:
            self.client.simPlotPoints([Vector3r(*path(t)) for t in np.linspace(0, 1, 1000)], color_rgba = [0.0, 0.0, 1.0, 1.0], duration = 60)

        if self.config['record'] and self.config['task'] != Task.HIKING:
            self.client.startRecording()

        endpointDirections = []
        imagesBuffer       = np.zeros((1, self.config['seq_len'], *self.config['image_shape']))
        gpsBuffer          = np.zeros((1, self.config['seq_len'], 3))
        numBufferEntries   = 0

        # control loop
        lastVelocity = None
        alpha        = 1.0
        while not reachedEnd:
            position, orientation = self.getPose()
            self.updateOccupancies()

            # handle timeout and collision check
            if self.config['autoKill'] and startTime + self.config['timeout'] < self.getTime():
                raise RunFailure('Run timed out.')

            if self.config['autoKill'] and self.client.simGetCollisionInfo().has_collided:
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

                planningThread.join(timeout=self.config['control_update_period'])

            # place marker if passed
            if marker is not None:
                markerT, markerPosition = self.getLookAhead(path, t, position, lookAhead)

            #  tangent  = normalize(path.tangent(markerT))
            #  normal   = normalize(np.cross(tangent, (0, 0, 1)))

            #  inclinationAngle = np.pi + MAX_INCLINATION * (1 - np.abs(tangent[2])) # pi is b/c the drone is upside down at 0 inclination
            #  inclination = R.from_rotvec(inclinationAngle*normal)
            #  tangent     = inclination.apply([tangent[0], tangent[1], 0])

            #  binormal = normalize(np.cross(tangent, normal))

            #  markerOrientation = R.from_matrix(np.array([tangent, normal, binormal]).T)

            #  markerPose.orientation = Quaternionr(*markerOrientation.as_quat())
                markerPose.position    = Vector3r(*markerPosition)
                self.client.simSetObjectPose(marker, markerPose)

            # advance the pursuit point if needed
            # TODO(cvorbach) move to its own thread
            t, lookAheadPoint = self.getLookAhead(path, t, position, lookAhead)
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
                    velocity = self.config['drone_speed'] * normalize(lookAheadDisplacement)
                else:
                    velocity = self.config['drone_speed'] * normalize(lookAheadDisplacement)
                    velocity = alpha * velocity + (1-alpha)*lastVelocity
                lastVelocity = velocity

                # plot
                if self.config['plot_debug']:
                    lastPlotTime = self.tryPlotting(lastPlotTime)

                # record direction vector to endpoint if needed 
                if self.config['record'] and recordingEndpoint is not None:
                    endpointDirections.append((time.time(), *normalize(recordingEndpoint - position)))

                # start control thread
                if controlThread is not None:
                    controlThread.join()
                controlThread = self.client.moveByVelocityAsync(float(velocity[0]), float(velocity[1]), float(velocity[2]), self.config['control_update_period'], yaw_mode=YawMode(is_rate = False, yaw_or_rate = yawAngle))

            # If we are flying by a model
            else:
                # place marker if passed
                if marker is not None:
                    markerPose.position = Vector3r(*lookAheadPoint)
                    self.client.simSetObjectPose(marker, markerPose)

                # get and format an image
                image = None
                while image is None or len(image) == 1:
                    image = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
                    image = np.fromstring(image.image_data_uint8, dtype=np.uint8).astype(np.float32) / 255
                image = np.reshape(image, self.config['image_shape'])
                image = image[:, :, ::-1]                 # Required since order is BGR instead of RGB by default

                # add the image to the sliding window
                if numBufferEntries < self.config['seq_len']:
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
                velocity   = self.config['drone_speed'] * direction

                yawRotation = R.from_euler('xyz', [0, 0, R.from_quat(orientation).as_euler('xyz')[2]])

                # get yaw angle to the endpoint
                lookAheadDisplacement = lookAheadPoint - position
                yawAngle = np.arctan2(lookAheadDisplacement[1], lookAheadDisplacement[0]) * RADIANS_2_DEGREES

                endpointDisplacement = path(1.0) - position
                yawAngle = np.arctan2(endpointDisplacement[1], endpointDisplacement[0]) * RADIANS_2_DEGREES

                # check if we've reached the endpoint
                if np.linalg.norm(endpointDisplacement) < self.config['endpoint_tolerance']:
                    reachedEnd = True
                    break

                # start control thread
                if controlThread is not None:
                    controlThread.join()
                controlThread = self.client.moveByVelocityAsync(float(velocity[0]), float(velocity[1]), float(velocity[2]), self.config['control_update_period'], yaw_mode=YawMode(is_rate = False, yaw_or_rate = yawAngle))

        # hide the marker
        if marker is not None:
            markerPose.position = Vector3r(0,0,100)
            self.client.simSetObjectPose(marker, markerPose)

        if self.config['record'] and self.config['task'] != Task.HIKING:
            self.client.stopRecording()

        # Write out the direction vectors if needed
        if self.config['record'] and recordingEndpoint is not None:
            recordingDir = sorted([d for d in os.listdir(RECORDING_DIRECTORY) if RECORDING_NAME_REGEX.match(d)])[-1]
            with open(RECORDING_DIRECTORY + '/' + recordingDir + '/endpoint_directions.txt', 'w') as f:
                endpointFileWriter = csv.writer(f) 
                endpointFileWriter.writerows(endpointDirections)


    def moveToEndpoint(self, endpoint, recordEndpointDirection=False, model=None):
        self.updateOccupancies()

        position, _  = self.getPose()
        print('first planning')
        pathKnots    = findPath(position, endpoint, self.occupancy_cache, self.config['endpoint_tolerance'])
        print('finshed first planning')
        pathToEndpoint = Path(pathKnots.copy())

        def planningWrapper(knots):
            # run planning
            newKnots = findPath(position, endpoint, self.occupancy_cache, self.config['endpoint_tolerance'])

            # replace the old knots
            knots.clear()
            for k in newKnots:
                knots.append(k)
                
            # print('Finished Planning')

        if recordEndpointDirection:
            recordingEndpoint = endpoint
        else:
            recordingEndpoint = None

        self.followPath(pathToEndpoint, earlyStopDistance=self.config['endpoint_tolerance'], planningWrapper=planningWrapper, planningKnots=pathKnots, recordingEndpoint=recordingEndpoint, model=model)
        print('Reached Endpoint')


    def checkHalfSpace(self, testPoint, p):
        '''
        Checks that testPoint is in the half space defined by
        point p and normal vector (p - x) / ||p - x||
        where x is the current posistion of the drone
        '''
        
        x, _ = self.getPose()

        return (testPoint - x).dot(normalize(p - x)) > np.linalg.norm(p - x)


    def checkBand(self, k, blazes, blazeStart, zLimit):

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
                    voxel = self.occupancy_cache.point2Voxel(corner + i * tangent)
                    band.append(voxel)

                    isOccupied    = voxel in self.occupancy_cache
                    isSpacedOut   = not np.any([distance(np.array(b), voxel) < self.config['min_blaze_gap'] for b in blazes])
                    isInHalfSpace = self.checkHalfSpace(np.array(voxel), np.array(blazeStart))

                    if isOccupied and isSpacedOut and isInHalfSpace:
                        return voxel

        # self.client.simPlotPoints([Vector3r(*v) for v in band])
        return None


    def generateHikingBlazes(self, start, numBlazes = 2, zLimit=(-15, -1), maxSearchDepth=50):
        blazes = []
        blazeStart = self.occupancy_cache.point2Voxel(start)

        # lastPlotTime = tryPlotting(-float('inf'), occupancyMap)

        for _ in range(numBlazes):
            nextBlaze   = None
            
            k = 5
            while nextBlaze is None and k < maxSearchDepth: 
                nextBlaze = self.checkBand(k, blazes, blazeStart, zLimit)
                k += 1

            if k == maxSearchDepth:
                raise Exception('Could not find a tree to blaze')

            blazes.append(nextBlaze)
            blazeStart = blazes[-1]

        return blazes