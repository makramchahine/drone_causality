## Copyright 2021 Charlie Vorbach
## Lots of planning utilities

import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import heapq
import random
import miniball
import math
import sys

# TODO(cvorbach) convert these to argments / toml configuration
CAMERA_FOV           = np.pi / 8
RADIANS_2_DEGREES    = 180 / np.pi
CAMERA_OFFSET        = np.array([0.5, 0, -0.5])
ENDPOINT_OFFSET      = np.array([0, -0.03, 0.025])
MAX_INCLINATION      = 0.3
DRONE_START          = np.array([-32295.757812, 2246.772705, 1894.547119])
WORLD_2_UNREAL_SCALE = 100

def normalize(vector):
    '''
    return x / ||x|| unit vector
    '''
    if np.linalg.norm(vector) == 0:
        raise ZeroDivisionError()
    return vector / np.linalg.norm(vector)


def distance(p1, p2):
    '''
    return ||p1 - p2||
    '''
    return np.linalg.norm(p2 - p1)


class ExtrapolationError(Exception):
    pass


class CatmullRomSegment:
    '''
    Single segment of a Catmull Rom Spline
    TODO(cvorbach) unimplemented
    '''
    def __init__(self, p, alpha=0.5):
        raise NotImplementedError
        if len(p) != 4:
            raise ValueError('Catmull Rom Segment requires 4 points')

        self.p = p

        t0 = 0
        t1 = t0 + distance(p[0], p[1])**alpha
        t2 = t1 + distance(p[1], p[2])**alpha
        t3 = t2 + distance(p[2], p[3])**alpha

        self.t = [t0, t1, t2, t3]

        self.coeff = self.getCoeff()
        
        # print('t:', t)
        # print('c:', self.coeff)

    def __call__(self, t):
        if t < 0 or t > 1:
            raise ValueError('Catmull Rom Segment cannot extrapolate t:', t) 

        # print('t', t)

        a, b, c, d = self.coeff
        value = a*t**3 + b*t**2 + c*t + d

        return value

    def ddt(self, t):
        if t < 0 or t > 1:
            raise ValueError('Catmull Rom Segment cannot extrapolate') 

        a, b, c, d = self.coeff
        derivative = 3*a*t**2 + 2*b*t + c
        return derivative

    def getCoeff(self):
        t0, t1, t2, t3 = self.t
        p0, p1, p2, p3 = self.p

        m1 = (t2-t1) * ( (p1-p0)/(t1-t0) - (p2-p0)/(t2-t0) + (p2-p1)/(t2-t1) )
        m2 = (t2-t1) * ( (p2-p1)/(t2-t1) - (p3-p1)/(t3-t1) + (p3-p2)/(t3-t2) )

        # print('t', self.t)

        a =  2*self.p[1] - 2*self.p[2] +  m1  + m2
        b = -3*self.p[1] + 3*self.p[2] - 2*m1 - m2
        c = m1
        d = self.p[1]

        return a, b, c, d

class CatmullRomSpline:
    '''
    Catmull Rom Spline implementation
    This class should allow a spline we can continuiously extend with new (x,y) points.
    TODO(cvorbach) Unimplemented
    '''
    def __init__(self, p):
        raise NotImplementedError
        if len(p) < 4:
            raise ValueError('Catmull Rom Spline requires at least 4 points')

        self.segments = []

        # Create the initial segment
        self.p = list(p[:4])
        self.extendSegment(p[:4])

        # Create additional segments
        self.extend(p[4:])

    def extend(self, newPoints):
        if len(newPoints) == 0: # If nothing to add
            return 

        # Extend each new segment from the previous last points
        for i, point in enumerate(newPoints):
            self.extendSegment([*self.p[-3:], point])
            self.p.append(point)

    def extendSegment(self, segmentPoints):
        if len(segmentPoints) > 4:
            raise ValueError('Too many points to extend Catmull Rom Spline')

        if len(segmentPoints) < 4:
            raise ValueError('Not enough points to extend Catmull Rom Spline')

        self.segments.append(CatmullRomSegment(segmentPoints))

    def __call__(self, t):
        '''
        Returns the value of the spline at s in [0, len(segments)+1]
        where s is the spline parameterization
        '''
        if t < 0 or t > 1:
            raise ExtrapolationError(f"Catmull Rom Spline cannot extrapolate")

        s = t*len(self)

        # Index of segment to use
        idx = min(int(s), len(self.segments)-1)
        ds  = s - idx

        value = self.segments[idx](ds)
        return value

    def ddt(self, t):
        '''
        Returns the value of the spline at s in [0, len(segments)+1]
        where s is the spline parameter
        '''
        if t < 0 or t > 1:
            raise ExtrapolationError(f"Catmull Rom Spline cannot extrapolate")

        s = t*len(self)

        # Index of segment to use
        idx = min(int(s), len(self.segments)-1)
        ds  = s - idx

        derivative = self.segments[idx].ddt(ds)

    def end(self):
        return self(len(segments))

    def pop(self, s):
        self.segments.pop()
        self.p.pop()

    def __len__(self):
      return len(self.segments)


class ExtendablePath:
    '''
    Implements a 3D path which can be continuously extended with new points.
    TODO(cvorbach) Unimplemented
    '''
    def __init__(self, knotPoints):
        raise NotImplementedError

        if len(knotPoints) < 4:
            raise ValueError('Extendable Path needs at least 4 knots')

        self.knots   = np.array(knotPoints)
        self.xSpline = CatmullRomSpline(list(self.knots[:, 0]))
        self.ySpline = CatmullRomSpline(list(self.knots[:, 1]))
        self.zSpline = CatmullRomSpline(list(self.knots[:, 2]))

    def extend(self, newPoints):
        self.xSpline.extend(list(newPoints[:, 0]))
        self.ySpline.extend(list(newPoints[:, 1]))
        self.zSpline.extend(list(newPoints[:, 2]))

    def __call__(self, t):
        return np.array([
            self.xSpline(t),
            self.ySpline(t),
            self.zSpline(t)
        ])

    def tangent(self, t):
        return np.array([
            self.xSpline.ddt(t),
            self.ySpline.ddt(t),
            self.zSpline.ddt(t)
        ])

    def project(self, point):
        tSamples    = np.linspace(0, 1, num=1000)
        nearestT    = tSamples[np.argmin([np.linalg.norm(point - self(t)) for t in tSamples])]

        return nearestT
    
    def end(self):
        return self.knots[-2]

    def __len__(self):
        return len(self.xSpline.segments)


class CubicSpline:
    '''
    Class for constructing natural cubic splines from (x,y) points.
    https://mathworld.wolfram.com/CubicSpline.html
    '''
    def __init__(self, x, y, tol=1e-10):
        if len(x) != len(y):
            raise ValueError(f'Lengths of spline interpolation x:{len(x)} and y:{len(y)} points don\'t match')

        if len(x) < 3:
            raise ValueError('Need at least three points to construct a spline')

        self.x = x
        self.y = y
        self.coeff = self.fit(x, y, tol)

        # print(self.x.shape)
        # print(self.y.shape)
        # print([c.shape for c in self.coeff])

    def fit(self, x, y, tol=1e-10):
        """
        Interpolate using natural cubic splines.
    
        Generates a strictly diagonal dominant matrix then solves.
    
        Returns coefficients:
        b, coefficient of x of degree 1
        c, coefficient of x of degree 2
        d, coefficient of x of degree 3
        """ 
        x = np.array(x)
        y = np.array(y)
    
        # check if sorted
        if np.any(np.diff(x) < 0):
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

        size = len(x)
        delta_x = np.diff(x)
        delta_y = np.diff(y)
    
        # Initialize to solve Ac = k
        A = np.zeros(shape = (size,size))
        k = np.zeros(shape=(size,1))
        A[0,0] = 1
        A[-1,-1] = 1
    
        for i in range(1,size-1):
            A[i, i-1] = delta_x[i-1]
            A[i, i+1] = delta_x[i]
            A[i,i] = 2*(delta_x[i-1]+delta_x[i])

            k[i,0] = 3*(delta_y[i]/delta_x[i] - delta_y[i-1]/delta_x[i-1])
    
        # Solves for c in Ac = k
        c = np.linalg.solve(A, k)
    
        # Solves for d and b
        d = np.zeros(shape = (size-1,1))
        b = np.zeros(shape = (size-1,1))
        for i in range(0,len(d)):
            d[i] = (c[i+1] - c[i]) / (3*delta_x[i])
            b[i] = (delta_y[i]/delta_x[i]) - (delta_x[i]/3)*(2*c[i] + c[i+1])    
    
        return b.squeeze(), c.squeeze(), d.squeeze()

    def __call__(self, t):
        '''
        Returns the value of the spline at t in [x[0], x[-1]]
        '''
        x = self.x
        y = self.y
        b, c, d = self.coeff

        # TODO(cvorbach) allow extrapolation
        if t < x[0] or t > x[-1]:
            raise ExtrapolationError("Can't extrapolate")

        # Index of segment to use
        idx = np.argmax(x > t) - 1
                
        dx = t - x[idx]
        value = y[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
        return value

    def ddt(self, t):
        '''
        Returns the derivative of the spline at t in [x[0], x[-1]]
        '''
        x = self.x
        y = self.y
        b, c, d = self.coeff

        # TODO(cvorbach) allow extrapolation
        if t < x[0] or t > x[-1]:
            raise ExtrapolationError("Can't extrapolate")

        # Index of segment to use
        idx = np.argmax(x > t) - 1

        dx         = t - x[idx]
        derivative = b[idx] + 2*c[idx]*dx + 3*d[idx]*dx**2
        return derivative

    def d2dt2(self, t):
        '''
        Returns the second derivative of the spline at t in [x[0], x[-1]]
        '''
        x = self.x
        y = self.y
        b, c, d = self.coeff

        # TODO(cvorbach) allow extrapolation
        if t < x[0] or t > x[-1]:
            raise ExtrapolationError("Can't extrapolate")

        # Index of segment to use
        idx = np.argmax(x > t) - 1
        secondDerivative = 2*c[idx] + 6*d[idx]*dx
        return secondDerivative

class Loop:
    def __init__(self, radialKnots, center=np.array([0,0,0]), orientation=R.from_euler('xyz',(0,0,0)), clockwise=True):
        '''
        Produces a closed polar loop with
        @param radialKnots knot values for radial spline interpolation
        @param center      3D origin point for the loop's radius
        @param orientation a rotation of the loop around the center relative to Airsim world axis
        @param clockwise   changes R(theta) to R(-theta) if false, e.i changes direction of rotation 
        '''
        self.radialKnots = radialKnots
        self.center      = center
        self.orientation = orientation
        self.clockwise   = clockwise
        self.fit(radialKnots)

    def fit(self, radialKnots):
        '''
        We fit each radius r equispaced t between 0 to 1
        '''
        knots = np.array(radialKnots)
        t     = np.linspace(0, 1, knots.shape[0])
        self.radialSpline = CubicSpline(t, knots)

    def __call__(self, t):
        '''
        Interpolate (x,y) point on radial curve 
        radius r(t) from spline(t) at t between 0 and 1
        theta(t) from theta between 0 and 3/2*pi and t between 0 and 1
        '''
        radius = self.radialSpline(t)
        if self.clockwise:
            angle  = 3/2*np.pi*t
        else:
            angle  = 3/2*np.pi*(1-t)

        loopPoint = radius * R.from_rotvec(angle*np.array([0,1,0])).apply(np.array([1,0,0]))
        loopPoint = self.orientation.apply(loopPoint)

        worldCoordinatePoint = np.array(self.center) + loopPoint
        return worldCoordinatePoint

    def project(self, point):
        '''
        Numerically finds the closest path(t) point
        to the passed point
        '''
        tSamples = np.linspace(0, 1, num=1000)
        nearstT  = tSamples[np.argmin([np.linalg.norm(point - self(t)) for t in tSamples])]
        return nearstT

def generateLoop(minRadius=6, maxRadius=9, knotCount=6, center=np.zeros((3,))):
    '''
    Generates a random closed loop with 'good' defaults.
    '''
    radialKnots = (maxRadius-minRadius)*np.random.random(knotCount) + minRadius
    radialKnots = np.array(list(radialKnots) + [radialKnots[0]])
    orientation = R.from_euler('xyz', (0,0,2*np.pi*random.random()))
    clockwise   = random.random() > 0.5

    return Loop(radialKnots, center=center, orientation=orientation, clockwise=clockwise)

class Trajectory:
    '''
    Implements a 3D + time spline trajectory
    '''

    def __init__(self, knotPoints):
        '''
        @param knotPoints [x,y,z,t] points
        '''
        self.knotPoints = np.array(knotPoints)
        self.fit(self.knotPoints)

    def fit(self, knotPoints):
        '''
        Constructs the path from the 3D knot points and times.
        @param knotPoints [x,y,z,t] points
        '''
        self.path = Path(knotPoints[:,:3], t=knotPoints[:, 3])

    def __call__(self, t):
        '''
        Returns the 3D point on path at t
        '''
        return self.path(t)

    def tangent(self, t):
        '''
        Returns the tangent vector to the path at t
        '''
        return self.path.tangent(t)

    def normal(self, t):
        '''
        Returns the Frenet-Serret normal vector at t
        '''
        return self.path.normal(t)

    def project(self, point):
        '''
        Numerically finds the closest path(t) point
        to the passed point
        '''
        return self.path.tangent(point)
    
    def end(self):
        return self.knotPoints[-1]

class Path:
    '''
    Implements a 3D spline path. 

    TODO(cvorbach) Add support for 1 and 2 point long paths,
                   currently these very short paths cause errors b/c can't build a spline
    '''
    def __init__(self, knotPoints, t=None):
        self.fit(knotPoints, t)

    def fit(self, knotPoints, t=None):
        '''
        Constructs the path from the 3D knot points.

        This can safely be updated with new knot points,
        but doing changes the t value of points on the curve.
        '''

        knots = np.array(knotPoints)

        if t is None:
            t = np.linspace(0, 1, knots.shape[0])

        self.xSpline = CubicSpline(t, knots[:, 0])
        self.ySpline = CubicSpline(t, knots[:, 1])
        self.zSpline = CubicSpline(t, knots[:, 2])
        self.knotPoints = knotPoints

    def __call__(self, t):
        '''
        Returns the 3D point on path at t
        '''
        return np.array([
            self.xSpline(t),
            self.ySpline(t),
            self.zSpline(t)
        ])

    def tangent(self, t):
        '''
        Returns the tangent vector to the path at t
        '''
        return np.array([
            self.xSpline.ddt(t),
            self.ySpline.ddt(t),
            self.zSpline.ddt(t)
        ])

    def normal(self, t):
        '''
        Returns the Frenet-Serret normal vector at t
        '''
        tangentDerivative = np.array([
            self.xSpline.d2dt2(t),
            self.ySpline.d2dt2(t),
            self.zSpline.d2dt2(t)
        ])
        return normalize(tangentDerivative)

    def project(self, point):
        '''
        Numerically finds the closest path(t) point
        to the passed point
        '''
        tSamples = np.linspace(0, 1, num=1000)
        nearstT  = tSamples[np.argmin([np.linalg.norm(point - self(t)) for t in tSamples])]
        return nearstT
    
    def end(self):
        return self.knotPoints[-1]

# claInfinitePath:
#     def __init__(self, start, occupancyMap, momentumWeight=0.9, stepSize=10, inclinationLimit=0.1, zLimit=(-20, -10)):
        # '''
        # The idea here is to continuously extend the path as we move, allowing infinite journeys.
        # '''
#         self.occupancyMap     = occupancyMap
#         self.momentumWeight   = momentumWeight
#         self.stepSize         = stepSize
#         self.inclinationLimit = inclinationLimit
# 
#         self.x                = start
#         self.momentum         = normalize(np.array([random.random(), random.random(), 0]))
# 
#         self.generateInitialPath()
# 
#     def getStep(self):
#         perturbance   = normalize(np.random.random_sample(3))
#         stepDirection = self.momentumWeight*self.momentum + (1-self.momentumWeight)*perturbance
#         stepDirection = normalize(stepDirection)
# 
#         self.momentum = stepDirection
#         step = self.stepSize * stepDirection
# 
#         return self.x + step
# 
#     def generateInitialPath(self, start):
#         isCollisionFree = False
#         while not isCollisionFree:
# 
# 
#         initialSteps = [self.getStep()]
# 
#     def ignite(self):
#         p = []
#         for i in range(4):
#             x, 
#             p.append() 
# 
#     # def update(self, nextX, nextMomentum):
#     #     self.x        = nextX
#     #     self.momentum = nextMomentum

def randomWalk(start, momentumWeight=0.5, stepSize=3, gradientLimit=np.pi/12, zLimit=(-20, -10), pathLength=5, occupancyMap=None, retryLimit = 10):
    '''
    Attempts to find an 'good' collision free path through the voxel map
    '''
    normalDistribution = np.random.default_rng().normal 
    momentum = normalize(np.array([normalDistribution(), normalDistribution(), normalDistribution()]))
    path = [start]

    for i in range(retryLimit):
        for _ in range(pathLength):

            # Generate an unoccupied next step in the random walk
            isUnoccupiedNextStep = False
            stuckSteps = 0
            while not isUnoccupiedNextStep:
                rotation = R.from_matrix([[momentum[0], 0, 0], [0, momentum[1], 0], [0, 0, momentum[2]]])
                perturbance = rotation.apply(normalize(np.array((0, normalDistribution(), normalDistribution()))))

                # the continue in previous direction with a random perturbance left/right and up/down
                stepDirection = normalize(momentumWeight * momentum + (1 - momentumWeight) * perturbance)

                # apply gradient limit
                heading = rotation.apply([1, 0, 0])
                stepDirection  = np.array([min(max(u - heading[i], -gradientLimit), gradientLimit) for i, u in enumerate(stepDirection)]) 
                stepDirection += heading

                step          = stepSize * stepDirection

                nextStep = path[-1] + step

                # apply altitude limits
                nextStep[2] = min(max(nextStep[2], zLimit[0]), zLimit[1])

                isUnoccupiedNextStep = occupancyMap is None or nextStep not in occupancyMap 

                stuckSteps += int(not isUnoccupiedNextStep)
                if stuckSteps > retryLimit:
                    break 

            path.append(nextStep)

        if stuckSteps < retryLimit:
            break

    if i > retryLimit:
        raise Exception('Couldn\'t find free random walk')

    return path

# Utilities
class LRUCache:
    '''
    Least-recently-used hash cache. 
    Used to implement local voxel occupancy testing in O(1) time. 
    '''
    def __init__(self, capacity: int):
        '''
        @param capacity max number of keys to store in the cache
        '''
        self.cache = OrderedDict()
        self.capacity = capacity

    def __contains__(self, key):
        '''
        Check if key in cache, mark as most-recently-used if so
        '''
        if key in self.cache:
            self.cache.move_to_end(key) # Move to front of LRU cache
            return True
        return False

    def add(self, key):
        '''
        Add a key to the cache, drop the least-recently-used cache
        item if we've reached the max capacity.
        '''
        self.cache[key] = None          # Don't care about the dict's value, just its set of keys
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def discard(self, key):
        '''
        Drops key from the cache
        '''
        self.cache.pop(key, None)

    def keys(self):
        '''
        Returns a list of all the keys in the cache.
        '''
        return list(self.cache.keys())

class VoxelOccupancyCache:
    '''
    Stores voxels which the drone can not pass through.

    These are the voxels which are passed as 'occupied'
    as well as the voxels adjacent to occupied voxels.
    This lets the cache represent configuration space.

    See addPoint for details.
    '''

    def __init__(self, voxelSize: float, capacity: int):
        '''
        @param voxelSize The side length of a voxel
        @param capacity  Max number of occupied voxels to cache
        '''
        self.voxelSize  = voxelSize
        self.cache      = LRUCache(capacity)
    
    def addPoint(self, point):
        '''
        @param point a point to add to the cache.

        Note, adding point also marks all of the points
        adjacent to it as occupied. This is to prevent the 
        drone from clipping / bumping against occupied voxels.
        E.I. we want to store configuration space rather than
        obstacle space.
        
        This is required because the drone has a non-zero radius.
        If you decrease the voxelSize too much, you may need to
        add more neighbors to the map to prevent this problem.
        '''
        voxel = self.point2Voxel(point)

        self.cache.add(voxel)
        for v in self.getAdjacentVoxels(voxel):
            self.cache.add(v)

    def __contains__(self, point):
        '''
        Check if drone can move through point.
        '''
        voxel = self.point2Voxel(point)
        return voxel in self.cache

    def point2Voxel(self, point):
        '''
        Round to nearset voxel and convert to hashable type.
        '''
        return tuple(self.voxelSize * int(round(v / self.voxelSize)) for v in point)

    def getAdjacentVoxels(self, voxel):
        '''
        Get the 27 voxels around voxel, (including the original voxel itself)
        This is manually unrolled for performance reasons.
        '''
        adjacentVoxels = [
            (voxel[0] - 1, voxel[1] - 1, voxel[2] - 1),
            (voxel[0],     voxel[1] - 1, voxel[2] - 1),
            (voxel[0] + 1, voxel[1] - 1, voxel[2] - 1),

            (voxel[0] - 1, voxel[1], voxel[2] - 1),
            (voxel[0],     voxel[1], voxel[2] - 1),
            (voxel[0] + 1, voxel[1], voxel[2] - 1),

            (voxel[0] - 1, voxel[1] + 1, voxel[2] - 1),
            (voxel[0],     voxel[1] + 1, voxel[2] - 1),
            (voxel[0] + 1, voxel[1] + 1, voxel[2] - 1),

            (voxel[0] - 1, voxel[1] - 1, voxel[2]),
            (voxel[0],     voxel[1] - 1, voxel[2]),
            (voxel[0] + 1, voxel[1] - 1, voxel[2]),

            (voxel[0] - 1, voxel[1], voxel[2]),
            (voxel[0],     voxel[1], voxel[2]),
            (voxel[0] + 1, voxel[1], voxel[2]),

            (voxel[0] - 1, voxel[1] + 1, voxel[2]),
            (voxel[0],     voxel[1] + 1, voxel[2]),
            (voxel[0] + 1, voxel[1] + 1, voxel[2]),

            (voxel[0] - 1, voxel[1] - 1, voxel[2] + 1),
            (voxel[0],     voxel[1] - 1, voxel[2] + 1),
            (voxel[0] + 1, voxel[1] - 1, voxel[2] + 1),

            (voxel[0] - 1, voxel[1], voxel[2] + 1),
            (voxel[0],     voxel[1], voxel[2] + 1),
            (voxel[0] + 1, voxel[1], voxel[2] + 1),

            (voxel[0] - 1, voxel[1] + 1, voxel[2] + 1),
            (voxel[0],     voxel[1] + 1, voxel[2] + 1),
            (voxel[0] + 1, voxel[1] + 1, voxel[2] + 1),
        ]

        return adjacentVoxels

    def getNextSteps(self, voxel, endpoint=None, endpoint_tolerance=None):
        '''
        Returns the possible voxels the drone can move into from
        voxel. 

        This is all the unoccupied adjacent voxels and the original voxel itself.

        We ignore the unoccupied requirement around the 'endpoint' to support
        tasks like hiking where we must plan a path into a final occupied voxel.
        '''
        neighbors = []
        possibleNeighbors = self.getAdjacentVoxels(voxel)

        for v in possibleNeighbors:
            isFeasible = v not in self.cache
            if endpoint is not None:
                isFeasible = isFeasible or distance(np.array(v), np.array(endpoint)) < endpoint_tolerance

            if isFeasible:
                neighbors.append(v) 

        return neighbors

    def plotOccupancies(self):
        '''
        Plots all of the occupied points. This will crash Airsim if left running.
        '''
        occupiedPoints = [Vector3r(float(v[0]), float(v[1]), float(v[2])) for v in self.cache.keys()]
        client.simPlotPoints(occupiedPoints, color_rgba = [0.0, 0.0, 1.0, 1.0], duration=args.plot_period/2.0) 


def world2UnrealCoordinates(vector):
    '''
    Converts from airsim to unreal coordinates system. 

    The airsim coordinates are meters with negative z
    https://github.com/microsoft/AirSim/blob/master/docs/apis.md#:~:text=All%20AirSim%20API%20uses%20NED,used%20internally%20by%20Unreal%20Engine.&text=The%20starting%20point%20of%20the,%2C%200)%20in%20NED%20system.

    The unreal coordinates are basically centimeters with different origin.
    TODO(cvorbach) Do I need to flip the z-direction?
    '''
    return (vector + DRONE_START) * WORLD_2_UNREAL_SCALE


def unreal2WorldCoordinates(vector):
    '''
    Converts from unreal to airsim coordinates system. 

    The airsim coordinates are meters with negative z
    https://github.com/microsoft/AirSim/blob/master/docs/apis.md#:~:text=All%20AirSim%20API%20uses%20NED,used%20internally%20by%20Unreal%20Engine.&text=The%20starting%20point%20of%20the,%2C%200)%20in%20NED%20system.

    The unreal coordinates are basically centimeters with different origin.
    TODO(cvorbach) Do I need to flip the z-direction?
    '''
    return (vector - DRONE_START) / WORLD_2_UNREAL_SCALE

def isVisible(point, position, orientation, config):
    '''
    Checks if a point is in the camera frustrum if the drone
    is at position in orientation.

    TODO(cvorbach) implement occlusion checking
    '''
    # print('point', point)
    # print('position', position)
    # print('orientation', orientation.as_euler('xyz'))
    cameraUnit = np.array([1,0,0])
    cameraUnit = orientation.apply(cameraUnit)

    pointRay = normalize(point - position)

    angle = np.arccos(np.dot(cameraUnit, pointRay)) 
    # print('pointRay', pointRay)
    # print('cameraUnit', cameraUnit)
    # print('angle', angle)
    return abs(angle) < config['camera_field_of_view']

# def isVisible(point, position, orientation):
#     '''
#     Checks if a point is in the camera frustrum if the drone
#     is at position in orientation.
# 
#     TODO(cvorbach) implement occlusion checking
#     '''
#     # Edge case point == position
#     if distance(point, position) < 0.05:
#         return True
# 
#     # Check if endpoint is in frustrum
#     xUnit = np.array([1, 0, 0])
#     cameraDirection = R.from_quat(orientation).apply(xUnit)
# 
#     endpointDirection = normalize(point - position)
# 
#     # TODO(cvorbach) check square, not circle
#     angle = np.arccos(np.dot(cameraDirection, endpointDirection)) 
# 
#     if abs(angle) > CAMERA_FOV:
#         return False
# 
#     # TODO(cvorbach) Check for occlusions with ray-tracing
# 
#     return True


def directionOf(point, position):
    '''
    Get the drone orientation that has camera facing towards the point at position
    '''
    displacement = np.array(point) - np.array(position)
    pointYaw = np.arctan2(displacement[1], displacement[0])
    orientation = R.from_euler('xyz', [0, 0, pointYaw]).as_quat()

    return orientation


def euclidean(voxel1, voxel2):
    '''
    Euclidean distance metric
    '''
    return distance(np.array(voxel1), np.array(voxel2))


def greedy(voxel1, voxel2):
    '''
    Gredy metric just heavily weighs euclidean distance
    '''
    return 100*euclidean(voxel1, voxel2)


def findPath(startpoint, endpoint, occupancyMap, endpoint_tolerance, h=greedy, d=euclidean):
    '''
    Workhorse path finding algorithm. 
    
    Finds a path of adjacent unoccupied voxels from from startpoint's voxel to endpoint's voxel.

    If h=euclidean and d=euclidean, then this is A*. If h=greedy and d=euclidean, then this is greedy search.
    Greedy search *almost never* is non-optimal and is much faster than A* because the environment is so sparse. 

    Ignores the occupancy requirement within endpoint_tolerance of the endpoint to support tasks
    like hiking which plan into final occupied voxel.
    '''
    start = occupancyMap.point2Voxel(startpoint)
    end   = occupancyMap.point2Voxel(endpoint)

    cameFrom = dict()

    gScore = dict()
    gScore[start] = 0

    fScore = dict()
    fScore[start] = h(start, endpoint)

    openSet = [(fScore[start], start)]

    while openSet:
        current = heapq.heappop(openSet)[1]

        # client.simPlotPoints([Vector3r(*current)], duration = 60)

        if current == end:
            path = [current]
            while path[-1] != start:
                current = cameFrom[current]
                path.append(current)
            
            return list(reversed(path))

        for neighbor in occupancyMap.getNextSteps(current, endpoint=end, endpoint_tolerance=endpoint_tolerance):

            # TODO(cvorbach) It would be nice to get this working
            # # skip neighbors from which the endpoint isn't visible
            # neighborOrientation = directionOf(endpoint, neighbor)
            # if not isVisible(np.array(end), np.array(neighbor), neighborOrientation):
            #     continue

            tentativeGScore = gScore.get(current, float("inf")) + d(current, neighbor)

            if tentativeGScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = current
                gScore[neighbor]   = tentativeGScore

                if neighbor in fScore:
                    try:
                        openSet.remove((fScore[neighbor], neighbor))
                    except:
                        pass
                fScore[neighbor]   = gScore.get(neighbor, float('inf')) + h(neighbor, endpoint)

                heapq.heappush(openSet, (fScore[neighbor], neighbor))
        
    raise ValueError("Couldn't find a path")


def genericDFS(start, isEnd, getFeasibleNext, h):
    openSet = []
    openSet.append(start)
    cameFrom = dict()

    while openSet:
        current = openSet.pop()
        print(current)

        # Test if we've found node in the end set
        if isEnd(current):
            # Walk back through parent pointers to find search path
            trajectory = [current]
            while trajectory[-1] != start:
                current = cameFrom[current]
                trajectory.append(current)

            return list(reversed(trajectory))

        for neighbor in sorted(getFeasibleNext(current), key=h, reverse=True):
            cameFrom[neighbor] = current
            openSet.append(neighbor)


def genericHeuristicSearch(start, isEnd, getFeasibleNext, h=euclidean, d=euclidean):
    '''
    Implements highly generic heuristic search. 
    @param start starting node
    @param isEnd(node) function which is True iff passed a valid end node
    @param getFeasibleNext(node) function which produces feasible adjacent nodes
    @param h heuristic function
    @param d edge cost function

    If h is consistent and admissible, then this is A*
    '''
    
    cameFrom = dict()

    gScore = dict()
    gScore[start] = 0

    fScore = dict()
    fScore[start] = h(start)

    openSet = [(fScore[start], start)]

    i = 0
    while openSet:
        if i % 100 == 1:
            print('t', current[3])
        i += 1
        current = heapq.heappop(openSet)[1]

        # Test if we've found node in the end set
        if isEnd(current):
            # Walk back through parent pointers to find search path
            trajectory = [current]
            while trajectory[-1] != start:
                current = cameFrom[current]
                trajectory.append(current)

            return list(reversed(trajectory))

        # print('feasible', getFeasibleNext(current))

        # Else, find the possible next steps
        for neighbor in getFeasibleNext(current):

            # Cost to reach neighbor from current
            tentativeGScore = gScore.get(current, float('inf')) + d(current, neighbor)

            # If it is less costly to reach neighbor from current
            # then update neighbor's cost-to-go estimate
            if tentativeGScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = current
                gScore[neighbor]   = tentativeGScore

                if neighbor in fScore:
                    try:
                        openSet.remove((fScore[neighbor], neighbor))
                    except:
                        pass # Ignore if neighbor wasn't already on the heap

                # Put neighbor back on heap with new cost-to-go
                fScore[neighbor] = gScore.get(neighbor, float('inf')) + h(neighbor)
                heapq.heappush(openSet, (fScore[neighbor], neighbor))

    raise ValueError("Search failed: no path-to-end found.")


def walzBoundingSphere(points):
    C, r2 = miniball.get_bounding_ball(np.array(points))
    return C, np.sqrt(r2)


def findTrackingKnots(startingPosition, targetTrajectories, occupancy_cache, config, dt=1e-1, Qp = np.eye(3), Qc = np.eye(4)):
    '''
    Our goal is to find a drone trajectory which keeps all of the targets in
    the camera frustrum as the targets move.

    @param targetKnots (numTargets, numKnots, len([x,y,z]))
    @param trajectoryTime len(trajectoryTime) = numKnots, the time of each knot from start
    '''

    endTime = targetTrajectories[0].end()[3]
    N = math.ceil(endTime/dt)
    time = np.linspace(0, endTime, N)

    # True if knot is in the end set
    def isEnd(knot):
        t = knot[3]
        return t >= endTime

    # Produce all possible next knots
    def getFeasibleNext(knot):
        # print('t', knot[3] + dt, endTime)
        p = np.array(knot[:3])
        t = min(knot[3] + dt, endTime)
        v = occupancy_cache.point2Voxel(p)

        adjacentVoxels = occupancy_cache.getNextSteps(v)
        
        # print('AV', adjacentVoxels)

        if v not in adjacentVoxels:
            raise RuntimeError('Staying in voxel wasn\'t a returned option, but it should be.')

        center, radius = walzBoundingSphere([traj(t) for traj in targetTrajectories])
        yawAngle = np.arctan2((center - p)[1], (center - p)[0])
        orientation = R.from_euler('xyz', [0,0,yawAngle])

        # print('position', p)
        # print('centerRay', normalize(center-p))

        feasibleNeighbors = []
        for neighbor in adjacentVoxels:

            allTargetsVisible = True
            for traj in targetTrajectories:
                # print(isVisible(traj(t), np.array(neighbor), orientation, config))
                allTargetsVisible = allTargetsVisible and isVisible(traj(t), np.array(neighbor), orientation, config)
                if not allTargetsVisible:
                    break
            
            if allTargetsVisible:
                feasibleNeighbors.append(neighbor)

        # print('t*', knot[3] + dt, endTime)
        return [(*voxel, t) for voxel in feasibleNeighbors]

    def costToGoHeuristic(knot, position, orientation):
        #TODO(cvorbach) Think about me?
        # centroid projected distance in camera plane
        # and euclidean distance from the start (make edge cost linear)
        p = np.array(knot[:3])
        t = knot[3]
        s = np.array(start[:3])

        center, radius = walzBoundingSphere([traj(t) for traj in targetTrajectories])

        z2x = R.from_euler('xyx', (-np.pi/2, -np.pi/2, 0))     # Rotates expected camera coordinates (z,x,y) to drone body frame (x,y,z)
        cameraRotation = (z2x*orientation).inv().as_matrix()   # Rotate drone body frame to world frame, then get the inverse rotation
        t = -cameraRotation.dot(position)[:, np.newaxis]       # Translation
        P = np.concatenate((cameraRotation, t), axis=1)
        P = np.concatenate((P, np.array(((0,0,0,1),))), axis=0) # Homogenous coordinates camera projection (unit focal length and pixel size)

        centerTilde  = np.array(((*center, 1),))       # Location of bounding sphere center in homogenous coordinates
        centerTilde  = P.dot(pTilde)                   # Project to image plane 
        centerCamera = centerTilde[:2]/centerTilde[2]  # Get camera x,y coordinates

        projectiveCost = np.linalg.norm(centerCamera)
        movementCost   = distance(p,s)

        return projectiveCost + movementCost

    def edgeCost(knot1, knot2):
        p1 = np.array(knot1[:3])
        p2 = np.array(knot2[:3])

        distanceCost = distance(p1, p2)
        # TODO(cvorbach) add difference in projected distance from camera plane origin
        return distanceCost

    start = (*startingPosition, 0)
    #trackingKnots = genericHeuristicSearch(start, isEnd=isEnd, getFeasibleNext=getFeasibleNext, h=costToGoHeuristic, d=edgeCost)
    trackingKnots = genericDFS(start, isEnd, getFeasibleNext, costToGoHeuristic)
    return trackingKnots
