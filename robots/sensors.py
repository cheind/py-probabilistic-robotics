import numpy as np
import math

from robots import transforms
from robots.frames import Frame

class LandmarkSensor:

    def __init__(self, landmarks, **kwargs):
        self.sense_err = kwargs.pop('err', 0)
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        self.measure = kwargs.pop('measure', 'position')
        self.environment = kwargs.pop('environment', None)
        self.landmarks = transforms.h(landmarks)
        self.frame = Frame(kwargs.pop('frame', 'robot'))

    def sense(self, robot, **kwargs):        
        sense_err = kwargs.pop('err', self.sense_err)
        environment = kwargs.pop('environment', self.environment)
        measure = kwargs.pop('measure', self.measure)
        
        # Transform into robot space
        pose = robot.pose
        tl = np.dot(robot.world_in_robot, self.landmarks)
        tl = transforms.hnorm(tl)

        # Add noise to measurement
        dists = np.linalg.norm(tl, axis=0)
        sigma = dists * sense_err
        e = np.random.randn(2, self.landmarks.shape[1])
        e *= sigma[np.newaxis, :]
        tl += e

        # Compute angles w.r.t landmarks in -pi,pi
        angles = np.arctan2(tl[1], tl[0])

        # Determine seen / unseen elements
        mask = np.logical_and(dists <= self.maxdist, np.abs(angles) <= self.fov / 2)

        # If obstacles are provided determine occluded landmarks via ray-tracing
        if environment:
            o = robot.pose[:2]
            for i, m in enumerate(mask):
                if not m:
                    continue                    
                d = self.landmarks[:2, i] - o
                n = np.linalg.norm(d)
                d /= n
                ret, t, cell = environment.intersect_with_ray(o, d, tmax=n)
                mask[i] = t > n

        if measure == 'position':
            if self.frame == Frame.Robot:                
                return mask, tl
            elif self.frame == Frame.World:
                return mask, transforms.transform(robot.robot_in_world, tl)
        elif measure == 'bearing':
            return mask, angles
        elif measure == 'distance':
            dists = np.linalg.norm(tl, axis=0)
            return mask, dists
        else:
            raise ValueError('Unknown measure type')
        

class LidarSensor:

    def __init__(self, environment, **kwargs):
        self.environment = environment
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.angular_res = kwargs.pop('angular_resolution', 0.1)        
        self.angles = np.arange(-self.fov/2, self.fov/2, self.angular_res)
        self.sense_err = kwargs.pop('err', 0)
        self.frame = Frame(kwargs.pop('frame', 'robot'))

    def sense(self, robot, **kwargs):
        sense_err = kwargs.pop('err', self.sense_err)
        environment = kwargs.pop('environment', self.environment)

        mask = np.zeros(len(self.angles), dtype=bool)
        points = np.zeros((2, len(self.angles)))

        pose = robot.pose
        for i, a in enumerate(self.angles):
            d = np.array([math.cos(pose[2] - a), math.sin(pose[2] - a)])
            ret, t, cell = environment.intersect_with_ray(pose[:2], d, tmax=self.maxdist)
            if ret:
                points[:, i] = pose[:2] + t * d
                mask[i] = True

        if self.frame == Frame.Robot:
            points = transforms.transform(robot.world_in_robot, points)

        return mask, points
