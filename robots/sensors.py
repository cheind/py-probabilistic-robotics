import numpy as np
import math

from robots import transforms

class LandmarkSensor:

    def __init__(self, landmarks, **kwargs):
        self.sense_err = kwargs.pop('err', 0)
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        self.measure = kwargs.pop('measure', 'position')
        self.landmarks = transforms.hom(landmarks)

    def sense(self, robot, **kwargs):        
        sense_err = kwargs.pop('err', self.sense_err)
        
        # Transform into robot space
        tl = np.dot(robot.world_in_robot(), self.landmarks)
        tl = transforms.hnorm(tl)

        # Add noise to measurement
        dists = np.linalg.norm(tl, axis=0)
        sigma = dists * sense_err
        e = np.random.randn(2, self.landmarks.shape[1])
        e *= sigma[np.newaxis, :]
        tl += e

        angles = np.arctan2(tl[1], tl[0])

        mask = np.logical_and(dists <= self.maxdist, np.abs(angles) <= self.fov / 2)

        if self.measure == 'position':
            return mask, tl
        elif self.measure == 'bearing':
            return mask, angles
        
