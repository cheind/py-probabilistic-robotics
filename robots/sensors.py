import numpy as np
import math

from robots import transforms
from robots.posenode import PoseNode


class LandmarkSensor(PoseNode):

    def __init__(self, landmarks, **kwargs):
        self.sense_err = kwargs.pop('err', 0)
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        self.measure = kwargs.pop('measure', 'position')
        self.environment = kwargs.pop('environment', None)
        self.landmarks = landmarks

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(LandmarkSensor, self).__init__(pose=pose)

    def sense(self, **kwargs):        
        sense_err = kwargs.pop('err', self.sense_err)
        environment = kwargs.pop('environment', self.environment)
        measure = kwargs.pop('measure', self.measure)
        
        # Transform into sensor space
        local_landmarks = transforms.transform(self.transform_from_world, self.landmarks)

        # Add noise to measurement
        dists = np.linalg.norm(local_landmarks, axis=0)
        sigma = dists * sense_err
        e = np.random.randn(2, self.landmarks.shape[1])
        e *= sigma[np.newaxis, :]
        noisy_landmarks = local_landmarks + e

        # Compute angles w.r.t landmarks in -pi,pi
        angles = np.arctan2(noisy_landmarks[1], noisy_landmarks[0])

        # Determine seen / unseen elements
        mask = np.logical_and(dists <= self.maxdist, np.abs(angles) <= self.fov / 2)

        # If obstacles are provided determine occluded landmarks via ray-tracing
        if environment:
            to_env = self.transform_to(environment)
            env_o = to_env[:2, 2]
            env_landmarks = transforms.transform(to_env, local_landmarks)
            for i, m in enumerate(mask):
                if not m:
                    continue                    
                d = env_landmarks[:2, i] - env_o
                n = np.linalg.norm(d)
                d /= n
                ret, t, cell = environment.intersect_with_ray(env_o, d, tmax=n)
                mask[i] = t > n

        if measure == 'position':
            return mask, tl
        elif measure == 'bearing':
            return mask, angles
        elif measure == 'distance':
            dists = np.linalg.norm(tl, axis=0)
            return mask, dists
        else:
            raise ValueError('Unknown measure type')
        

class LidarSensor(PoseNode):

    def __init__(self, environment, **kwargs):
        self.environment = environment
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.angular_res = kwargs.pop('angular_resolution', 0.1)        
        self.angles = np.arange(-self.fov/2, self.fov/2, self.angular_res)
        self.sense_err = kwargs.pop('err', 0)
        
        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(LidarSensor, self).__init__(pose=pose)

    def sense(self, **kwargs):
        sense_err = kwargs.pop('err', self.sense_err)
        environment = kwargs.pop('environment', self.environment)

        mask = np.zeros(len(self.angles), dtype=bool)
        points = np.zeros((2, len(self.angles)))

        to_env = self.transform_to(environment)        
        pose = transforms.pose_from_transform(to_env)
        for i, a in enumerate(self.angles):
            d = np.array([math.cos(pose[2] - a), math.sin(pose[2] - a)])
            ret, t, cell = environment.intersect_with_ray(pose[:2], d, tmax=self.maxdist)
            if ret:
                points[:, i] = pose[:2] + t * d
                mask[i] = True

        from_env = transforms.rigid_inverse(to_env)

        return mask, transforms.transform(from_env, points)
