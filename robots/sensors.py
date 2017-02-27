import numpy as np
import math

from robots import transforms
from robots.posenode import PoseNode


class LandmarkSensor(PoseNode):
    """A sensor simulating detection of landmarks.
    
    Landmarks are identifyable points in the environment. The simulation assumes
    that the true landmark positions are known and those do not change of time. The
    sensor additional assumes that the indexing of landmarks remains constant.
    
    The sensor supports custom field of views as well as limiting the max distance
    to detectable landmarks. Optionally occluded landmarks can be disregarded when
    an environment Grid is given. An error model allows measuring deviations from
    true landmark positions.

    Besides relative landmark positions with respect to this sensor, reporting 
    bearings (angles) or distances is also supported.    
    """

    def __init__(self, landmarks, **kwargs):
        """Create a LandmarkSensor.

        Params
            landmarks : 2xN vector of true landmark positions in world frame.

        Kwargs
            err : (optional) scalar 
                The error associated with sensing. The error model assumes a zero centered normal 
                distribution having a standard deviation proportional to the distance of landmarks.
            fov : (optional) scalar 
                Field of view of sensor [0..2pi]
            maxdist : (optional) scalar
                Maximum detectable range
            measure : (optional) string
                Type of measurement. One of ['position', 'bearing', 'distance']
            environment : (optional) Grid 
                Grid used to determine occlusion of landmarks with respect to sensor.
            pose : (optional) 1x3 array
                Pose vector. If omitted identity is assumed.
        """
        self.sense_err = kwargs.pop('err', 0)
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        self.measure = kwargs.pop('measure', 'position')
        self.environment = kwargs.pop('environment', None)
        self.landmarks = landmarks

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(LandmarkSensor, self).__init__(pose=pose)

    def sense(self, **kwargs):
        """Observe landmarks.

        Kwargs
            err : (optional) scalar 
                The error associated with sensing. The error model assumes a zero centered normal distribution 
                having a standard deviation proportional to the distance of landmarks. If omitted value from 
                construction is used.
            environment : (optional) Grid 
                Grid to determine occlusion of landmarks with respect to sensor. If omitted value from 
                construction is used.
            measure : (optional) string
                Type of measurement. One of ['position', 'bearing', 'distance']. If omitted value from 
                construction is used.

        Returns
            mask : 1xN boolean array
                A mask indicating the visibility of individual landmarks
            position : 2xN array 
                Landmark positions with respect to sensor frame. Only if measure is 'position'            
            bearing : 1xN array 
                Landmark bearings measured with respect to the sensor x-axis. Only if measure is 'bearing'
            distance : 1xN array 
                Euclidean distances to sensor position. Only if measure is 'distance'
        """

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
            return mask, noisy_landmarks
        elif measure == 'bearing':
            return mask, angles
        elif measure == 'distance':
            dists = np.linalg.norm(noisy_landmarks, axis=0)
            return mask, dists
        else:
            raise ValueError('Unknown measure type')
        

class LidarSensor(PoseNode):
    """A sensor simulating LIDAR range sensors.

    LIDAR measures distances through the reflection of laser beams sent out
    radially from the sensor. For each ray a range measurement is produced if 
    an obstacle reflected the light.

    This sensor simulation takes an evironment map (Grid) to determine
    intersections with light rays. The sensor features a field of view, a max 
    range distance as well as angular resolution. The error model produces
    errors proportional to the distance of intersections.    
    """

    def __init__(self, environment, **kwargs):
        """Create a LidarSensor.

        Params
            environment : Grid
                Grid used to determine intersection with light rays.

        Kwargs
            err : (optional) scalar 
                The error associated with sensing. The error model assumes a zero centered normal 
                distribution having a standard deviation proportional to the distance of intersections.
            fov : (optional) scalar 
                Field of view of sensor [0..2pi]
            maxdist : (optional) scalar
                Maximum detectable range
            angular_resolution : (optional) scalar
                Angular resolution between two consecutive rays in radians.
            pose : (optional) 1x3 array
                Pose vector. If omitted identity is assumed.
        """
        self.environment = environment

        self.sense_err = kwargs.pop('err', 0)
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        self.angular_res = kwargs.pop('angular_resolution', 0.1)        
        self.angles = np.arange(-self.fov/2, self.fov/2, self.angular_res)
        
        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(LidarSensor, self).__init__(pose=pose)

    def sense(self, **kwargs):
        """Perform range measurements.

        Kwargs
            err : (optional) scalar 
                The error associated with sensing. The error model assumes a zero centered normal distribution 
                having a standard deviation proportional to the distance of intersections. If omitted value from 
                construction is used.
            environment : (optional) Grid 
                Grid to determine occlusion of landmarks with respect to sensor. If omitted value from 
                construction is used.

        Returns
            mask : 1xN boolean array
                A mask indicating the success of range measurement of individual rays
            position : 2xN array 
                Ray/environment intersection locations in the sensor frame.
        """
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
