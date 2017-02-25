import numpy as np
import math

class LandmarkSensor:

    def __init__(self, landmarks, **kwargs):
        self.landmarks = landmarks
        self.sense_err = kwargs.pop('err', 0)
        self.fov = kwargs.pop('fov', 2 * math.pi)
        self.maxdist = kwargs.pop('maxdist', np.finfo(np.float32).max)
        self.measure = kwargs.pop('measure', 'position')

    def sense(self, robot, **kwargs):        
        sense_err = kwargs.pop('err', self.sense_err)
        
        # Transform into robot space
        tl = np.empty((3, self.landmarks.shape[0]))
        tl[:-1,:] = self.landmarks
        tl[-1, :] = 1.
        tl = np.dot(robot.world_in_robot(), tl)        

        # Add noise to measurement
        dists = np.linalg.norm(tl, axis=0)
        sigma = dists * sense_err
        e = np.random.randn(self.landmarks.shape[0], self.landmarks.shape[1])
        e *= sigma[np.newaxis, :]
        tl += e

        if self.measure == 'position':
            return tl
        
if __name__ == '__main__':
    
    from robots import Robot

    landmarks = np.array([
        [0, 10],
        [0, 10]
    ])
    s = LandmarkSensor(landmarks, err=0.01)

    r = Robot(state=[1,5,0])
    print(s.sense(r))

