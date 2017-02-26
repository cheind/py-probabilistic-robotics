import numpy as np
import math

import transforms

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
        
if __name__ == '__main__':
    
    from robots import Robot
    from draw import RobotDrawer, LandmarkDrawer, LandmarkSensorDrawer
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    landmarks = np.array([
        [0, 10, 3],
        [0, 10, 5]
    ], dtype=float)
    s = LandmarkSensor(landmarks, err=0.01, fov=math.pi/4, maxdist=5., measure='bearing')
    r = Robot(state=[0,0,0])
    
    rd = RobotDrawer()
    rsd = LandmarkSensorDrawer()
    ld = LandmarkDrawer()

    fig, ax = plt.subplots()
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_aspect('equal')
    ax.grid()

    def update(i):
        r.move([0.05, 0.1])
        print(s.sense(r)[0]) # print mask of visible landmarks
        u = rd.draw(r, ax, key='robot')
        u += rsd.draw(r, s, ax, key='sensor')
        landmarks[0, 0] += 0.1
        u += ld.draw(landmarks, ax, key='landmarks', with_labels=True)
        return u

    ani = animation.FuncAnimation(fig, update, 25, interval=30, blit=True)
    plt.show()


