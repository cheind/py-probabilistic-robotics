

import numpy as np
from robots.posenode import PoseNode
from robots.robots import XYRobot
from robots.grid import Grid
from robots.sensors import LandmarkSensor, LidarSensor
from robots.bbox import BBox
from robots.draw import Drawer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

class Kalman:
    def __init__(self, true_state, x, P):
        self.true_state = true_state.copy()    
        self.x = x.copy()
        self.P = P.copy()

        self.F = np.eye(x.shape[0])
        self.B = np.zeros((x.shape[0], 2))
        self.B[0, 0] = 1.0
        self.B[1, 1] = 1.0
        
        # only the robot positions is affected in prediction step
        # landmark positions do not change!
        self.Q = np.zeros(self.P.shape)
        self.Q[0, 0] = 0.1
        self.Q[1, 1] = 0.1
        
        self.H = np.zeros((len(x) - 2, len(x)))
        for r in range(0, self.H.shape[0], 2):
            self.H[r, 0] = -1
            self.H[r + 1, 1] = -1
        self.H[:, 2:] = np.eye(self.H.shape[0])
        self.R = np.eye(self.H.shape[0]) * 0.1
        
    def predict(self, u):
        self.x = self.F.dot(self.x) + self.B.dot(u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        
    def update(self, landmark_mask, landmark_pos):
        z = landmark_pos.reshape((-1, 1))
        
        lm = landmark_mask.copy()
        lm = np.hstack((lm, lm)).reshape((-1,))
        H = self.H[lm, :]
        
        y = z - H.dot(self.x)

        S = self.H.dot(self.P).dot(self.H.T) + self.R
        
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # hmmm??
        yy = np.zeros((self.H.shape[0], 1))
        yy[lm] = y
        
        self.x = self.x + K.dot(yy)
        self.P = (np.eye(self.P.shape[0]) - K.dot(self.H)).dot(self.P)
        
    def true_measurement(self):
        ts = np.repeat(self.true_state[:2], (self.x.shape[0] - 2) / 2, 1)
        tm = self.true_state[2:].reshape((2, -1), order='F') - ts
        return tm.reshape((-1, 1), order='F')


if __name__ == '__main__':
    mask = np.zeros((10, 10))
    world = Grid(mask, [0,0], [10,10])
    
    np.random.seed(0)
    
    # Landmarks in world space
    landmarks = np.random.uniform(0.0, 100.0, [2, 100])

    # Virtual x,y,phi robot
    robot = XYRobot(pose=[50,50,np.pi / 2.0], err=0.) 
    world['robot'] = robot  

    # Virtual sensor reporting bearings in robot space. Detectable landmarks are limited by FOV, max-dist and obstacles
    sensor = LandmarkSensor(landmarks, err=0.01, fov=1.8 * math.pi, maxdist=10., measure='bearing', environment=world)
    world['robot']['sensor'] = sensor

    drawer = Drawer()
    fig, ax = plt.subplots()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_aspect('equal')
    ax.grid()
    
    true_state = np.vstack((np.array([robot.pose[:2]]).T, landmarks.reshape((-1, 1))))
    # we know nothing but the first landmark!
    x = np.zeros(true_state.shape)
    x[2:4] = true_state[2:4]
    P = np.eye(x.shape[0])
    P[2, 2] = 0.0
    P[3, 3] = 0.0
    k = Kalman(true_state, x, P)
    
    drawer.draw_grid(world, ax, alpha=0.5)
    
    def movement():
        '''
        Robot tries to follow a rectangle
        '''
        u = np.zeros((2, 1))
        v = 1.0
        def f(x):
            return max(-v, min(v, x))
        while True:
            while robot.pose[0] < 90:
                u[0] = f(90 - robot.pose[0])
                u[1] = f(10 - robot.pose[1])
                yield u
            while robot.pose[1] < 90:
                u[0] = f(90 - robot.pose[0])
                u[1] = f(90 - robot.pose[1])
                yield u
            while robot.pose[0] > 10:
                u[0] = f(10 - robot.pose[0])
                u[1] = f(90 - robot.pose[1])
                yield u
            while robot.pose[1] > 10:
                u[0] = f(10 - robot.pose[0])
                u[1] = f(10 - robot.pose[1])
                yield u
    movement_gen = movement()
    
    def update(i):
        m = next(movement_gen)
        robot.move(m)
        
        k.predict(m)
        
        landmark_mask, landmark_pos = sensor.sense(measure='position')
        landmark_pos = landmark_pos.T[landmark_mask].T
        #landmark_indices = np.where(landmark_mask)[0]
        if np.sum(landmark_mask):
            k.update(landmark_mask, landmark_pos)        

        # First sensor
        mask, _ = sensor.sense()
        colors = ['g' if m else 'b' for m in mask]
        u = drawer.draw_robot(robot, ax, radius=0.5)        
        u += drawer.draw_sensor(sensor, ax)        
        u += drawer.draw_points(landmarks, ax, fc=colors)
        
        landmarks2 = k.x[2:].reshape((2, -1))
        
        u += drawer.draw_points(landmarks2, ax, fc='r')

        return u

    ani = animation.FuncAnimation(fig, update, 25, interval=30, blit=True)
    plt.show()


