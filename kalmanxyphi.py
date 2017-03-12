

import numpy as np
from robots.posenode import PoseNode
from robots.robots import XYRobot, XYPhiRobot
from robots.grid import Grid
from robots.sensors import LandmarkSensor, LidarSensor
from robots.bbox import BBox
from robots.draw import Drawer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

np.set_printoptions(linewidth=180)

class Kalman:
    def __init__(self, true_state, x, P):
        self.true_state = true_state.copy()    
        self.x = x.copy()
        self.P = P.copy()
        # self.landmarks_seen = np.zeros(((x.shape[0] - 2) // 2, 1), dtype=np.bool)
        # self.landmarks_seen[0] = True
        
        self.F = np.eye(x.shape[0])
        self.B = np.zeros((x.shape[0], 2))
        self.B[0, 0] = 1.0
        self.B[1, 1] = 1.0
        
        # only the robot positions is affected in prediction step
        # landmark positions do not change!
        self.Q = np.zeros(self.P.shape)
        self.Q[0, 0] = 0.1
        self.Q[1, 1] = 0.1
        
        # self.H = np.zeros((len(x) - 2, len(x)))
        # for r in range(0, self.H.shape[0], 2):
        #     self.H[r, 0] = -1
        #     self.H[r + 1, 1] = -1
        # self.H[:, 2:] = np.eye(self.H.shape[0])
        # self.R = np.eye(self.H.shape[0]) * 0.1
        
    def predict(self, u):
        self.x[2] += u[0]   # simply update phi
        self.x[0] += math.cos(self.x[2]) * u[1]
        self.x[1] += math.sin(self.x[2]) * u[1]

        self.F[2] = [0.0, 0.0, 1.0] # constant
        self.F[0] = [1.0, 0.0, -math.sin(self.x[2]) * u[1]]
        self.F[1] = [0.0, 1.0, math.cos(self.x[2]) * u[1]]

        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    # def update(self, landmark_mask, landmark_pos):
    #     def to_state_mask(a):
    #         return np.hstack((a, a)).reshape(-1)

    #     z = landmark_pos.reshape((-1, 1), order='F')
    #     z_mask = to_state_mask(~self.landmarks_seen[landmark_mask])
    #     n = np.sum(z_mask)
    #     if n:
    #         print('init {}'.format(n))
    #         pos = np.repeat(self.x[:2], n // 2, axis=0)
    #         init_mask = np.logical_and(~self.landmarks_seen, np.expand_dims(landmark_mask, 1))
    #         init_mask_state = np.hstack(([False, False], to_state_mask(init_mask)))
    #         self.x[init_mask_state] = pos + z[z_mask]
    #         self.landmarks_seen = np.logical_or(self.landmarks_seen, init_mask)

    #     lm = landmark_mask.copy()
    #     lm = np.vstack((lm, lm)).reshape(-1, order='F')
               
    #     H = self.H[lm, :]
    #     y = z - H.dot(self.x)
        
    #     S = H.dot(self.P).dot(H.T) + self.R[lm][:, lm]

    #     K = self.P.dot(H.T).dot(np.linalg.inv(S))
        
    #     self.x = self.x + K.dot(y)
    #     self.P = (np.eye(self.P.shape[0]) - K.dot(H)).dot(self.P)
        
    # def true_measurement(self):
    #     ts = np.repeat(self.true_state[:2], (self.x.shape[0] - 2) / 2, 1)
    #     tm = self.true_state[2:].reshape((2, -1), order='F') - ts
    #     return tm.reshape((-1, 1), order='F')


if __name__ == '__main__':
    mask = np.zeros((10, 10))
    mask[5:, 5] = True
    world = Grid(mask, [-20,-20], [120,120])
    
    np.random.seed(0)
    
    # Landmarks in world space
    landmarks = np.hstack((
        np.random.uniform(-20.0, 120.0, [50, 1]),
        np.random.uniform(-20.0, 120.0, [50, 1])))

    # Virtual x,y,phi robot
    robot = XYPhiRobot(pose=[50, 30,0], err=[0.5, 0.1])

    true_state = robot.pose
    kalman = Kalman(true_state, true_state, np.eye(3) * 0.01)

    # robot starts exactly at landmark
    #landmarks[:, 0] = robot.transform_to_world[:2, 2]

    # Virtual sensor reporting bearings in robot space. Detectable landmarks are limited by FOV, max-dist and obstacles
    sensor = LandmarkSensor(landmarks, pose=[0,0,0], err=0.1, fov=math.pi, maxdist=20., measure='position', environment=world, parent=robot)

    drawer = Drawer()
    fig, ax = plt.subplots()
    ax.set_xlim([-50, 150])
    ax.set_ylim([-50, 150])
    ax.set_aspect('equal')
    ax.grid()
    
    #true_state = np.vstack((np.array([robot.pose[:2]]).T, landmarks.reshape((-1, 1), order='F')))
        
    # we know nothing but the first landmark!
    #x = np.zeros(true_state.shape)
    #x[2:4] = true_state[2:4]
    #P = np.eye(true_state.shape[0])
    #P[2, 2] = 0.0
    #P[3, 3] = 0.0
    #k = Kalman(true_state, x, P)
    
    drawer.draw_grid(world, ax, alpha=0.5)
    
    def movement():
        '''
        Robot tries to follow a rectangle
        '''
        def norm(phi):
            return np.arctan2(np.sin(phi), np.cos(phi))
        def target():
            return landmarks[np.random.randint(landmarks.shape[0])]
        while True:
            t = target()
            print('new target ', t)
            while True:
                tr = robot.pose
                v = t - tr[:2]
                if np.linalg.norm(v) <= 1.0:
                    break
                phir = tr[2]
                phit = np.arctan2(v[1], v[0])
                dphi = phit - phir
                u = np.array([dphi, 1.0])
                yield u
    movement_gen = movement()
    
    def update(i):
        if i == 0:
            return []
        m = next(movement_gen)
        robot.move(m)
        kalman.predict(m)
        
        landmark_mask, landmark_pos = sensor.sense()
        landmark_pos = landmark_pos[landmark_mask]
        #landmark_indices = np.where(landmark_mask)[0]
        #if np.sum(landmark_mask) > 2:
        #    k.update(landmark_mask, landmark_pos)

        # First sensor
        colors = ['g' if m else 'b' for m in landmark_mask]
        u = drawer.draw_robot(robot, ax, radius=1.5)        
        u += drawer.draw_sensor(sensor, ax)        
        u += drawer.draw_points(landmarks, ax, fc=colors)
        u += drawer.draw_confidence_ellipses([robot.pose[:2]], [kalman.P[:2,:2]], ax, key='conf', scale=40)
        #u += drawer.draw_confidence_ellipses([landmarks[:, 1], robot.pose[:2]], [k.P[4:6,4:6], k.P[:2,:2]], ax, key='conf', scale=40)
        
        #landmarks2 = k.x[2:].reshape((2, -1), order='F')
        #u += drawer.draw_points(landmarks2, ax, fc='r', zorder=10, marker='.', key='landmarks2')

        return u

    ani = animation.FuncAnimation(fig, update, 200, interval=30, blit=True, repeat=True)
    plt.show()

