

import numpy as np
from robots.robots import Robot
from robots.grid import Grid
from robots.sensors import LandmarkSensor
from robots.bbox import BBox
from robots.draw import DefaultDrawer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


if __name__ == '__main__':

    landmarks = np.array([
        [-3, 10, 3],
        [-3, 10, 5]
    ], dtype=float)
    s = LandmarkSensor(landmarks, err=0.01, fov=math.pi/4, maxdist=5., measure='bearing')
    r = Robot(state=[0,0,0])

    bbox = BBox([0,0], [10,10])

    mask = np.zeros((10, 10))
    mask[:, -1] = 1.
    mask[:, 0] = 1.    

    grid = Grid(mask.shape, bbox)
    
    drawer = DefaultDrawer()

    fig, ax = plt.subplots()
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_aspect('equal')
    ax.grid()

    drawer.draw_grid(grid, mask, ax, alpha=0.5)
    drawer.draw_landmarks(landmarks, ax, key='landmarks')


    def update(i):
        r.move([0.05, 0.1])
        mask, bearings = s.sense(r)

        colors = ['g' if m else 'b' for m in mask]

        u = drawer.draw_robot(r, ax, key='robot')
        u += drawer.draw_landmark_sensor(r, s, ax, key='sensor')
        u += drawer.draw_landmarks(landmarks, ax, fc=colors)
        return u

    ani = animation.FuncAnimation(fig, update, 25, interval=30, blit=True)
    plt.show()


