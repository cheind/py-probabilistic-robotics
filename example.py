

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

    # Define the collision map
    bbox = BBox([0,0], [10,10])
    mask = np.zeros((10, 10))
    mask[:, -1] = 1.
    mask[:, 0] = 1. 
    mask[5, 5] = 1.  
    grid = Grid(mask, bbox)

    # Landmarks in world space
    landmarks = np.array([
        [3, 5, 6],
        [3, 8, 1]
    ], dtype=float)

    # Virtual sensor reporting bearings in robot space. Detectable landmarks are limited by FOV, max-dist and obstacles
    sensor = LandmarkSensor(landmarks, err=0.01, fov=math.pi/4, maxdist=5., measure='bearing', obstacles=grid)

    # Virtual x,y,phi robot
    robot = Robot(state=[-1,4,0])


    drawer = DefaultDrawer()
    fig, ax = plt.subplots()
    ax.set_xlim([-5, 15])
    ax.set_ylim([-5, 15])
    ax.set_aspect('equal')
    ax.grid()

    drawer.draw_grid(grid, ax, alpha=0.5)
    drawer.draw_landmarks(landmarks, ax, key='landmarks')

    def update(i):
        robot.move([0.02, 0.1])
        mask, bearings = sensor.sense(robot)
        colors = ['g' if m else 'b' for m in mask] # Visible landmarks are colored green

        u = drawer.draw_robot(robot, ax, key='robot', radius=0.5)
        u += drawer.draw_landmark_sensor(robot, sensor, ax, key='sensor')
        u += drawer.draw_landmarks(landmarks, ax, fc=colors, key='landmarks')

        ret, cell = grid.intersect_with_circle(robot.state[:2], 0.5)
        if ret:
            print('ups')

        return u

    ani = animation.FuncAnimation(fig, update, 25, interval=30, blit=True)
    plt.show()


