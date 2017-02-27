

import numpy as np
from robots.robots import XYPhiRobot
from robots.grid import Grid
from robots.sensors import LandmarkSensor, LidarSensor
from robots.bbox import BBox
from robots.draw import Drawer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


if __name__ == '__main__':

    # Define the collision map
    bbox = BBox([0,0], [10,10])
    mask = np.zeros((10, 10))
    mask[:, -1] = 1.
    mask[:, 0] = 1. 
    mask[6, 5] = 1.  
    grid = Grid(mask, bbox)

    # Landmarks in world space
    landmarks = np.array([
        [3, 5, 6],
        [3, 8, 1]
    ], dtype=float)

    # Virtual sensor reporting bearings in robot space. Detectable landmarks are limited by FOV, max-dist and obstacles
    sensor = LandmarkSensor(landmarks, err=0.01, fov=math.pi/4, maxdist=5., measure='bearing', environment=grid)

    # Virtual x,y,phi robot
    robot = XYPhiRobot(state=[-1,4,0], err=[0., 0.])   

    drawer = Drawer()
    fig, ax = plt.subplots()
    ax.set_xlim([-5, 15])
    ax.set_ylim([-5, 15])
    ax.set_aspect('equal')
    ax.grid()

    drawer.draw_grid(grid, ax, alpha=0.5)
    drawer.draw_points(landmarks, ax, key='landmarks')

    lidar = LidarSensor(grid, fov=math.pi/4, maxdist=5., frame='world')

    def update(i):
        robot.move([0.02, 0.1])
        mask, bearings = sensor.sense(robot)
        colors = ['g' if m else 'b' for m in mask] # Visible landmarks are colored green

        u = drawer.draw_robot(robot, ax, key='robot', radius=0.5)
        u += drawer.draw_sensor(robot, sensor, ax, key='sensor')
        u += drawer.draw_points(landmarks, ax, fc=colors, key='landmarks')

        mask, points = lidar.sense(robot)
        points = points[:, np.where(mask)[0]]
        u += drawer.draw_points(points, ax, marker='o', key='lidar')

        ret, cell = grid.intersect_with_circle(robot.state[:2], 0.5)
        if ret:
            print('robot collision')

        return u

    ani = animation.FuncAnimation(fig, update, 25, interval=30, blit=True)
    plt.show()


