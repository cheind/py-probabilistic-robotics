

import numpy as np
from robots.posenode import PoseNode
from robots.robots import XYPhiRobot
from robots.grid import Grid
from robots.sensors import LandmarkSensor, LidarSensor
from robots.bbox import BBox
from robots.draw import Drawer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


if __name__ == '__main__':
    mask = np.zeros((10, 10))
    mask[:, -1] = 1.
    mask[:, 0] = 1. 
    mask[6, 5] = 1.    
    world = Grid(mask, [0,0], [10,10])

    
    # Landmarks in world space
    landmarks = np.array([
        [3, 3],
        [5, 8],
        [6, 1]
    ], dtype=float)

    # Virtual x,y,phi robot
    robot = XYPhiRobot(pose=[-1,4,0], err=[0., 0.], parent=world) 
    
    # Virtual sensor reporting bearings in robot space. Detectable landmarks are limited by FOV, max-dist and obstacles
    sensor = LandmarkSensor(landmarks, err=0.01, fov=math.pi/4, maxdist=5., measure='bearing', environment=world, parent=robot)
    lidar = LidarSensor(world, fov=math.pi/4, maxdist=5., pose=[0,0,math.pi], parent=robot)

    drawer = Drawer()
    fig, ax = plt.subplots()
    ax.set_xlim([-5, 15])
    ax.set_ylim([-5, 15])
    ax.set_aspect('equal')
    ax.grid()

    def init():
        return drawer.draw_grid(world, ax, alpha=0.5)

    def update(i):
        robot.move([0.02, 0.1])

        # First sensor
        mask, bearings = sensor.sense()
        colors = ['g' if m else 'b' for m in mask]

        u = []
        u += drawer.draw_robot(robot, ax, radius=0.5)        
        u += drawer.draw_sensor(sensor, ax)        
        u += drawer.draw_points(landmarks, ax, fc=colors)
        
        # Second sensor
        mask, points = lidar.sense()        
        points = points[np.where(mask)[0]]
        
        u += drawer.draw_points(points, ax, size=5, marker='o', transform=lidar.transform_to_world, key='lidar') # need to use drawing key as points is always a new object
        
        u += drawer.draw_sensor(lidar, ax, fc='green', ec='green')     

        ret, cell = world.intersect_with_circle(robot.pose[:2], 0.5)
        if ret:
            print('robot collision')

        return u

    ani = animation.FuncAnimation(fig, update, frames=10000, interval=30, init_func=init, blit=True)
    plt.show()


