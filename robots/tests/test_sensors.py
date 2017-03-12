import numpy as np
import math
from pytest import approx

from robots.robots import XYPhiRobot
from robots.sensors import LandmarkSensor, LidarSensor
from robots.grid import Grid
from robots.bbox import BBox
from robots.posenode import PoseNode

def test_landmark_sensor():

    r = XYPhiRobot(pose=[5, 5, 0])

    landmarks = np.array([
        [0, 0],
        [3, 6],
        [-2, 5]
    ])
    s = LandmarkSensor(landmarks, parent=r)

    mask, obs = s.sense(measure='position')
    np.testing.assert_allclose(obs, np.array([[-5, -5],[-2, 1],[-7, 0]]))
    assert mask.all()

    mask, obs = s.sense(measure='bearing')
    np.testing.assert_allclose(obs, np.array([math.atan2(-5,-5), math.atan2(1,-2), math.atan2(0,-7)]))
    assert mask.all()

    mask, obs = s.sense(measure='distance')
    np.testing.assert_allclose(obs, np.array([math.sqrt(5**2+5**2),math.sqrt(1**2+2**2), math.sqrt(7**2)]))
    assert mask.all()

def test_lidar_sensor():
    mask = np.zeros((10, 10))
    mask[:, -1] = 1. 
    grid = Grid(mask, [0,0], [10,10])

    r = XYPhiRobot(pose=[5, 5, 0])

    s = LidarSensor(grid, fov=math.pi/4, parent=r)

    mask, points = s.sense()
    assert mask.all()
    np.testing.assert_allclose(points[:, 0], 4)

    r.pose = [5, 5, math.pi]
    mask, points = s.sense()
    assert np.logical_not(mask).all()


def test_landmark_sensor_regression_raytrace():

    mask = np.zeros((10, 10))
    world = Grid(mask, [0,0], [10,10])
    r = XYPhiRobot(pose=[10, 10, 0], parent=world)

    np.random.seed(0)
    landmarks = np.hstack((
        np.random.uniform(0.0, 100.0, [100, 1]),
        np.random.uniform(0.0, 20.0, [100, 1]))) 
   
    s = LandmarkSensor(landmarks, err=0.0, fov=2.0 * math.pi, maxdist=200, measure='position', environment=world, parent=r)

    mask, points = s.sense()
    assert mask.all()
    

