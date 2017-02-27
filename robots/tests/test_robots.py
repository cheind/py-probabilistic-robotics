import numpy as np
import math
from pytest import approx

from robots.robots import XYPhiRobot

def test_robot_transforms():   
    r = XYPhiRobot(state=[10, 0, math.pi/2]) 

    k = r.robot_in_world()
    np.testing.assert_allclose(
        k,
        np.array([
            [0, -1, 10],
            [1, 0, 0],
            [0, 0, 1],
        ]),
        atol=1e-4
    )

    k = r.world_in_robot()

    m = np.eye(3)
    m[0:2,0:2] = [[0, -1], [1, 0]]
    m[0:2,2] = [10, 0]
    m = np.linalg.inv(m)

    np.testing.assert_allclose(
        k,
        m,
        atol=1e-4
    )

def test_xyphi_pose():
     r = XYPhiRobot(state=[10, 0, math.pi/2]) 
     np.testing.assert_allclose(r.pose, [10, 0, math.pi/2])

def test_xyphi_exact_move():

    r = XYPhiRobot()
    np.testing.assert_allclose(r.state, 0.)

    r.move([0., 1])
    np.testing.assert_allclose(r.state, [1, 0., 0.])

    r.move([math.pi/2, 0])
    np.testing.assert_allclose(r.state, [1, 0., math.pi/2])

    r.move([0.0, 1])
    np.testing.assert_allclose(r.state, [1, 1., math.pi/2])

    r.move([math.pi/2, 1])
    np.testing.assert_allclose(r.state, [0, 1., math.pi])

def test_xyphi_noisy_move():

    locs = []
    for i in range(100):
        r = XYPhiRobot(err=[1, 0.1])
        r.move([0., 10])
        locs.append(r.state[0])
    
    mu = np.mean(locs)
    std = np.std(locs)
    
    assert mu == approx(10, abs=0.5)
    assert std == approx(1, abs=0.5)


