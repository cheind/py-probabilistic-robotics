import numpy as np
import math
from pytest import approx

from robots import robot

def test_exact_move():

    r = robot.default_state()
    np.testing.assert_allclose(r, 0.)

    robot.imove(r, [0., 1])
    np.testing.assert_allclose(r, [1, 0., 0.])

    robot.imove(r, [math.pi/2, 0])
    np.testing.assert_allclose(r, [1, 0., math.pi/2])

    robot.imove(r, [0.0, 1])
    np.testing.assert_allclose(r, [1, 1., math.pi/2])

    robot.imove(r, [math.pi/2, 1])
    np.testing.assert_allclose(r, [0, 1., math.pi])

def test_noisy_move():

    locs = []
    for i in range(100):
        r = robot.default_state()
        robot.imove(r, [0., 10], err_motion=[1, 0.1])    
        locs.append(r[0])
    
    mu = np.mean(locs)
    std = np.std(locs)
    
    assert mu == approx(10, abs=0.5)
    assert std == approx(1, abs=1e-1)

def test_robot_transforms():    
    r = np.array([10, 0, math.pi/2])

    k = robot.robot_to_world(r)
    np.testing.assert_allclose(
        k,
        np.array([
            [0, -1, 10],
            [1, 0, 0]
        ]),
        atol=1e-4
    )

    k = robot.world_to_robot(r)

    m = np.eye(3)
    m[0:2,0:2] = [[0, -1], [1, 0]]
    m[0:2,2] = [10, 0]
    m = np.linalg.inv(m)

    np.testing.assert_allclose(
        k,
        m[0:2, :],
        atol=1e-4
    )
