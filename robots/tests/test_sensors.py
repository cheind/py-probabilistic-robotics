import numpy as np
import math
from pytest import approx

from robots.robots import Robot
from robots.sensors import LandmarkSensor

def test_landmark_sensor():

    r = Robot(state=[5, 5, 0])

    landmarks = np.array([
        [0, 3, -2],
        [0, 6, 5]
    ])
    s = LandmarkSensor(landmarks)

    mask, obs = s.sense(r, measure='position')
    np.testing.assert_allclose(obs, np.array([[-5, -2, -7], [-5, 1, 0]]))
    assert mask.all()

    mask, obs = s.sense(r, measure='bearing')
    np.testing.assert_allclose(obs, np.array([math.atan2(-5,-5), math.atan2(1,-2), math.atan2(0,-7)]))
    assert mask.all()

    mask, obs = s.sense(r, measure='distance')
    np.testing.assert_allclose(obs, np.array([math.sqrt(5**2+5**2),math.sqrt(1**2+2**2), math.sqrt(7**2)]))
    assert mask.all()
