import numpy as np

from robots.planning.trajectories.quintic import QuinticTrajectory

import matplotlib.pyplot as plt
from pytest import approx


def test_with_accelerations():
    traj = QuinticTrajectory(q=[10, 20, 0, 30, 40], dq=[0, 0, 0, 0, 0], ddq=[0, 0, 0, 0, 0], qt=[0, 2, 4, 8, 10])

    t = np.linspace(0, 10, 1000)
    x, dx, ddx = traj([0, 2, 4, 8, 10])

    np.testing.assert_allclose(x[:,0], [10, 20, 0, 30, 40])
    np.testing.assert_allclose(dx[:,0], [0, 0, 0, 0, 0], atol=1e-4)
    np.testing.assert_allclose(ddx[:,0], [0, 0, 0, 0, 0], atol=1e-4)
