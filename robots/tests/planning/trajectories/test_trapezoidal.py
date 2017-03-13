import numpy as np

from robots.planning.trajectories.trapezoidal import TrapezoidalTrajectory

import matplotlib.pyplot as plt
from pytest import approx

def plot_traj(t, x, dx, ddx):
    fig, ax = plt.subplots()
    plt.plot(t, x[:, 0])
    plt.plot(t, dx[:, 0])
    plt.plot(t, ddx[:, 0])


def test_multi_segment1():

    # Zero velocities at sample points
    # No coasting as velocity of 20 is never reached

    traj = TrapezoidalTrajectory([0, 20, 50, 40, 10], 20, 10)
    x, dx, ddx = traj(traj.t)

    np.testing.assert_allclose(x[:, 0], [0, 20, 50, 40, 10])
    np.testing.assert_allclose(dx[:, 0], 0, atol=1e-4)
    np.testing.assert_allclose(ddx[:, 0], [10, 10, -10, -10, 10])