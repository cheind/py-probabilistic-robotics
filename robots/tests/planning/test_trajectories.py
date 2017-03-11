import numpy as np

from robots.planning.trajectories import PolynomialTrajectory
import matplotlib.pyplot as plt
from pytest import approx

def plot_traj(t, x, dx, ddx):
    fig, ax = plt.subplots()
    plt.plot(t, x[:, 0])
    plt.plot(t, dx[:, 0])
    plt.plot(t, ddx[:, 0])
    plt.show()


def test_cubic_linear():

    # Constant position
    traj = PolynomialTrajectory([10, 10], [1], [0, 0])
    t = np.arange(0, 10, 0.1)
    x, dx, ddx = traj(t)
    np.testing.assert_allclose(x[:,0], 10)
    np.testing.assert_allclose(dx[:,0], 0)
    np.testing.assert_allclose(ddx[:,0], 0)

    # Constant velocity
    traj = PolynomialTrajectory([10, 20], [1], [10, 10])
    t = np.arange(0, 10, 0.1)
    x, dx, ddx = traj(t)
    np.testing.assert_allclose(dx[:,0], 10)
    np.testing.assert_allclose(ddx[:,0], 0)

    # Constant acceleration
    traj = PolynomialTrajectory([10, 20], [1], [0, 0], [0, 0])
    t = np.linspace(0, 1, 100)
    x, dx, ddx = traj(t)

    assert x[0, 0] == approx(10)
    assert x[-1, 0] == approx(20)
    assert dx[-1, 0] == approx(0)
    assert dx[-1, 0] == approx(0)
    assert ddx[-1, 0] == approx(0)
    assert ddx[-1, 0] == approx(0)
    
