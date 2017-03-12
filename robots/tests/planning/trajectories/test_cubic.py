import numpy as np

from robots.planning.trajectories.cubic import CubicTrajectory

import matplotlib.pyplot as plt
from pytest import approx

def plot_traj(t, x, dx, ddx):
    fig, ax = plt.subplots()
    plt.plot(t, x[:, 0])
    plt.plot(t, dx[:, 0])
    plt.plot(t, ddx[:, 0])
    plt.show()

def test_with_velocities():
    traj = CubicTrajectory([10, 20, 0, 30, 40], [0, -10, 10, 3, 0], [0, 2, 4, 8, 10])

    #t = np.linspace(0, 10, 100)
    #plot_traj(t, *traj(t))
    x, dx, ddx = traj([0, 2, 4, 8, 10])

    np.testing.assert_allclose(x[:,0], [10, 20, 0, 30, 40])
    np.testing.assert_allclose(dx[:,0], [0,-10,10,3,0])
    np.testing.assert_allclose(ddx[:,0], [25., -20., -0.25, 9.,-12.])

def test_intermediate_velocities():
    traj = CubicTrajectory([10, 20, 0, 30, 40], [0, 0], [0, 2, 4, 8, 10])

    x, dx, ddx = traj([0, 2, 4, 8, 10])

    np.testing.assert_allclose(x[:,0], [10, 20, 0, 30, 40])
    np.testing.assert_allclose(dx[:,0], [0,0,0,6.25,0])
    np.testing.assert_allclose(ddx[:,0], [15., -30., 8.125, 2.5, -8.75])


def test_constant_position():
    traj = CubicTrajectory([10, 10], [0, 0], [0, 1])
    t = np.arange(0, 10, 1)
    x, dx, ddx = traj(t)
    np.testing.assert_allclose(x[:,0], 10)
    np.testing.assert_allclose(dx[:,0], 0)
    np.testing.assert_allclose(ddx[:,0], 0)

def test_constant_velocity():
    traj = CubicTrajectory([10, 20], [10, 10], [0, 1])
    t = np.arange(0, 10, 1)
    x, dx, ddx = traj(t)
    np.testing.assert_allclose(dx[:, 0], 10)
    np.testing.assert_allclose(ddx[:, 0], 0)



    """
    # Constant position
    traj = cubic.CubicTrajectory([10, 10], [0, 1], [0, 0])
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
    """
