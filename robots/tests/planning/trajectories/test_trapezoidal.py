import numpy as np

from robots.planning.trajectories.trapezoidal import TrapezoidalTrajectory, ApproximateTrapezoidalTrajectory

import matplotlib.pyplot as plt
from pytest import approx

def plot_traj(t, x, dx, ddx, i=0):
    plt.plot(t, x[:, i])
    plt.plot(t, dx[:, i])
    plt.plot(t, ddx[:, i])

def test_trapezoidal():

    # Zero velocities at sample points
    # No coasting as velocity of 20 is never reached

    traj = TrapezoidalTrajectory([0, 20, 50, 40, 10], dq_max=15, ddq_max=20)
    x, dx, ddx = traj(traj.t)
    np.testing.assert_allclose(x[:, 0], [0, 20, 50, 40, 10])
    np.testing.assert_allclose(dx[:, 0], [0, 15, 0, -15, 0], atol=1e-4)
    np.testing.assert_allclose(ddx[:, 0], [20, 0, -20, 0, 20])

    traj = TrapezoidalTrajectory([[0,0],[0,20]], dq_max=15, ddq_max=20)    
    x, dx, ddx = traj(traj.t)    
    np.testing.assert_allclose(x[:, 0], [0, 0])
    np.testing.assert_allclose(x[:, 1], [0, 20])
    np.testing.assert_allclose(dx[:, 0], [0, 0], atol=1e-4)
    np.testing.assert_allclose(dx[:, 1], [0, 0], atol=1e-4)
    np.testing.assert_allclose(ddx[:, 0], [0, 0])
    np.testing.assert_allclose(ddx[:, 1], [20, -20])

    traj = TrapezoidalTrajectory([0, 10, 20], ddq_max=2, dq=[0, 4, 0])  
    x, dx, ddx = traj(traj.t)

    #t = np.linspace(0, 20, 100)
    #plot_traj(t, *traj(t))
    #plt.show()

    np.testing.assert_allclose(x[:, 0], [0, 10, 20])
    np.testing.assert_allclose(dx[:, 0], [0, 4, 0])
    np.testing.assert_allclose(ddx[:, 0], [2, 0, -2])


def test_approximate_trapezoidal():

    traj = ApproximateTrapezoidalTrajectory([0, 20, 0], ddq_max=10, dt=[2,2])
    x, dx, ddx = traj(traj.t)
    np.testing.assert_allclose(x[:, 0], [1.25, 15, 1.25])
    np.testing.assert_allclose(dx[:, 0], [5, 0, -5])
    np.testing.assert_allclose(ddx[:, 0], [10, -10, 10])
    
    traj = ApproximateTrapezoidalTrajectory([0, 20, 0], ddq_max=100, dt=[2,2])
    x, dx, ddx = traj(traj.t)
    
    np.testing.assert_allclose(x[:, 0], [0.125, 19.5, 0.125])
    np.testing.assert_allclose(dx[:, 0], [5, 0, -5])
    np.testing.assert_allclose(ddx[:, 0], [100, -100, 100])

    #t = np.linspace(0, traj.total_time, 100)
    #plot_traj(t, *traj(t))
    #plt.scatter(traj.t, [0, 20, 0])
    #plt.show()