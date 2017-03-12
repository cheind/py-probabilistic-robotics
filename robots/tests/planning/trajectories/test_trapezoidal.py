import numpy as np

from robots.planning.trajectories.trapezoidal import TrapezoidalTrajectory

import matplotlib.pyplot as plt
from pytest import approx

def plot_traj(t, x, dx, ddx):
    fig, ax = plt.subplots()
    plt.plot(t, x[:, 0])
    plt.plot(t, dx[:, 0])
    plt.plot(t, ddx[:, 0])
    plt.show()

def test_single_segment():
    traj = TrapezoidalTrajectory([30, 10], 10, 10)
    t = np.linspace(0, traj.t[-1], 100)
    plot_traj(t, *traj(t))

def test_multi_segment():
    traj = TrapezoidalTrajectory([10, 20, 50, 40], 10, 10)
    t = np.linspace(0, traj.t[-1], 100)
    plot_traj(t, *traj(t))