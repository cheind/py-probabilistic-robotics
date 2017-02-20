"""Localize a mobile robot in a cyclic grid by probabilistic Markov localization.

https://www.cs.princeton.edu/courses/archive/fall11/cos495/COS495-Lecture14-MarkovLocalization.pdf
"""

import numpy as np
from scipy.stats import norm
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def uniform_prior(grid):
    """Initializes the grid with an uniform prior."""
    n = grid.size
    grid[:] = 1. / n

def predict(grid, u):
    """Perform prediction step using movement command `u`.
    
    Updates the belief of being in a specific state given the movement
    command and previous state. Based on the law of total probability,
    we get for the new state

        P(x_t) =  sum P(x_t|x_t-1, u) * P(x_t-1)
                 x_t-1

    Meaning that for each possible state we need to sum over all the
    probable ways x_t could have been reached from x_t-1 times the
    prior probability of being in x_t-1.

    In signal theory this corresponds to performing a convolution
    of the signal P(x_t-1) and the probabilistic signal u. Note that
    convolution rotates the signal u by 180°, because we are interested
    in the ways x_t could have been reached from x_t-1 (inverse motion).
    """
    grid[:] = ndimage.convolve(grid, u, mode='constant')
 
def correct(grid, z, stddev=1.):
    """Perform correction/measurement step.
    
    Updates the belief state by incorporating a measurement. The measurement
    updated is given by Bayes rule

        P(x_t|z) = n * P(z|x_t) * P(x_t)

    Here n is a normalizer to make P(x_t|z) a PMF given by

        n = sum P(z|xi_t) * P(xi_t)
             i
    """
    n = 0.
    for i in range(grid.shape[0]):
        # Should get a reading of i (measured towards wall at zero), got z 
        alpha = norm.pdf(z, loc=i, scale=stddev) * grid[i]
        grid[i] = alpha
        n += alpha

    grid /= n

u = np.array([
    [0.1, 0.7, 0.2, 0.0, 0.0],  # Move one to the left under uncertainty
    [0,0, 0.0, 0.2, 0.7, 0.1]   # Move one to the right under uncertainty    
])

grid = np.zeros(100)
moves = np.random.randint(0, 2, size=100)

fig, ax = plt.subplots()
im = ax.imshow(grid.reshape(1, -1), interpolation='none', cmap='hot', extent=[0, grid.size, 0, 1], vmin=0, vmax=1, aspect='auto', animated=True)
line, = ax.plot((0, 0), (0, 1), 'r-')

def init():
    global pos

    pos = 50
    uniform_prior(grid)

    im.set_array(grid.reshape(1, -1))
    line.set_xdata((pos,pos))
    return im, line

def update(i):
    global pos

    m = moves[i % len(moves)]
    pos += -1 if m == 0 else 1
    predict(grid, u[m])  

    if i % 20 == 0:
        correct(grid, pos, stddev=2.)

    im.set_array(grid.reshape(1, -1))
    line.set_xdata((pos,pos))
    return im, line

ani = FuncAnimation(fig, update, init_func=init, interval=200, frames=len(moves), repeat=False, blit=True)
plt.show()