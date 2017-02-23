
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def default_state():
    return np.array([0.,0.,0.])

def move(state, params):
    """Move robot by first turning and then driving along the robot's x-axis."""
    phi = state[2] + params[0]
    return np.array([
        state[0] + math.cos(phi) * params[1],
        state[1] + math.sin(phi) * params[1],
        phi
    ])

    # Angle wrap
    # math.atan2(math.sin(a), math.cos(a)) -> -pi..pi
    # math.atan2(math.sin(a-math.pi), math.cos(a-math.pi)) + math.pi -> 0..2pi

def robot_to_world(state):
    c = math.cos(state[2])
    s = math.sin(state[2])

    return np.array([
        [c, -s, state[0]],
        [s, c, state[1]]
    ])

def world_to_robot(state):
    c = math.cos(state[2])
    s = math.sin(state[2])

    return np.array([
        [c, s, -(c * state[0] + s * state[1])],
        [-s, c, -(-s * state[0] + c * state[1])]
    ])

def draw(state, ax, radius=0.5):

    m = robot_to_world(state)
    o = np.dot(m, [0,0,1])
    x = np.dot(m, [radius,0,1])
    y = np.dot(m, [0,radius,1])

    dx = math.cos(state[2]) * radius
    dy = math.sin(state[2]) * radius

    c = plt.Circle(state[:2], radius=radius, fc='none', ec='k')
    ax.add_artist(c)

    lx, = ax.plot([o[0], x[0]], [o[1], x[1]], color='r')
    ly, = ax.plot([o[0], y[0]], [o[1], y[1]], color='g')

    return c, lx, ly

fig, ax = plt.subplots()
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_aspect('equal')

r = np.zeros(3)

def update(i):    
    global r
    r = move(r, [0.05, 0.1])
    
    e = []
    e.extend(draw(r, ax))
    return e


ani = animation.FuncAnimation(fig, update, 25, interval=50, blit=True)
plt.show()


