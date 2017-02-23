
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def default_state():
    return np.array([0.,0.,0.])

def move(state, motion, err_motion=[0., 0.], inplace=False):
    """Move robot by first turning and then driving along the robot's x-axis."""
    e_phi = 0. if err_motion[0] == 0. else np.random.normal(scale=err_motion[0])
    e_delta = 0. if err_motion[1] == 0. else np.random.normal(scale=err_motion[1])

    if inplace:
        state[2] += motion[0] + e_phi
        state[0] += math.cos(state[2]) * (motion[1] + e_delta)
        state[1] += math.sin(state[2]) * (motion[1] + e_delta)
        return state
    else:
        phi = state[2] + motion[0] + e_phi
        return np.array([
            state[0] + math.cos(phi) * (motion[1] + e_delta),
            state[1] + math.sin(phi) * (motion[1] + e_delta),
            phi
        ])

    # Angle wrap
    # math.atan2(math.sin(a), math.cos(a)) -> -pi..pi
    # math.atan2(math.sin(a-math.pi), math.cos(a-math.pi)) + math.pi -> 0..2pi

def robot_to_world(state):
    """Returns a 2x3 matrix representing the robot's location in world space."""
    c = math.cos(state[2])
    s = math.sin(state[2])

    return np.array([
        [c, -s, state[0]],
        [s, c, state[1]]
    ])

def world_to_robot(state):
    """Returns a 2x3 matrix representing the world in robot space."""
    c = math.cos(state[2])
    s = math.sin(state[2])

    return np.array([
        [c, s, -(c * state[0] + s * state[1])],
        [-s, c, -(-s * state[0] + c * state[1])]
    ])

def draw(state, ax, radius=0.5, fc='None', ec='k'):

    m = robot_to_world(state)
    o = np.dot(m, [0,0,1])
    x = np.dot(m, [radius,0,1])
    y = np.dot(m, [0,radius,1])

    dx = math.cos(state[2]) * radius
    dy = math.sin(state[2]) * radius

    c = plt.Circle(state[:2], radius=radius, fc=fc, ec=ec)
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
    move(r, [0.05, 0.1], err_motion=[0.001, 0.01], inplace=True)
    
    e = []
    e.extend(draw(r, ax, fc='b', ec='None'))
    return e


ani = animation.FuncAnimation(fig, update, 25, interval=50, blit=True)
plt.show()


