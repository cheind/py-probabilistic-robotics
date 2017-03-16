import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from robots.draw import Drawer
from robots.grid import Grid
from robots.planning.gridgraph import GridGraph
from robots.planning.astar import astar
from robots.planning.smooth import smooth_path
from robots.planning.trajectories.quintic import QuinticTrajectory
from robots.planning.trajectories.trapezoidal import ApproximateTrapezoidalTrajectory

submask = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0]
])

mask = np.zeros((10,10))
mask[:5, :6] = submask
grid = Grid(mask, [0,0], [10,10])

def cost(a, b):
    """Cost for moving between a and neighbor b. 
    
    Here we assume that each move costs us 1.
    """
    return 1.

def heuristic(a, goal):
    """An optimistic guess to the goal from node a. 
    
    A star uses this method to priorize nodes to be searched next. The heuristic
    should be an optimistic guess that shall never overestimate the shortest 
    distance. 
    
    Here we use the Manhatten distance based on the available movements of the robot
    N,E,S,W. Distance is weighted by minimal cost per unit move (1. in this case). 
    As an optimistic guess we assume no obstacles in the way.    
    """
    dx = abs(a[0] - goal[0])
    dy = abs(a[1] - goal[1])
    return 1. * (dx + dy) # Here we set d to the lowest possible cost for each move

# A* algorithms assumes hashable nodes, so we use tuples instead of np.ndarray
start = (0, 0) # x,y not row,col
goal = (9, 1)

def draw_path(drawer, ax, path, color):
    lines = np.asarray(path) + 0.5
    drawer.draw_lines(lines.reshape(1,-1, 2), ax, ec=color)

# The Gridgraph provides movements on a grid like structure. A* does not make any
# assumption about topology of movements.
graph = GridGraph(grid.values, cost, heuristic, connectivity=GridGraph.FourN)
path, c, explored = astar(start, goal, graph, return_explored=True)

"""
if path:    
    print('Costs are {}'.format(c))

    fig, ax = plt.subplots()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_aspect('equal')

    d = Drawer()

    for i, c in enumerate(explored):
        grid.values[c[1], c[0]] = 1. / (i + 2)

    d.draw_grid(grid, ax, zorder=1)
    d.draw_points(np.asarray([start]).T + 0.5, ax, fc='r')
    d.draw_points(np.asarray([goal]).T + 0.5, ax, fc='g')

    draw_path(d, ax, path, 'k')
    
    spath = smooth_path(path, 0.5)
    draw_path(d, ax, spath, 'y')

    traj = PolynomialTrajectory(spath, [2]*(len(path)-1), [0,0], [0,0])
    t = np.arange(0, traj.total_time, 0.1)
    x, dx, ddx = traj(t)
    draw_path(d, ax, x, 'g')
    
    ax.invert_yaxis()
    plt.grid()
    plt.show()
    
else:
    print('No path found.')
"""

fig, ax = plt.subplots()
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 11, 1))
ax.set_aspect('equal')
ax.invert_yaxis()
d = Drawer()

d.draw_grid(grid, ax, zorder=1)
d.draw_points(np.asarray([start]) + 0.5, ax, fc='r')
d.draw_points(np.asarray([goal]) + 0.5, ax, fc='g')
d.draw_points(np.asarray([start]) + 0.5, ax, fc='r', marker='o', key='loc')

#traj = QuinticTrajectory(path, [0]*len(path), [0]*len(path), np.linspace(0, 10, len(path)))
traj = ApproximateTrapezoidalTrajectory(path, [0.5]*(len(path)-1), 6)
t = np.linspace(0, traj.total_time, 500)
x, dx, ddx = traj(t)
draw_path(d, ax, x, 'g')

def update(i):
    t = i / 30.
    x, dx, ddx = traj(t)
    return d.draw_points(x + 0.5, ax, fc='r', marker='o', key='loc')

ani = animation.FuncAnimation(fig, update, int(traj.t[-1] * 30), interval=30, blit=True, repeat=True)


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Position')
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Velocity')
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Acceleration')
ax1.plot(t, x[:, 0], label='x')
ax1.plot(t, x[:, 1], label='y')
ax2.plot(t, dx[:, 0], label='dx/dt')
ax2.plot(t, dx[:, 1], label='dy/dt')
ax3.plot(t, ddx[:, 0], label='ddx/ddt')
ax3.plot(t, ddx[:, 1], label='ddx/ddt')
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper left')
plt.tight_layout()

plt.show()





"""
## Trying differnt settings of smoothness

path = np.array([
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 3],
    [2, 3],
    [3, 3],
    [4, 3],
    [4, 4],
    [4, 5],
    [4, 6]
])

fig, ax = plt.subplots()
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 11, 1))
ax.set_aspect('equal')

d = Drawer()
draw_path(d, ax, path, 'k')
draw_path(d, ax, smooth_path(path, smoothweight=0.1), 'g')
draw_path(d, ax, smooth_path(path, smoothweight=0.5), 'r')
draw_path(d, ax, smooth_path(path, smoothweight=1.0), 'b')
draw_path(d, ax, smooth_path(path, smoothweight=10.0), 'y')


ax.invert_yaxis()
plt.grid()
plt.show()
"""