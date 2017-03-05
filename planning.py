import numpy as np
import matplotlib.pyplot as plt

from robots.draw import Drawer
from robots.grid import Grid
from robots.planning.gridgraph import GridGraph
from robots.planning.astar import astar

submask = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 1, 1],
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
goal = (6, 2)

# The Gridgraph provides movements on a grid like structure. A* does not make any
# assumption about topology of movements.
graph = GridGraph(grid.values, cost, heuristic, connectivity=GridGraph.FourN)
path, c, explored = astar(start, goal, graph, return_explored=True)

if path:
    print('Costs are {}'.format(c))

    fig, ax = plt.subplots()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_aspect('equal')

    d = Drawer()

    for c in explored:
        grid.values[c[1], c[0]] = 0.5

    d.draw_grid(grid, ax, zorder=1)
    d.draw_points(np.asarray([start]).T + 0.5, ax, fc='r')
    d.draw_points(np.asarray([goal]).T + 0.5, ax, fc='g')
    lines = np.asarray(path).T + 0.5
    d.draw_lines(lines.reshape(1,2,-1), ax)
    ax.invert_yaxis()
    plt.grid()
    plt.show()
else:
    print('No path found.')