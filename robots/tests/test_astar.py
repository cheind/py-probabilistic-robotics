
import numpy as np

from robots.grid import Grid
from robots.planning.astar import GridGraph
from robots.planning.astar import astar

def test_gridgraph():

    mask = np.zeros((10,10))
    mask[5, 5] = 1.
    grid = Grid(mask, [0,0], [10,10])
    graph = GridGraph(grid, GridGraph.FourN)

    np.testing.assert_allclose(graph.neighbors([0,0]), [[1, 0], [0, 1]])
    np.testing.assert_allclose(graph.neighbors([9,9]), [[9, 8], [8, 9]])
    np.testing.assert_allclose(graph.neighbors([5,4]), [[5, 3], [6,4], [4,4]])

    assert graph.cost((0,0), (0,1)) == 1.

def test_astar():

    mask = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0]
    ])
    grid = Grid(mask, [0,0], [6,5])
    graph = GridGraph(grid, GridGraph.FourN)

    def heuristic(node, goal):
        return 0.

    path = astar((0, 0), (5, 3), graph, heuristic)
    print(path)