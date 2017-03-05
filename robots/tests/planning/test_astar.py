
import numpy as np

from robots.planning.gridgraph import GridGraph
from robots.planning.astar import astar

def test_astar():

    mask = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    graph = GridGraph(mask, lambda a,b: 1., lambda a,b: 0.)

    path, costs = astar((0, 0), (4, 4), graph)
    np.testing.assert_allclose(path, [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4],
        [4, 4]
    ]) # note, cells are [col, row] as points are [x, y]
    assert costs == 8
        
def test_astar_nopath():

    mask = np.array([
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    graph = GridGraph(mask, lambda a,b: 1., lambda a,b: 0.)

    def heuristic(node, goal):
        return 0.

    path, cost = astar((0, 0), (2, 0), graph)
    np.testing.assert_allclose(path, [])
    assert np.isinf(cost)