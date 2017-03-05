import numpy as np
from robots.planning.gridgraph import GridGraph

def test_gridgraph():

    mask = np.zeros((10,10))
    mask[5,5] = 1
    graph = GridGraph(mask, lambda a,b: 1., lambda a,b: 0.)

    np.testing.assert_allclose(graph.neighbors([0,0]), [[1, 0], [0, 1]])
    np.testing.assert_allclose(graph.neighbors([9,9]), [[9, 8], [8, 9]])
    np.testing.assert_allclose(graph.neighbors([5,4]), [[5, 3], [6,4], [4,4]])