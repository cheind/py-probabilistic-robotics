
import numpy as np
from heapq import heappush, heappop

def astar(start, goal, graph, heuristic):
    """A-star path planning."""
    
    costs = {} # Costs so far per node
    source = {} # For each node the node we came from

    costs[start] = 0.
    source[start] = None

    marchfront = [(0, start)] # Open list of nodes to explore
    pathFound = False
    while marchfront:
        p = heappop(marchfront)[1]

        if p == goal:
            pathfound = True
            break
        
        for n in graph.neighbors(p):
            cost = costs[p] + graph.cost(p, n)
            if not n in costs or cost < costs[n]:
                costs[n] = cost
                heappush(marchfront, (cost + heuristic(n, p), n))
                source[n] = p
        
    if pathfound:
        # Reconstruct path walking from goal to start
        n = goal
        path = [goal]
        while n is not start:
            s = source[n]
            path.append(s)
            n = s
        return path[::-1]
    else:
        return []


class GridGraph:

    FourN = np.array([
        [0, -1], #N
        [1, 0],  #E
        [0, 1],  #S
        [-1, 0]  #W 
    ], dtype=int)

    def __init__(self, grid, neighborhood=FourN):
        self.map = grid.values
        self.nhood = neighborhood
        self.shape = self.map.shape[::-1]

    def cost(self, nsrc, ndst):
        return 1.
    
    def neighbors(self, node):
        node = np.asarray(node)
        nodes = node + self.nhood
        nodes = self.filter_inbounds(nodes)
        nodes = self.filter_freespace(nodes)
        return tuple(map(tuple, nodes)) # convert to tuples to make them hashable

    def filter_inbounds(self, nodes):
        inbounds = np.logical_and(nodes >= 0, nodes < self.shape)
        return nodes[np.logical_and.reduce(inbounds, axis=1)]

    def filter_freespace(self, nodes):

        print(nodes)
        print('-------')
        cells = self.map[nodes[:, 1], nodes[:, 0]].astype(bool)
        return nodes[~cells]

