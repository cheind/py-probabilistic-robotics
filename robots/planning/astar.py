
import numpy as np
from heapq import heappush, heappop

def astar(start, goal, graph):
    """A-star path planning."""
    
    costs = {} # Costs so far per node
    source = {} # For each node the node we came from

    costs[start] = 0.
    source[start] = None

    marchfront = [(0, start)] # Open list of nodes to explore
    pathfound = False
    while marchfront:
        p = heappop(marchfront)[1]

        if p == goal:
            pathfound = True
            break
        
        for n in graph.neighbors(p):
            cost = costs[p] + graph.cost(p, n)
            if not n in costs or cost < costs[n]:
                costs[n] = cost
                heappush(marchfront, (cost + graph.heuristic(n, p), n))
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

    def __init__(self, grid, cost, heuristic, connectivity=FourN):
        self.map = grid.values
        self.nhood = connectivity
        self.shape = self.map.shape[::-1] # account for xy reversal of cell axis
        self.cost = cost
        self.heuristic = heuristic

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
        cells = self.map[nodes[:, 1], nodes[:, 0]].astype(bool)
        return nodes[~cells]

