
import numpy as np
from heapq import heappush, heappop

def astar(start, goal, graph, return_explored=False):
    """A-star path planning.
    
    A* is a graph based algorithm to compute the a path between
    a start and goal node. A* explores unknown regions of the 
    graph using the connectivity of nodes provided by the graph.
    Nodes to be explored are priorized by the cost of getting
    to the node (sum of costs) plus some remaining cost estimage 
    to the goal (heuristic). A good heuristic allows A* to prune
    a huge number of potential nodes to explore, improving its 
    performance.

    Note that all nodes need to be hashable types.  

    Params
    ------
    start
        Start node for planning. Node is of unspecified type but needs to
        be hashable.
    goal
        Goal node for planning. Node is of unspecified type but needs to
        be hashable.
    graph : object
        An object that needs to provide the following operations
        
        graph.neighbors(n) -> list of nodes
        All explorable neighbor nodes of n

        graph.cost(a, b) -> float
        The cost of moving from a to neighbor b

        graph.heuristic(a, goal) -> float
        An optimistic guess of the remaining costs for moving from a to goal.
    return_explored : bool
        Wether or not to return all explored nodes in order of discovery.

    Returns
    -------
    path : array of nodes
        Sorted array of nodes from start to goal, or empty array if no path was found.
    finalcost : float
        Final cost for getting from start to goal, or infinite if no path was found.
    explored : array of nodes
        Array of nodes explored during traversal in order of discovery.
    """
    
    costs = {} # Costs so far per node
    source = {} # Track parents for each node for path reconstruction

    costs[start] = 0.
    source[start] = None

    marchfront = [(0, start)] # Open list of nodes to explore
    pathfound = False
    explored = []
    while marchfront:
        p = heappop(marchfront)[1]

        if return_explored:
            explored.append(p)

        if p == goal:
            pathfound = True
            break
        
        for n in graph.neighbors(p):
            cost = costs[p] + graph.cost(p, n)
            if not n in costs or cost < costs[n]:
                costs[n] = cost
                heappush(marchfront, (cost + graph.heuristic(n, goal), n))
                source[n] = p
        
    path = []
    finalcost = float('inf')
    if pathfound:
        # Reconstruct path by walking from goal to start
        n = goal
        path.append(goal)
        while n is not start:
            s = source[n]
            path.append(s)
            n = s
        path = path[::-1]
        finalcost = costs[goal]
    
    if return_explored:
        return path, finalcost, explored
    else:
        return path, finalcost

