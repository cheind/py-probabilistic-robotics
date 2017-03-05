import numpy as np

class GridGraph:
    """Graph for dense grid meshes to be used with A* algorithm.
    
    The graph takes a Grid object that defines map. All free cells
    are assumed to be cells that evaluate to False. All blocked cells
    are cells that evaluate to True.

    A cost(a,b) function needs to be provided that returns the costs
    for moving between node a and neighbor b.

    A heuristic(a, goal) function needs to be provided that returns
    an optimisitic guess of the remaining costs between node a and node goal.
    """

    FourN = np.array([
        [0, -1], #N
        [1, 0],  #E
        [0, 1],  #S
        [-1, 0]  #W 
    ], dtype=int)
    """4-neighbor grid connectivity."""

    def __init__(self, map, cost, heuristic, connectivity=FourN):
        """Create a GridGraph.

        Params
        ------
        map : MxN array
            The grid map in divided in free and blocked cells.
        cost : (a, b) -> float
            The cost function for a move between cell a and b.
        heuristic: (a, goal) -> float
            The optimistic guess of the remaining cost between cell a and goal.
        connectivity: Kx2 array
            Potential neighbors of each cell.
        """

        self.map = map
        self.nhood = connectivity
        self.shape = self.map.shape[::-1] # account for xy reversal of cell axis
        self.cost = cost
        self.heuristic = heuristic

    def neighbors(self, node):
        """Returns the available neighbors of given node.

        The potential neighbors of a cell are based on the connectivity 
        given at construction and is limited to filter criteria such as
            - is neighbor within map bounds
            - is neighbor a free cell to move

        Params
        ------
        node : tuple
            Neighbors of this cell are returned
        """
        node = np.asarray(node)
        nodes = node + self.nhood
        nodes = self.filter_inbounds(nodes)
        nodes = self.filter_freespace(nodes)
        return tuple(map(tuple, nodes)) # convert to tuples to make them hashable

    def filter_inbounds(self, nodes):
        """Removes nodes that aren't within map bounds."""
        inbounds = np.logical_and(nodes >= 0, nodes < self.shape)
        return nodes[np.logical_and.reduce(inbounds, axis=1)]

    def filter_freespace(self, nodes):
        """Removes nodes that are blocked by map."""
        cells = self.map[nodes[:, 1], nodes[:, 0]].astype(bool)
        return nodes[~cells]