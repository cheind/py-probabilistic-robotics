import numpy as np
import math

from robots import transforms


class PoseNode:
    """Represents a graph node with positional information.

    A PoseNode may contain children which have positional information themselves. 
    A full graph of PoseNodes thus represents kinematic chains of hierarchical ordered
    elements. A hierarchical representation of position information is often useful 
    in robotics. Most of the objects in this libary inherit PoseNode and are therefore
    compositable in a hierarchical fashion.

    Attributes
    ----------
    node_parent : PoseNode
        Parent of this node. None if root.
    node_children : dict
        Map of name to PoseNode for each child of this node.
    pose : 1x3 array
        Pose vector of this node with respect to parent frame.
    """

    def __init__(self, pose=[0,0,0], parent=None):
        """Create a PoseNode.

        Params
        pose : 1x3 array, optional
            Pose vector of this node representing x, y, phi.
        parent : PoseNode, optional
            Parent of this node.
        """            
        self.node_parent = parent
        self.node_children = {}
        self.pose = np.asarray(pose, dtype=float)

    def __getitem__(self, name):
        """Returns a child node by name.

        The name may contain multiple occurances of '.' as path separator. I.e            
            node['a.b.c']
        has the same meaning as writing
            node['a']['b']['c']

        Params
        ------
        name : str
            Name of node

        Returns
        -------
        PoseNode
            PoseNode associated with given name
        """
        path = name.split('.')
        n = self
        for item in path:
            n = n.node_children[item]
        return n
    
    def __setitem__(self, name, obj):
        """Adds a new node as children of this node.

        Params
        ------
        name : str
            Name of new node
        node : PoseNode 
            PoseNode to be added
        """
        self.node_children[name] = obj
        obj.node_parent = self

    @property
    def root_node(self):
        """Returns the root PoseNode of the tree."""
        n = self
        while n.node_parent is not None:
            n = n.node_parent
        return n

    @property
    def transform_to_parent(self):
        """Returns relative 3x3 transformation between this node and its parent."""        
        return transforms.transform_from_pose(self.pose)

    @property
    def transform_from_parent(self):
        """Returns relative 3x3 transformation between this node's parent and this node."""        
        return transforms.rigid_inverse(self.transform_to_parent)

    @property
    def transform_to_world(self):
        """Returns the relative 3x3 transformation between this node and the world frame."""
        t = self.transform_to_parent
        n = self.node_parent
        while n is not None:
            t = np.dot(n.transform_to_parent, t)
            n = n.node_parent
        return t

    @property
    def transform_from_world(self):
        """Returns the relative 3x3 transformation the world frame and this node."""
        t = self.transform_to_world
        return transforms.rigid_inverse(t)

    def transform_to(self, target):
        """Returns the relative transformation between this node and `target` node."""
        if isinstance(target, PoseNode):
            t1 = self.transform_to_world
            t2 = target.transform_from_world
            return np.dot(t2, t1)
        elif isinstance(target, str):
            raise NotImplementedError

    def transform_from(self, node):
        """Returns the relative transformation between `target` node and this node."""
        t = self.transform_to(node)
        return transforms.rigid_inverse(t)
        
