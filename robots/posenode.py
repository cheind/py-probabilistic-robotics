import numpy as np
import math

from robots import transforms


class PoseNode:

    def __init__(self, pose=[0,0,0], parent=None):
        self.node_parent = None
        self.node_children = {}
        self.pose = np.asarray(pose, dtype=float)

    def __getitem__(self, name):
        path = name.split('.')
        n = self
        for item in path:
            n = n.node_children[item]
        return n
    
    def __setitem__(self, name, obj):
        self.node_children[name] = obj
        obj.node_parent = self

    @property
    def root_node(self):
        n = self
        while n.node_parent is not None:
            n = n.node_parent
        return n

    @property
    def transform_to_parent(self):
        return transforms.pose_in_world(self.pose)

    @property
    def transform_from_parent(self):
        return transforms.world_in_pose(self.pose)

    @property
    def transform_to_world(self):
        t = self.transform_to_parent
        n = self.node_parent
        while n is not None:
            t = np.dot(n.transform_to_parent, t)
            n = n.node_parent
        return t

    @property
    def transform_from_world(self):
        t = self.transform_to_world
        return transforms.rigid_inverse(t)

    def transform_to(self, node):
        if isinstance(node, PoseNode):
            t1 = self.transform_to_world
            t2 = node.transform_from_world
            return np.dot(t2, t1)
        elif isinstance(node, str):
            raise NotImplementedError

    def transform_from(self, node):
        t = self.transform_to(node)
        return transforms.rigid_inverse(t)
        
