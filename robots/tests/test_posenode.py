import numpy as np
import math

from robots.posenode import PoseNode

def test_world():
    n = PoseNode()
    np.testing.assert_allclose(n.pose, [0,0,0])
    np.testing.assert_allclose(n.transform_to_world, np.eye(3))
    np.testing.assert_allclose(n.transform_from_world, np.eye(3))
    np.testing.assert_allclose(n.transform_to_parent, np.eye(3))
    np.testing.assert_allclose(n.transform_from_parent, np.eye(3))
    np.testing.assert_allclose(n.transform_to(n), np.eye(3))

def test_singlenode():
    w = PoseNode()
    c0 = PoseNode([10,10,0], parent=w)
    np.testing.assert_allclose(c0.transform_to_world, [[1, 0, 10],[0, 1, 10], [0,0,1]])
    np.testing.assert_allclose(c0.transform_from_world, [[1, 0, -10],[0, 1, -10], [0,0,1]])
    np.testing.assert_allclose(c0.transform_to_parent, [[1, 0, 10],[0, 1, 10], [0,0,1]])
    np.testing.assert_allclose(c0.transform_from_parent, [[1, 0, -10],[0, 1, -10], [0,0,1]])
    np.testing.assert_allclose(c0.transform_to(w), [[1, 0, 10],[0, 1, 10], [0,0,1]])

def test_siblingnodes():
    w = PoseNode()
    c0 = PoseNode([10,10,0], parent=w)
    c1 = PoseNode([-10,-10,math.pi/2], parent=w)

    np.testing.assert_allclose(c0.transform_to(c1), [[0, 1, 20],[-1, 0, -20], [0,0,1]], atol=1e-4)
    np.testing.assert_allclose(c0.transform_from(c1), [[0, -1, -20],[1, 0, -20], [0,0,1]], atol=1e-4)

def test_nestednodes():
    w = PoseNode()
    c0 = PoseNode([10,10,0], name='c0', parent=w)
    c1 = PoseNode([5, 5,math.pi/2], name='c1', parent=c0)

    np.testing.assert_allclose(w['c0.c1'].transform_to_world, [[0, -1, 15],[1, 0, 15], [0,0,1]], atol=1e-4)
    np.testing.assert_allclose(c1.transform_to_world, [[0, -1, 15],[1, 0, 15], [0,0,1]], atol=1e-4)
