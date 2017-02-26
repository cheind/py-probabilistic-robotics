
import numpy as np
import math
from pytest import approx

from robots.grid import Grid
from robots.bbox import BBox

def test_ray_grid():
    
    bbox = BBox([0,0], [10,10])

    mask = np.zeros((10, 10))
    mask[:, -1] = 1.
    mask[:, 0] = 1.    

    grid = Grid(mask, bbox)

    o = np.array([-2, 0])
    d = np.array([1, 0])
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert ret
    assert thit == approx(2)
    assert (cell == [0,0]).all()

    o = np.array([0, 0])
    d = np.array([1, 0])
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert ret
    assert thit == approx(0)
    assert (cell == [0,0]).all()

    o = np.array([2.2, 0])
    d = np.array([1, 0])
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert ret
    assert thit == approx(7-0.2)
    assert (cell == [9,0]).all()

    o = np.array([2.2, 0])
    d = np.array([-1, 0])
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert ret
    assert thit == approx(1.2)
    assert (cell == [0,0]).all()

    o = np.array([2.2, 0])
    d = np.array([0, 1])
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert not ret

    o = np.array([-2.2, 0])
    d = np.array([0, 1])
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert not ret

    o = np.array([0, 0])
    d = np.array([0, 1])
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert ret

    o = np.array([2.5, 2.5])
    d = np.array([9.0, 9.5]) - o
    d = d / np.linalg.norm(d)
    ret, thit, cell = grid.intersect_with_ray(o, d)
    assert ret
    assert (cell == [9,9]).all()