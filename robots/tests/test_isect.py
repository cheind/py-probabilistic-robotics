
import numpy as np
import math
from pytest import approx


from robots import isect

def test_ray_box():
    bounds = np.array([[0,0],[10,10]])

    o = np.array([-2, 0])
    d = np.array([1, 0])
    ret, tbox, tboxexit = isect.ray_box(o, d, bounds)
    assert ret
    assert tbox == approx(2.)
    assert tboxexit == approx(12)

    o = np.array([-2, -1])
    d = np.array([1, 0])
    ret, tbox, tboxexit = isect.ray_box(o, d, bounds)
    assert not ret
    
    o = np.array([5, 5])
    d = np.array([1, 1])
    d = d / np.linalg.norm(d)
    ret, tbox, tboxexit = isect.ray_box(o, d, bounds)
    assert ret
    assert tbox == approx(0.)
    assert tboxexit == approx(math.sqrt(5**2+5**2))

    o = np.array([5, 5])
    d = np.array([-1, -1])
    d = d / np.linalg.norm(d)
    ret, tbox, tboxexit = isect.ray_box(o, d, bounds)
    assert ret
    assert tbox == approx(0.)
    assert tboxexit == approx(math.sqrt(5**2+5**2))


def test_ray_grid():
    
    bounds = np.array([[0,0],[10,10]])
    mask = np.zeros((10, 10))
    mask[:, -1] = 1.
    mask[:, 0] = 1.

    o = np.array([-2, 0])
    d = np.array([1, 0])
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert ret
    assert thit == approx(2)
    assert (cell == [0,0]).all()

    o = np.array([0, 0])
    d = np.array([1, 0])
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert ret
    assert thit == approx(0)
    assert (cell == [0,0]).all()

    o = np.array([2.2, 0])
    d = np.array([1, 0])
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert ret
    assert thit == approx(7-0.2)
    assert (cell == [9,0]).all()

    o = np.array([2.2, 0])
    d = np.array([-1, 0])
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert ret
    assert thit == approx(1.2)
    assert (cell == [0,0]).all()

    o = np.array([2.2, 0])
    d = np.array([0, 1])
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert not ret

    o = np.array([-2.2, 0])
    d = np.array([0, 1])
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert not ret

    o = np.array([0, 0])
    d = np.array([0, 1])
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert ret

    o = np.array([2, 2])
    d = np.array([1, 1])
    d = d / np.linalg.norm(d)
    ret, thit, cell = isect.ray_grid(o, d, bounds, mask.shape, hitmask=mask)
    assert ret
    assert thit == approx(math.sqrt(7*7 + 7*7))
    assert (cell == [9,9]).all()
