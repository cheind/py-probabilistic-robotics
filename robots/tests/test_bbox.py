
import numpy as np
import math
from pytest import approx


from robots.bbox import BBox

def test_ray_box():

    bbox = BBox([0,0], [10,10])

    o = np.array([-2, 0])
    d = np.array([1, 0])
    ret, tbox, tboxexit = bbox.intersect_with_ray(o, d)
    assert ret
    assert tbox == approx(2.)
    assert tboxexit == approx(12)

    o = np.array([-2, -1])
    d = np.array([1, 0])
    ret, tbox, tboxexit = bbox.intersect_with_ray(o, d)
    assert not ret
    
    o = np.array([5, 5])
    d = np.array([1, 1])
    d = d / np.linalg.norm(d)
    ret, tbox, tboxexit = bbox.intersect_with_ray(o, d)
    assert ret
    assert tbox == approx(0.)
    assert tboxexit == approx(math.sqrt(5**2+5**2))

    o = np.array([5, 5])
    d = np.array([-1, -1])
    d = d / np.linalg.norm(d)
    ret, tbox, tboxexit = bbox.intersect_with_ray(o, d)
    assert ret
    assert tbox == approx(0.)
    assert tboxexit == approx(math.sqrt(5**2+5**2))