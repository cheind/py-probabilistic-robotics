import numpy as np
import math
from pytest import approx

from robots.draw import Drawer
from scipy.stats import chi2


def test_confidence_ellipse_params():

    c0 = np.array([
        [1, 0],
        [0, 1]
    ])

    c1 = np.array([
        [1, 0],
        [0, 2]
    ])

    d = Drawer()
    w, h, a = d._compute_ellipse_parameters([c0, c1])
    np.testing.assert_allclose(a, [0, math.pi/2], atol=1e-4)
