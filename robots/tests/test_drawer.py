import numpy as np
import math
import pytest
from pytest import approx

import matplotlib.pyplot as plt
from robots.draw import Drawer

# These tests use https://github.com/matplotlib/pytest-mpl
# Generate baseline images from project root with
#   pytest --mpl-generate-path=robots/tests/baseline_images
# Run tests with mpl comparison
#   pytest --mpl

@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_draw_points():

    fig, ax = plt.subplots()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect('equal')

    d = Drawer()
    
    points = np.array([
        [0, 2, 4], # x
        [0, 0, 2]  # y
    ], dtype=float)

    d.draw_points(points, ax, marker=(5,1), fc=('r', 'b', 'g'))
    return fig


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
