
import numpy as np


def ray_box(o, d, minc, maxc):
    #http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    invd = np.array([
        1. / d[0] if d[0] != 0. else 1e12,
        1. / d[1] if d[1] != 0. else 1e12
    ])
    low = (minc - o) * invd
    high = (maxc - o) * invd

    tmin = np.max(np.minimum(low, high))
    tmax = np.min(np.maximum(low, high))
    
    if tmax < 0:
        return False, tmax
    elif tmin > tmax:
        return False, tmax
    else:
        return True, tmin

class Grid:

    def __init__(self, shape, mincorner=[0.,0.], cellsize=[1,1], dtype=None):
        self.cells = np.zeros(shape, dtype=dtype)
        self.mincorner = np.asarray(mincorner, dtype=float)
        self.cellsize = np.asarray(cellsize, dtype=float)

    def cell_index(self, x):
        """Returns cell indices for one or more world space points."""
        return np.floor((np.asarray(x) - self.mincorner) / self.cellsize).astype(int)

    def index_inside(self, idx):
        """Tests if one or more cell indices are contained within the bounds of the grid."""
        idx = np.asarray(idx)
        b = np.logical_and(idx >= (0,0), idx < self.cells.shape)
        return np.all(b, axis=1)

    def trace_ray(self, origin, direction):
        # http://lodev.org/cgtutor/raycasting.html


        ogrid = origin - self.mincorner
        delta = np.zeros(2)
        step = np.zeros(2)

        if direction[0] < 0.:
            delta[0] = -self.cells.shape[0] / direction[0]
            step[0] = (np.floor(ogrid[0] / self.cellsize[0]) * self.cellsize[0] - ogrid[0]) / direction[0]
        else:
            delta[0] = self.cells.shape[0] / direction[0]
            step[0] = ((np.floor(ogrid[0] / self.cellsize[0]) + 1.) * self.cellsize[0] - ogrid[0]) / direction[0]

        if direction[1] < 0.:
            delta[1] = -self.cells.shape[1] / direction[1]
            step[1] = (np.floor(ogrid[1] / self.cellsize[1]) * self.cellsize[1] - ogrid[1]) / direction[1]
        else:
            delta[1] = self.cells.shape[1] / direction[1]
            step[1] = ((np.floor(ogrid[1] / self.cellsize[1]) + 1.) * self.cellsize[1] - ogrid[1]) / direction[1]

        print(step)
        print(origin + step[0] * direction)
        print(origin + step[1] * direction)
        
g = Grid((10,10), cellsize=(0.5, 0.5))
print(g.cell_index(np.array([
    [0,0],
    [1,2]
])))

print(g.index_inside(((0,0), (-0.5,0.0))))


print(
    ray_box(np.array([-2, 0]), np.array([1, 0]), np.array([0,0]), np.array([10,10]))
)

print(
    ray_box(np.array([5, 5]), np.array([1, 0]), np.array([0,0]), np.array([10,10]))
)