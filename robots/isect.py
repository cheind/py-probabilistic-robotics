
import numpy as np
import matplotlib.pyplot as plt

def inverse_direction(d):
    with np.errstate(divide='ignore'):
        return np.where(d == 0., 1e12, 1./d)

def ray_box(o, d, bounds):
    #http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms

    invd = inverse_direction(d)
    low = (bounds[0] - o) * invd
    high = (bounds[1] - o) * invd

    tmin = np.max(np.minimum(low, high))
    tmax = np.min(np.maximum(low, high))
    
    if tmax < 0:
        return False, tmin, tmax
    elif tmin > tmax:
        return False, tmin, tmax
    else:
        return True, max(tmin, 0.), tmax

def ray_grid(o, d, bounds, shape, hitmask=None, hitfnc=None):
    """Returns the intersection of a ray and grid.

    Based on
        Amanatides, John, and Andrew Woo. 
        "A fast voxel traversal algorithm for ray tracing." 
        Eurographics. Vol. 87. No. 3. 1987.
    """
    def mask_hit(o, d, t, cell):
        return hitmask[cell[1], cell[0]], t
    
    if hitfnc is None:
        hitfnc = mask_hit

    ret, tbox, tboxexit = ray_box(o, d, bounds)
    if not ret:
        return False, tbox, np.array([-1, -1])

    res = np.asarray(shape)
    cellsize = (bounds[1] - bounds[0]) / res
    ocell = (o + d * tbox) - bounds[0]
    cell = np.clip(np.floor(ocell / cellsize), 0, res - 1).astype(int)
    
    invd = inverse_direction(d)
    sgn = np.sign(d)
    add = np.where(sgn < 0, 0., 1.)
    delta = sgn * cellsize * invd
    nextcross = tbox + ((cell + add) * cellsize - ocell) * invd    

    t = tbox
    while (cell >= 0).all() and (cell < res).all():
        axis = np.argmin(nextcross)

        ret, hit = hitfnc(o, d, t, cell)
        if ret and hit < nextcross[axis]:
            return True, hit, cell    
                
        cell[axis] += sgn[axis]
        t = nextcross[axis]
        nextcross[axis] += delta[axis]        
        axis = np.argmin(nextcross)

    return False, t, cell



# class Grid:

#     def __init__(self, shape, mincorner=[0.,0.], cellsize=[1,1], dtype=None):
#         self.cells = np.zeros(shape, dtype=dtype)
#         self.mincorner = np.asarray(mincorner, dtype=float)
#         self.cellsize = np.asarray(cellsize, dtype=float)

#     def cell_index(self, x):
#         """Returns cell indices for one or more world space points."""
#         return np.floor((np.asarray(x) - self.mincorner) / self.cellsize).astype(int)

#     def index_inside(self, idx):
#         """Tests if one or more cell indices are contained within the bounds of the grid."""
#         idx = np.asarray(idx)
#         b = np.logical_and(idx >= (0,0), idx < self.cells.shape)
#         return np.all(b, axis=1)

#     def trace_ray(self, origin, direction):
#         # http://lodev.org/cgtutor/raycasting.html


#         ogrid = origin - self.mincorner
#         delta = np.zeros(2)
#         step = np.zeros(2)

#         if direction[0] < 0.:
#             delta[0] = -self.cells.shape[0] / direction[0]
#             step[0] = (np.floor(ogrid[0] / self.cellsize[0]) * self.cellsize[0] - ogrid[0]) / direction[0]
#         else:
#             delta[0] = self.cells.shape[0] / direction[0]
#             step[0] = ((np.floor(ogrid[0] / self.cellsize[0]) + 1.) * self.cellsize[0] - ogrid[0]) / direction[0]

#         if direction[1] < 0.:
#             delta[1] = -self.cells.shape[1] / direction[1]
#             step[1] = (np.floor(ogrid[1] / self.cellsize[1]) * self.cellsize[1] - ogrid[1]) / direction[1]
#         else:
#             delta[1] = self.cells.shape[1] / direction[1]
#             step[1] = ((np.floor(ogrid[1] / self.cellsize[1]) + 1.) * self.cellsize[1] - ogrid[1]) / direction[1]

#         print(step)
#         print(origin + step[0] * direction)
#         print(origin + step[1] * direction)
        
# g = Grid((10,10), cellsize=(0.5, 0.5))
# print(g.cell_index(np.array([
#     [0,0],
#     [1,2]
# ])))

# print(g.index_inside(((0,0), (-0.5,0.0))))


print(
    ray_box(np.array([-2, 0]), np.array([1, 0]), np.array([[0,0],[10,10]]))
)

fig, ax = plt.subplots()
ax.set_xticks(np.arange(-100,100,2))
ax.set_yticks(np.arange(-100,100,2))
ax.set_aspect('equal')

mask = np.zeros((10, 10))
mask[:, -1] = 1.


bounds = np.array([[2,2],[22,22]])
ax.imshow(mask, interpolation='none', cmap='summer', extent=[bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]])

dirs = np.random.uniform(-1., 1., size=(20  , 2))
dirs /= np.linalg.norm(dirs, axis=1)[:, np.newaxis]

for i in range(dirs.shape[0]):
    o = np.array([5, 5])
    
    """r = ray_grid(
        o,
        dirs[i],
        bounds,
        mask)"""

    def myhit(o, d, t, cell):
        p = o + t * d
        ax.scatter(p[0], p[1])
        return mask[cell[1], cell[0]], t

    r = ray_grid(
        o,
        dirs[i],
        bounds,
        mask.shape,
        hitfnc=myhit)

    d = o + r[1] * dirs[i]
    colors = ['red', 'blue']
    ax.plot((o[0], d[0]), (o[1], d[1]), color=colors[r[0]])

plt.grid()
plt.show()

#w = np.array([0, 0]) + r[1] * np.array([1, 0])
#plt.scatter()

#print( np.array([-2, 0]) + np.array([1, 0]) * r[1])

#print(r)

#plt.show()
