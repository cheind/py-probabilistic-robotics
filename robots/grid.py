import numpy as np
from robots.bbox import safe_invdir

class Grid:

    def __init__(self, values, bbox):
        self.values = np.asarray(values)
        self.bbox = bbox
        self.resolution = np.asarray(self.values.shape)
        self.cellsize = (bbox.maxcorner - bbox.mincorner) / self.resolution

    @property
    def mincorner(self):
        return self.bbox.mincorner

    @property
    def maxcorner(self):
        return self.bbox.maxcorner

    @property
    def bounds(self):
        return self.bbox.bounds

    def intersect_with_ray(self, o, d, tmax=None, hitmask=None):
        """Returns the intersection of a ray and grid.

        Based on
            Amanatides, John, and Andrew Woo. 
            "A fast voxel traversal algorithm for ray tracing." 
            Eurographics. Vol. 87. No. 3. 1987.
        """

        ret, tbox, tboxexit = self.bbox.intersect_with_ray(o, d)
        if not ret:
            return False, tbox, np.array([-1, -1])

        ocell = (o + d * tbox) - self.bbox.mincorner
        cell = np.clip(np.floor(ocell / self.cellsize), 0, self.resolution - 1).astype(int)
        
        invd = safe_invdir(d)
        sgn = np.sign(d)
        add = np.where(sgn < 0, 0., 1.)
        delta = sgn * self.cellsize * invd
        nextcross = tbox + ((cell + add) * self.cellsize - ocell) * invd    

        if hitmask is None:
            hitmask = self.values

        if tmax is None:
            tmax = float('inf')
        tmax = min(tboxexit, tmax)
        
        t = tbox
        while (t <= tmax) and (cell >= 0).all() and (cell < self.resolution).all():
            axis = np.argmin(nextcross)

            if hitmask[cell[1], cell[0]]:
                return True, t, cell
                    
            cell[axis] += sgn[axis]
            t = nextcross[axis]
            nextcross[axis] += delta[axis]        
            axis = np.argmin(nextcross)

        return False, t, cell


    def cell_floor(self, x):
        return np.floor((x - self.bbox.mincorner) / self.cellsize).astype(int)
    
    def cell_ceil(self, x):
        return np.ceil((x - self.bbox.mincorner) / self.cellsize).astype(int)

    def world_coords(self, cell):
        return self.bbox.mincorner + cell * self.cellsize

    def intersect_with_circle(self, center, radius, hitmask=None):
        if hitmask is None:
            hitmask = self.values


        c_min = np.clip(self.cell_floor(center - [radius, radius]), 0, self.resolution - 1)
        c_max = np.clip(self.cell_ceil(center + [radius, radius]), 0, self.resolution - 1)

        r2 = radius**2
        for i in range(c_min[0], c_max[0]):
            for j in range(c_min[1], c_max[1]):
                if not hitmask[j, i]:
                    continue

                # need to check circle to closest point
                b_min = self.world_coords([i, j])
                b_max = self.world_coords([i + 1, j + 1])

                nearest = np.maximum(b_min, np.minimum(center, b_max))
                delta = nearest - center
                if np.dot(delta, delta) < r2:
                    return True, [i, j]

        return False, [-1, -1]
