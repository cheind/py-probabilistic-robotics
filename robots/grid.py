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

    def intersect_with_ray(self, o, d, hitmask=None):
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

        t = tbox
        while (cell >= 0).all() and (cell < self.resolution).all():
            axis = np.argmin(nextcross)

            if hitmask[cell[1], cell[0]]:
                return True, t, cell
                    
            cell[axis] += sgn[axis]
            t = nextcross[axis]
            nextcross[axis] += delta[axis]        
            axis = np.argmin(nextcross)

        return False, t, cell
    