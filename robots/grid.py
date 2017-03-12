import numpy as np

from robots.bbox import BBox
from robots.bbox import safe_invdir
from robots.posenode import PoseNode

class Grid(PoseNode):
    """A discretized axis aligned two-dimensional grid.

    Grids store information in discrete cells. Cells can 
    be empty or filled. Ray-tracing allows detecting intersections
    between filled cells and rays. Grids are often used to
    store obstacle / environment maps. Raytracing can then be used
    to determine visibility of landmarks.

    Attributes
    ----------
    values : MxM array 
        Cell values
    bbox : BBox
        Bounding box in local frame
    resolution : 1x2 array
        Shape of cell array
    cellsize : 1x2 array
        Width and height of each cell
    """

    def __init__(self, values, mincorner, maxcorner, **kwargs):
        """Create a Grid.

        Params
        ------
        values : NxM array 
            Cell values
        mincorner : 1x2 array
            Minimum corner of grid bounds
        maxcorner : 1x2 array
            Maximum corner of grid bounds

        Kwargs
        ------
        pose : 1x3 array, optional
            Pose vector. If omitted identity is assumed.
        """
        self.values = np.asarray(values)
        self.bbox = BBox(mincorner, maxcorner)
        self.resolution = np.asarray(self.values.shape)
        self.cellsize = (self.bbox.maxcorner - self.bbox.mincorner) / self.resolution

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(Grid, self).__init__(pose=pose)

    def intersect_with_ray(self, o, d, tmax=None, hitmask=None):
        """Returns the intersection of a ray and filled grid cells.

        Empty cells are cells that evaluate to False. This holds true
        for cells of boolean type and numeric type when the numeric value
        is zero.

        Params
        ------
        o : 1x2 array
            Ray orgin in the coordinate frame of the grid.
        d : 1x2 array
            Unit length ray direction in the coordinate frame of the grid.
        tmax : float, optional
            Maximum parametric ray time
        hitmask : MxM array, optional 
            State of cells. If omitted self.values is used.
        
        Returns
        -------
        ret : bool
            Indicating whether or not an intersection occurred.
        t : float
            Parametric ray time of intersection
        cell : 1x2 array
            Cell index of intersection

        References
        ----------
            Amanatides, John, and Andrew Woo. "A fast voxel traversal algorithm for ray tracing." Eurographics. Vol. 87. No. 3. 1987.
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
        """Returns the cell index by flooring fractional parts."""
        return np.floor((x - self.bbox.mincorner) / self.cellsize).astype(int)
    
    def cell_ceil(self, x):
        """Returns the cell index by ceiling fractional parts."""
        return np.ceil((x - self.bbox.mincorner) / self.cellsize).astype(int)

    def coords_in_parent(self, cell):
        """Returns cell coordinates in parent frame."""
        return self.bbox.mincorner + cell * self.cellsize

    def intersect_with_circle(self, center, radius, hitmask=None):
        """Returns the intersection of a circle and filled grid cells.

        Empty cells are cells that evaluate to False. This holds true
        for cells of boolean type and numeric type when the numeric value
        is zero.

        Params
        ------
        center : 1x2 array
            Circle center in the coordinate frame of the grid.
        radius : float
            Radius of circle.
        hitmask : MxM array, optional
            State of cells. If omitted self.values is used.
        
        Returns
        -------
        ret : bool 
            Whether or not an intersection occurred.
        cell : 1x2 array 
            Cell index of intersection.
        """

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
                b_min = self.coords_in_parent([i, j])
                b_max = self.coords_in_parent([i + 1, j + 1])

                nearest = np.maximum(b_min, np.minimum(center, b_max))
                delta = nearest - center
                if np.dot(delta, delta) < r2:
                    return True, [i, j]

        return False, [-1, -1]
