import numpy as np

from robots.posenode import PoseNode

def safe_invdir(d):
    """Returns `1/d` with special care about zero values in `d`."""
    with np.errstate(divide='ignore'):
        return np.where(d == 0., 1e12, 1./d)

class BBox(PoseNode):
    """A positional node representing an axis aligned box."""

    def __init__(self, mincorner, maxcorner, **kwargs):
        """Create a BBox.

        Params
            mincorner : 1x2 minimum corner of box
            maxcorner : 1x2 maximum corner of box

        Kwargs
            pose : (optional) 1x3 pose vector. If omitted identity is assumed.
        """
        self.bounds = np.column_stack((mincorner, maxcorner)).astype(float)

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(BBox, self).__init__(pose=pose)
        
    @property
    def mincorner(self):
        """Returns the 1x2 min-corner of the box."""
        return self.bounds[:, 0]

    @property
    def maxcorner(self):
        """Returns the 1x2 max-corner of the box."""
        return self.bounds[:, 1]

    def intersect_with_ray(self, o, d):
        """Intersects box with ray.

        Params
            o: 1x3 ray orgin in the coordinate frame of the box.
            d: 1x3 unit length ray direction in the coordinate frame of the box.

        Returns
            ret: boolean value indicating whether the ray hit the box or not.
            tmin: parametric ray time of ray entering box.
            tmax: parametric ray time of ray exiting box.
        """
        invd = safe_invdir(d)
        low = (self.mincorner - o) * invd
        high = (self.maxcorner - o) * invd

        tmin = np.max(np.minimum(low, high))
        tmax = np.min(np.maximum(low, high))
        
        if tmax < 0:
            return False, tmin, tmax
        elif tmin > tmax:
            return False, tmin, tmax
        else:
            return True, max(tmin, 0.), tmax