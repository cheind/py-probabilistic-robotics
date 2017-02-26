import numpy as np

def safe_invdir(d):
    with np.errstate(divide='ignore'):
        return np.where(d == 0., 1e12, 1./d)

class BBox:
    def __init__(self, mincorner, maxcorner):
        self.bounds = np.column_stack((mincorner, maxcorner)).astype(float)
        
    @property
    def mincorner(self):
        return self.bounds[:, 0]

    @property
    def maxcorner(self):
        return self.bounds[:, 1]

    def intersect_with_ray(self, o, d):
        #http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms

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