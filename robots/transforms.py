import numpy as np
import math

def h(x, n=1., axis=0):    
    s = [d if i != axis else 1 for i, d in enumerate(x.shape)]
    h = np.full((s), n)
    return np.append(x, h, axis=axis)

def hnorm(x, axis=0, skip_division=False):
    selh = [slice(None) if i != axis else slice(-1, None) for i in range(len(x.shape))]
    selx = [slice(None) if i != axis else slice(-1) for i in range(len(x.shape))]
    
    if skip_division:
        return np.copy(x[selx])
    else:
        h = x[selh]
        return x[selx] / h

def rigid_inverse(m):
    t = m[:2, 2]
    r = m[:2, :2].T

    mnew = np.eye(3)
    mnew[:2, :2] = r
    mnew[:2, 2] = -np.dot(r, t)

    return mnew

def pose_in_world(pose):
    """Returns a 3x3 matrix representing the pose vector in world space."""

    c = math.cos(pose[2])
    s = math.sin(pose[2])

    return np.array([
        [c, -s, pose[0]],
        [s, c, pose[1]],
        [0, 0., 1]
    ])

def world_in_pose(pose):
    """Returns a 3x3 matrix representing the world in space defined by pose."""
    return rigid_inverse(pose_in_world(pose))
    
def transform(m, x, hvalue=1.):
    needh = x.shape[0] == 2
    if needh:
        x = h(x, n=hvalue)
    x = np.dot(m, x)    
    if needh:
        x = hnorm(x, skip_division=True)
    return x

def pose_from_transform(m):
    return np.array([m[0,2], m[1,2], math.atan2(m[1,0], m[0,0])])