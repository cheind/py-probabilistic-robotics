import numpy as np
import math

def h(x, n=1., axis=1):
    """Returns the homogeneous version of given vectors.

    Params
    ------
    x : NxM array 
        Array of row or column vectors
    n : float
        Scalar value to use for padding
    axis : int
        Axis along which padding is performed [0,1]

    Returns
    -------
        (N+1)xM array of vectors when axis = 0, else Nx(M+1)
    """
    s = [d if i != axis else 1 for i, d in enumerate(x.shape)]
    h = np.full((s), n)
    return np.append(x, h, axis=axis)

def hnorm(x, axis=1, skip_division=False):
    """Returns the normalized, homogeneous undone, version of given vectors.

    Params
    ------
    x : NxM array
        Array of row or column vectors  
    axis : int
        Axis along which normalization is performed 0,1
    skip_division : bool, optional
        If true does not perform normalizing division

    Returns
    -------
        (N-1)xM array of vectors when axis = 0, else Nx(M-1)
    """
    selh = [slice(None) if i != axis else slice(-1, None) for i in range(len(x.shape))]
    selx = [slice(None) if i != axis else slice(-1) for i in range(len(x.shape))]
    
    if skip_division:
        return np.copy(x[selx])
    else:
        h = x[selh]
        return x[selx] / h

def rigid_inverse(m):
    """Returns the inverse of a rigid 3x3 transform.

    Params
    ------
    m : 3x3 array
        Matrix to invert

    Returns
    -------
    3x3 matrix 
        Inverse of `m`
    """
    t = m[:2, 2]
    r = m[:2, :2].T

    mnew = np.eye(3)
    mnew[:2, :2] = r
    mnew[:2, 2] = -np.dot(r, t)

    return mnew

def transform_from_pose(pose):
    """Returns the 3x3 transform associated with the 3x1 pose vector.
    
    Params
    ------
    pose : 1x3 array 
        Pose vector representing x, y, phi
    
    Returns
    -------
    3x3 matrix
        Matrix associated with pose

    """
    c = math.cos(pose[2])
    s = math.sin(pose[2])

    return np.array([
        [c, -s, pose[0]],
        [s, c, pose[1]],
        [0, 0., 1]
    ])

def pose_from_transform(m):
    """Returns the 1x3 pose vector associated with the given transform matrix.

    Params
    ------
    m : 3x3 matrix
        Matrix to extract pose vector from

    Returns
    -------
    1x3 array
        Pose vector
    """
    return np.array([m[0,2], m[1,2], math.atan2(m[1,0], m[0,0])])

def transform(m, x, hvalue=1.):
    """Applies and returns the transformation `m` to the vectors `x`.

    This method will add an ambient dimeension to given vectors in order
    to make them compatible with the given 3x3 matrix. The value of the
    ambient dimension is given by `hvalue`. Use 1. for point like types
    and 0. for direction like vectors. 

    Params
    ------
    m : 3x3 array
        Transformation matrix
    x : Nx2 or Nx3 array
        List of vectors in rows.
    hvalue: float, optional
        Ambient dimension value used when `x` is of dimension Nx2

    Returns
    -------
    Nx2 or Nx3 array
        List of transformed vectors.
    """
    needh = x.shape[1] == 2
    if needh:
        x = h(x, n=hvalue)
    x = np.dot(m, x.T).T  
    if needh:
        x = hnorm(x, skip_division=True)
    return x
