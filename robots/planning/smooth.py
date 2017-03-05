import numpy as np

def smooth_path(path, smoothweight=0.5):
    """Smooth path by adjusting x,y locations of path waypoints.

    Smoothing is formulated as a minimization problem. Call the original 
    waypoints X, we search for a set of waypoints Y that minimize

                 n-2                         n-2
        f(Y;X) = sum (Xi - Yi)**2 + lambda * sum (Yn+1 + Yn-1 - 2Yn)**2
       argmin Y  i=1                         i=1
    
    f(Y;X) as presented above assumes that the first and last point remains fixed, thus
    the waypoints optimized are the inner points of the path. The first sum in f(Y;X)
    is the data term that attempts to keep the Ys as close as possible to the Xs. The 
    second term attempts to bring subsequent Yns as close as possible. Yn+1 + Yn-1 - 2Yn
    is the sum of 

        Yn+1 - Yn
        Yn-1 - Yn

    The derivative of f with respect to Ys is linear and can be solved using system of 
    linear equations Ax = b. The partial derivative w.r.t Yn is given by differentiation
    w.r.t to the data term

        df/dYn = 2 * (Xn - Yn)

    The smoothness term is bit tricker as each Yn potentially contributes to three terms

        Yn + Yn+2 - 2Yn+1
        Yn-2 + Yn - 2Yn-1
        Yn+1 + Yn-1 - 2Yn

    This algorithm currently does not lead to collision free paths. Adding obstacle avoidance
    will probably lead to a non-linear system of equations and needs to be solved iteratively.

    Params
    ------
    path : Nx2 array
        Waypoints of the path 
    smoothweight:
        Balance factor between data and smoothness term.

    Returns
    -------
    Nx2 array
        Smoothes waypoints
    """

    A = np.zeros((len(path)*2,len(path)*2))
    b = np.zeros(len(path)*2)

    # Boundaries
    A[0,0] = 1.
    A[1,1] = 1.
    b[0] = path[0][0]
    b[1] = path[0][1]

    A[-2,-2] = 1.
    A[-1,-1] = 1.
    b[-2] = path[-1][0]
    b[-1] = path[-1][1]

    # data term
    for i in range(1, len(path)-1):
        j = i*2; jj = j+1
        A[j, j] += 2.
        A[jj, jj] += 2.
        b[j] += 2 * path[i][0]
        b[jj] += 2 * path[i][1]

    # smoothness term
    weight = smoothweight

    k_leftboundary = np.array([-6, 12, -8, 2], dtype=float) * weight
    k_noboundary = np.array([2, -8, 12, -8, 2], dtype=float) * weight
    k_rightboundary = np.array([2, -8, 12, -6], dtype=float) * weight

    for i in range(1, len(path)-1):
        if i == 1:
            A[i*2, (i-1)*2:(i+3)*2:2] += k_leftboundary
            A[i*2+1, (i-1)*2+1:(i+3)*2+1:2] += k_leftboundary
        elif i == len(path) - 2:
            A[i*2, (i-2)*2:(i+2)*2:2] += k_rightboundary
            A[i*2+1, (i-2)*2+1:(i+2)*2+1:2] += k_rightboundary
        else:
            A[i*2, (i-2)*2:(i+3)*2:2] += k_noboundary
            A[i*2+1, (i-2)*2+1:(i+3)*2+1:2] += k_noboundary
        
    x = np.linalg.solve(A, b)
    return x.reshape(len(path), 2)