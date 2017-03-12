import numpy as np

from robots.planning.trajectories.cubic import CubicTrajectory

class QuinticTrajectory:
    """Multiple point multiple axis trajectory planning using piecewise fifth order polynomials.

    Given N points in M coordinates, N time instants and N target velocities at given time instants,
    this class computes a fith order polynomial per motion segment. Position, velocity and acceleration
    profiles are continous, while jerk is usually not. Points will be reached exactly.

    References
    ----------
    
    Biagiotti, Luigi, and Claudio Melchiorri. 
    Trajectory planning for automatic machines and robots. 
    Springer Science & Business Media, 2008. pp. 27-
    """

    def __init__(self, q, dq, ddq, qt):
        """Create a CubicTrajectory.

        Params
        ------
        q : NxM array
            Sequence of N points in M coordinates
        dq : NxM array, Nx1 array or 2x1 array
            Sequence of N velocities in M coordinates.
        ddq : NxM array, Nx1 array or 2x1 array
            Sequence of N accelerations in M coordinates.
        qt : 1xN array
            Time instants of sampling points `q`.
        """

        assert len(q) > 1
        assert len(q) == len(qt)

        q = np.asarray(q).astype(float)
        if q.ndim == 1:
            q = q.reshape(-1, 1)
        
        dq = np.asarray(dq).astype(float)
        if dq.ndim == 1:
            dq = dq.reshape(-1, 1)
            if q.shape[1] > 1:
                dq = np.tile(dq, (1, q.shape[1]))

        ddq = np.asarray(ddq).astype(float)
        if ddq.ndim == 1:
            ddq = ddq.reshape(-1, 1)
        if q.shape[1] > 1:
            ddq = np.tile(ddq, (1, q.shape[1]))

        self.t = np.asarray(qt).astype(float)

        if len(dq) == 2:
            # Heuristically determine velocities
            dq = CubicTrajectory.estimate_velocities(q, dq[0], dq[-1], self.t)

        if len(ddq) == 2:
            ddq = CubicTrajectory.estimate_velocities(dq, ddq[0], ddq[-1], self.t)

        self.nseg = q.shape[0] - 1
        self.coeff = np.zeros((self.nseg*6, q.shape[1]))

        for i in range(self.nseg):
            self.coeff[i*6:(i+1)*6] = QuinticTrajectory.solve(q[i], q[i+1], dq[i], dq[i+1], ddq[i], ddq[i+1], qt[i+1]-qt[i])

    def __call__(self, t):
        """Compute positions, velocities and accelerations at query time instants.

        Params
        ------
        t : scalar or 1xN array
            Time instants to compute trajectory configurations for. [qt[0]..qt[-1]]. If query
            times are outside the interval, they are clipped to the nearest valid value.
        
        Returns
        -------
        q : NxM array
            Position information in rows. One row per query time instant.
        dq : NxM array
            Velocity information in rows. One row per query time instant. 
        ddq : NxM array
            Acceleration information in rows. One row per query time instant.
        """

        isscalar = np.isscalar(t)
        
        t = np.atleast_1d(t)                
        i = np.digitize(t, self.t) - 1
        i = np.clip(i, 0, self.nseg - 1)

       
        t = t - self.t[i]
        t = t.reshape(-1, 1) # Needed for elementwise ops
        
        r = i*6
        q = self.coeff[r+0] + self.coeff[r+1]*t + self.coeff[r+2]*t**2 + self.coeff[r+3]*t**3 + self.coeff[r+4]*t**4 + self.coeff[r+5]*t**5
        dq = self.coeff[r+1] + 2 * self.coeff[r+2]*t + 3*self.coeff[r+3]*t**2 + 4*self.coeff[r+4]*t**3 + 5*self.coeff[r+5]*t**4
        ddq = 2*self.coeff[r+2] + 6*self.coeff[r+3]*t + 12*self.coeff[r+4]*t**2 + 20*self.coeff[r+5]*t**3
        
        return (q, dq, ddq) if not isscalar else (q[0], dq[0], ddq[0])


    @staticmethod
    def solve(q0, q1, v0, v1, a0, a1, dt):
        """Solve for the quintic polynomial coefficients.
        
        Note that the coefficients given in "Trajectory planning for automatic machines and robots" 
        did not work out, so we reframe it as A*x=b.        
        """

        na = len(q0)
        A = np.zeros((6, 6))
        b = np.zeros((6, na))

        ti = 0.
        tf = dt

        A[0] = [1., ti, ti**2, 1*ti**3, 1*ti**4, 1*ti**5]; b[0] = q0
        A[1] = [1., tf, tf**2, 1*tf**3, 1*tf**4, 1*tf**5]; b[1] = q1 
        A[2] = [0., 1., 2*ti, 3*ti**2, 4*ti**3, 5*ti**4];  b[2] = v0 
        A[3] = [0., 1., 2*tf, 3*tf**2, 4*tf**3, 5*tf**4];  b[3] = v1 
        A[4] = [0., 0., 2., 6*ti, 12*ti**2, 20*ti**3];     b[4] = a0 
        A[5] = [0., 0., 2., 6*tf, 12*tf**2, 20*tf**3];     b[5] = a1 
            
        c = np.linalg.solve(A, b)
        return c.reshape(6, na)