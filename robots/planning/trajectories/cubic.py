import numpy as np

class CubicTrajectory:
    """Multiple point multiple axis trajectory planning using piecewise third order polynomials.

    Given N points in M coordinates, N time instants and N target velocities at given time instants,
    this class computes a third order polynomial per motion segment. Position and velocity profiles 
    are continous, while acceleration is not. Points will be reached exactly.

    References
    ----------
    
    Biagiotti, Luigi, and Claudio Melchiorri. 
    Trajectory planning for automatic machines and robots. 
    Springer Science & Business Media, 2008. pp. 23-
    """

    def __init__(self, q, dq, qt):
        """Create a CubicTrajectory.

        Params
        ------
        q : NxM array
            Sequence of N points in M coordinates
        dq : NxM array, Nx1 array or 2x1 array
            Sequence of N velocities in M coordinates.
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

        self.t = np.asarray(qt).astype(float)

        if len(dq) == 2:
            # Heuristically determine velocities
            dq = CubicTrajectory.estimate_velocities(q, dq[0], dq[-1], self.t)

        self.nseg = q.shape[0] - 1
        self.coeff = np.zeros((self.nseg*4, q.shape[1]))

        for i in range(self.nseg):
            self.coeff[i*4:(i+1)*4] = CubicTrajectory.solve(q[i], q[i+1], dq[i], dq[i+1], qt[i+1]-qt[i])            

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
        
        r = i*4
        q = self.coeff[r+0] + self.coeff[r+1]*t + self.coeff[r+2]*t**2 + self.coeff[r+3]*t**3
        dq = self.coeff[r+1] + 2 * self.coeff[r+2]*t + 3*self.coeff[r+3]*t**2
        ddq = 2 * self.coeff[r+2] + 6*self.coeff[r+3]*t            

        return (q, dq, ddq) if not isscalar else (q[0], dq[0], ddq[0])

    @staticmethod
    def estimate_velocities(q, dq0, dqf, qt):
        """Estimate intermediate velocities."""
        dq = np.zeros(q.shape)      
        for i in range(1, len(dq) - 1):
            vk = (q[i] - q[i-1]) / (qt[i] - qt[i-1])
            vkn = (q[i+1] - q[i]) / (qt[i+1] - qt[i])
            mask = np.sign(vk) == np.sign(vkn)
            dq[i] = np.select([mask, ~mask], [0.5 * (vk + vkn), 0.])
        dq[0] = dq0
        dq[-1] = dqf
        return dq


    @staticmethod
    def solve(q0, q1, v0, v1, dt):
        """Solve for the cubic polynomial coefficients"""

        h = q1 - q0
        a0 = q0
        a1 = v0
        a2 = (3*h - (2*v0 + v1)*dt) / dt**2
        a3 = (-2*h + (v0 + v1)*dt) / dt**3

        return np.vstack((a0, a1, a2, a3))
    