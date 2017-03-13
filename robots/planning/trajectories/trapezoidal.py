import numpy as np
import math

class TrapezoidalTrajectory:

    def __init__(self, q, dq_max, ddq_max):

        assert len(q) > 1

        self.q = np.asarray(q).astype(float)
        if self.q.ndim == 1:
            self.q = self.q.reshape(-1, 1)

        self.nseg = len(q) - 1

        self.dt = np.zeros(self.nseg)
        self.t = np.zeros(self.nseg+1)
        self.ta = np.zeros(self.nseg)
        self.v = np.zeros((self.nseg, self.q.shape[1]))
        self.a = np.zeros((self.nseg, self.q.shape[1]))

        for i in range(self.nseg):
            self.dt[i], self.ta[i], self.v[i], self.a[i] = TrapezoidalTrajectory.solve(q[i], q[i+1], dq_max, ddq_max)

        self.t[1:] = np.cumsum(self.dt)

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
        t = np.clip(t, 0, self.t[-1])            
        i = np.digitize(t, self.t) - 1
        i = np.clip(i, 0, self.nseg - 1)
       
        t = t - self.t[i] # Time relative to segment
        t = t.reshape(-1, 1) # Needed for elementwise ops
        ta = self.ta.reshape(-1, 1)
        dt = self.dt.reshape(-1, 1)


        isa = t <= ta[i]
        isd = t > (dt[i] - ta[i])
        isl = np.logical_and(~isa, ~isd)

        q = \
            (self.q[i] + 0.5*self.a[i]*t**2) * isa + \
            (self.q[i] + self.v[i]*(t - ta[i]*0.5)) * isl + \
            (self.q[i+1] - 0.5*self.a[i]*(dt[i]-t)**2) * isd

        dq = \
            (self.a[i] * t) * isa + \
            self.v[i] * isl + \
            (self.a[i] * (dt[i]-t)) * isd

        ddq = \
            self.a[i] * isa + \
            0. * isl + \
            -self.a[i] * isd

        
        return (q, dq, ddq) if not isscalar else (q[0], dq[0], ddq[0])

    @staticmethod
    def solve(q0, q1, dq_max, ddq_max):

        q0 = np.atleast_1d(q0)
        q1 = np.atleast_1d(q1)

        # For coordinated motion, timing is build for axis with largest displacement
        d = q1 - q0
        h = np.abs(d)
        s = np.sign(d)
        i = np.argmax(h)

        ta = dq_max / ddq_max # Acceleration time
        t = (h[i] * ddq_max + dq_max**2) / (ddq_max * dq_max) # Total time

        if h[i] < (dq_max**2 / ddq_max):
            # No linear segment
            ta = math.sqrt(h[i] / ddq_max)
            t = 2 * ta
            dq_max = ddq_max * ta 

        ddq = (d) / (ta * (t - ta))
        dq = (d)/ (t - ta)

        dq[i] = s[i] * dq_max
        ddq[i] = s[i] * ddq_max

        return t, ta, dq, ddq







        
