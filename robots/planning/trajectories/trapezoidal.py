import numpy as np
import math

class TrapezoidalTrajectory:

    def __init__(self, q, dq_max, ddq_max, dq=None):

        assert len(q) > 1

        self.q = np.asarray(q).astype(float)
        if self.q.ndim == 1:
            self.q = self.q.reshape(-1, 1)

        self.nseg = len(q) - 1

        self.dt = np.zeros(self.nseg)
        self.t = np.zeros(self.nseg+1)
        self.ta = np.zeros(self.nseg)
        self.td = np.zeros(self.nseg)
        self.v = np.zeros((self.nseg, self.q.shape[1]))        
        self.a = np.zeros((self.nseg, self.q.shape[1]))

        
        
        if dq is None:            
            # Initial / final velocities zero, others according to sign of velocity in
            # neighboring segments.
            self.v0 = np.zeros(self.q.shape)
            s = np.sign(np.diff(self.q, axis=0))
            for i in range(1, len(q)-1):
                if s[i-1] == s[i]:
                    self.v0[i] = s[i-1] * dq_max        
        else:
            self.v0 = np.asarray(dq)            
            if self.v0.ndim == 1:
                self.v0 = self.v0.reshape(-1, 1)
            assert self.v0.shape == (len(q), 1) or self.v0.shape == self.q.shape

        for i in range(self.nseg):
            self.dt[i], self.ta[i], self.td[i], self.v[i], self.a[i] = TrapezoidalTrajectory.solve(self.q[i], self.q[i+1], self.v0[i], self.v0[i+1], dq_max, ddq_max)

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
        td = self.td.reshape(-1, 1)
        dt = self.dt.reshape(-1, 1)

        isa = t < ta[i]
        isd = t > (dt[i] - td[i])
        isl = np.logical_and(~isa, ~isd)

        q = \
            (self.q[i] + self.v0[i]*t + 0.5*self.a[i]*t**2) * isa + \
            (self.q[i] + self.v0[i]*ta[i]*0.5 + self.v[i]*(t - ta[i]*0.5)) * isl + \
            (self.q[i+1] - self.v0[i+1]*(dt[i]-t) - 0.5*self.a[i]*(dt[i]-t)**2) * isd

        dq = \
            (self.v0[i] + self.a[i] * t) * isa + \
            self.v[i] * isl + \
            (self.v0[i+1] + self.a[i] * (dt[i]-t)) * isd

        ddq = \
            self.a[i] * isa + \
            0. * isl + \
            -self.a[i] * isd

        
        return (q, dq, ddq) if not isscalar else (q[0], dq[0], ddq[0])

    @staticmethod
    def solve(q0, q1, dq0, dq1, dq_max, ddq_max):

        q0 = np.atleast_1d(q0)
        q1 = np.atleast_1d(q1)
        dq0 = np.abs(np.atleast_1d(dq0))
        dq1 = np.abs(np.atleast_1d(dq1))

        # For coordinated motion, timing is build for axis with largest displacement
        d = q1 - q0
        h = np.abs(d)
        s = np.sign(d)
        i = np.argmax(h)

        assert ddq_max*h >= np.abs(dq0**2 - dq1**2) * 0.5, "Trajectory with given constraints impossible. Lower velocities or increase maximum acceleration."

        if h[i]*ddq_max > dq_max**2 - (dq0[i]**2 + dq1[i]**2) * 0.5:
            # vmax is reached
            vlim = dq_max
            ta = (dq_max - dq0[i]) / ddq_max
            td = (dq_max - dq1[i]) / ddq_max
            t = h[i] / dq_max + (dq_max / (2 * ddq_max)) * (1 - dq0[i]/dq_max)**2 + (dq_max / (2 * ddq_max)) * (1 - dq1[i]/dq_max)**2
        else:
            # vmax is not reached, only accel and deaccel
            vlim = math.sqrt(h[i]*ddq_max + (dq0[i]**2 + dq1[i]**2) * 0.5)
            ta = (vlim - dq0[i]) / ddq_max
            td = (vlim - dq1[i]) / ddq_max
            t = ta + td

        dq = s * vlim
        ddq = s * ddq_max
        
        """
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
        """

        return t, ta, td, dq, ddq







        

