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
        self.ta = np.zeros((self.nseg, self.q.shape[1]))
        self.td = np.zeros((self.nseg, self.q.shape[1]))
        self.v = np.zeros((self.nseg, self.q.shape[1]))        
        self.a = np.zeros((self.nseg, self.q.shape[1]))
        
        if dq is None:            
            # Initial / final velocities zero, others according to sign of velocity in
            # neighboring segments.
            self.v0 = np.zeros(self.q.shape)
            s = np.sign(np.diff(self.q, axis=0))
            for i in range(1, len(q)-1):
                cond = s[i-1] == s[i]
                self.v0[i] = np.select([cond, ~cond], [s[i-1] * dq_max, 0.])
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
        ta = self.ta
        td = self.td
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
    def eval_constraint(q0, q1, dq0, dq1, ddq_max):
        h = np.abs(q1-q0)
        return ddq_max*h >= np.abs(dq0**2 - dq1**2) * 0.5 # eq. 3.14

    @staticmethod
    def solve(q0, q1, dq0, dq1, dq_max, ddq_max):

        q0 = np.atleast_1d(q0)
        q1 = np.atleast_1d(q1)
        dq0 = np.abs(np.atleast_1d(dq0))
        dq1 = np.abs(np.atleast_1d(dq1))

        n = len(q0)

        # For coordinated motion, timing is build for axis with largest displacement
        d = q1 - q0
        h = np.abs(d)
        s = np.sign(d)
        i = np.argmax(h)

        assert np.all(TrapezoidalTrajectory.eval_constraint(q0, q1, dq0, dq1, ddq_max)), "Trajectory with given constraints impossible. Lower velocities or increase maximum acceleration."

        # From the axis with the largest displacement we first ta/td and t first using dq_max/ddq_max
        # as constraints. Given total duration t, we compute ta/td for all other axes using the constraints
        # ddq_max and t.

        ta = np.zeros(n)
        td = np.zeros(n)
        vlim = np.zeros(n)

        if h[i]*ddq_max > dq_max**2 - (dq0[i]**2 + dq1[i]**2) * 0.5:
            # vmax is reached
            vlim[i] = dq_max
            ta[i] = (dq_max - dq0[i]) / ddq_max
            td[i] = (dq_max - dq1[i]) / ddq_max
            t = h[i] / dq_max + (dq_max / (2 * ddq_max)) * (1 - dq0[i]/dq_max)**2 + (dq_max / (2 * ddq_max)) * (1 - dq1[i]/dq_max)**2
        else:
            # vmax is not reached, only accel and deaccel
            vlim[i] = math.sqrt(h[i]*ddq_max + (dq0[i]**2 + dq1[i]**2) * 0.5)
            ta[i] = (vlim[i] - dq0[i]) / ddq_max
            td[i] = (vlim[i] - dq1[i]) / ddq_max
            t = ta[i] + td[i]

        # For all other axes
        if n > 1:
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            vlim[mask] = 0.5*(dq0[mask] + dq1[mask] + ddq_max*t - np.sqrt(ddq_max**2*t**2 - 4*ddq_max*h[mask] + 2*ddq_max*(dq0[mask] + dq1[mask])*t - (dq0[mask] - dq1[mask])**2))

        dq = s * vlim
        ddq = s * ddq_max

        return t, ta, td, dq, ddq


def lspb(q, dt, ddq_max, t):
    q = np.asarray(q).astype(float)
    q = np.concatenate(([q[0]], q, [q[-1]]))
    dt = np.concatenate(([1], dt, [1]))
    
    dq = np.diff(q) / dt
    tb = np.abs(np.diff(dq)) / ddq_max
    ddq = (np.diff(dq)) / tb

    T = np.concatenate(([0.], np.cumsum(dt[1:-1])))
    tf = tb[0]*0.5 + np.sum(dt[1:-1]) + tb[-1]*0.5

    TS = T + tb[0]*0.5
    t = np.atleast_1d(t)        
    t = np.clip(t, 0, tf) 

    i = np.digitize(t, T+tb[0]*0.5, right=True)

    print(TS)
    print(t)
    print(i)
    
    t = t.reshape(-1, 1)
    print(np.logical_and(t >= (TS - tb*0.5), (t <= TS + tb*0.5)))
        


    """
    isb = np.logical_and(t >= (TS[i] - tb[i]*0.5), (t <= TS[i] + tb[i]*0.5))
    
    return \
        (q[i+1] + dq[i] * (t - TS[i]) + 0.5*ddq[i]*(t - TS[i] + tb[i]*0.5)**2) * isb + \
        (q[i+1] + dq[i+1] * (t - TS[i])) * ~isb
    """
    

t = np.linspace(0, 4, 20)
q = lspb([0, 1, 0.5], [2, 2], 1, t)

"""
import matplotlib.pyplot as plt

plt.scatter([0, 2, 4], [0, 1, 0.5])
plt.plot(t, q)
plt.show()
"""





        

