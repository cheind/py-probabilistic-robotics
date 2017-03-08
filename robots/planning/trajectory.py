import numpy as np
import math

from collections import namedtuple

class LSPBTrajectory:

    def __init__(self, viapoints, dt, acc):
        """
        Params
        ------
        viapoints : MxN array
            Waypoints of path in configuration space of M dimensions. For Cartesian space provide a 2xN array.
        dt: float or 1x(N-1) array
            Time between two waypoints.
        acc: float or 1xN array
            Default magnitude of accelerations in distance units / s**2.

        References
        ----------
        Kunz, Tobias, and Mike Stilman. Turning paths into trajectories using parabolic blends. Georgia Institute of Technology, 2011.
        https://smartech.gatech.edu/bitstream/handle/1853/41948/ParabolicBlends.pdf    
        """

        self.pos = np.asarray(viapoints)
        self.n = self.pos.shape[1]

        if np.isscalar(dt):
            dt = np.repeat(dt, self.n-1)
        else:            
            assert len(dt) == self.n-1
            dt = np.copy(dt)

        if np.isscalar(acc):
            acc = np.tile([acc], self.pos.shape)
        else:
            acc = np.asarray(acc)
            assert acc.shape == self.pos.shape
            
        self.v = np.zeros((self.pos.shape[0], self.n-1))
        self.a = np.zeros(self.pos.shape)
        self.bt = np.zeros(self.pos.shape)
        
        for i in range(self.pos.shape[0]): # For each axis
            r = self.lspb(self.pos[i, :], dt, acc[i, :])
            self.v[i, :] = r.v
            self.a[i, :] = r.a
            self.bt[i, :] = r.bt

        self.total_time = np.sum(dt)
        self.t = np.zeros(len(dt) + 1)
        self.t[1:] = np.cumsum(dt)

    LSPBResult = namedtuple('LSPBResult', 'v, a, bt')
    def lspb(self, pos, dt, acc):
        n = len(pos)

        v = np.zeros(n-1, dtype=float)
        a = np.zeros(n, dtype=float)
        bt = np.zeros(n, dtype=float)

        dpos = np.diff(pos)
        aa = np.abs(acc)

        # First segment
        a[0] = np.sign(dpos[0]) * aa[0]
        if a[0] != 0.:
            bt[0] = dt[0] - math.sqrt(dt[0]**2 - (2. * dpos[0]) / a[0]) # what if a == 0?            
        v[0] = dpos[0] / (dt[0] - 0.5 * bt[0])

        # Last segment
        a[-1] = np.sign(-dpos[-1]) * aa[-1]
        if a[-1] != 0.:
            bt[-1] = dt[-1] - math.sqrt(dt[-1]**2 + (2. * dpos[-1]) / a[-1])
        v[-1] = dpos[-1] / (dt[-1] - 0.5 * bt[-1])

        # Inner segments
        v[1:-1] = dpos[1:-1] / dt[1:-1]
        a[1:-1] = np.sign(np.diff(v)) * aa[1:-1]

        with np.errstate(all='ignore'):
            mask = a[1:-1] == 0.
            bt[1:-1] = np.select([mask, ~mask], [0., np.diff(v) / a[1:-1]])

        return LSPBTrajectory.LSPBResult(v=v, a=a, bt=bt)


    def __call__(self, t):
        assert t >= 0. and t <= self.total_time

        # Find interval associated with t
        
        #https://github.com/jhu-cisst/cisst/blob/2e90a6b69ac141a0ac386a933ad53d83327d448b/cisstRobot/code/robLSPB.cpp
        #http://www.ee.nmt.edu/~wedeward/EE570/SP09/gentraj.html
        torig = t
        i = np.digitize(t, self.t) - 1
        tf = self.t[i+1] - self.t[i]
        t = t - self.t[i]

        # Only one of the following will be true for each axis
        isacc = t <= self.bt[:, i]
        isdeacc = t >= (tf - self.bt[:, i])
        islinear = np.logical_and(~isacc, ~isdeacc)

        t2 = t**2

        p = (self.pos[:, i] + 0.5 * self.a[:, i] * t2) * isacc + \
            (self.pos[:, i+1] - 0.5 * self.a[:, i] * tf**2 + self.a[:, i] * tf * t - 0.5 * self.a[:, i] * t2) * isdeacc + \
            (0.5 * (self.pos[:, i+1] + self.pos[:, i] - self.v[:, i] * tf) + self.v[:, i] * t) * islinear

        

            

        print('{} - {}|{}|{}|{}'.format(torig, isacc, isdeacc, islinear, p))

        """
        T = self.wayt[i]
        BTH = self.bthalf[i]

        isblend = (t >= T - BTH) and (t <= T + BTH)
        if isblend:
            x = self.viapoints[:, i] + self.v[:, i-1]*(t-T) + 0.5 * self.a[:, i] * (t - T + BTH)**2
        else:
            x = self.viapoints[:, i] + self.v[:, i]*(t-T)
    
        """
        #print(i)

traj = LSPBTrajectory(np.array([
    [10, 35, 25, 10],
    [0, 0, 0, 0]
]), [2, 1, 3], 50)

for t in np.arange(0, traj.total_time, 0.1):
    traj(t)