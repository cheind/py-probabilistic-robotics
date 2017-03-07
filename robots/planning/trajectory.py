import numpy as np

class LSPBTrajectory:

    def __init__(self, viapoints, dt, bt):
        """
        Params
        ------
        viapoints : MxN array
            Waypoints of path in configuration space of M dimensions. For Cartesian space provide a 2xN array.
        dt: float or 1x(N-1) array
            Time between two waypoints.
        bt: float or 1xN array
            Parabolic blend time between waypoints. These are also provided for first and last waypoint.

        References
        ----------
        Kunz, Tobias, and Mike Stilman. Turning paths into trajectories using parabolic blends. Georgia Institute of Technology, 2011.
        https://smartech.gatech.edu/bitstream/handle/1853/41948/ParabolicBlends.pdf    
        """

        self.viapoints = np.asarray(viapoints)
        self.n = self.viapoints.shape[1]

        if np.isscalar(dt):
            self.dt = np.repeat(dt, self.n-1)
        else:
            assert len(dt) == self.n-1
            self.dt = np.copy(dt)

        if np.isscalar(bt):
            bt = np.repeat(bt, self.n)
        else:
            assert len(bt) == self.n

        self.bthalf = bt * 0.5

        # Assert timing of blend phases see eq. 4
        for i in range(self.n - 1):
            assert self.bthalf[i] + self.bthalf[i+1] <= self.dt[i]

        self.total_time = self.bthalf[0] + np.sum(self.dt) + self.bthalf[-1]
        self.wayt = (np.cumsum(self.dt) - self.dt[0]) + self.bthalf[0]
        
        # First and final velocities are zero.
        self.v = np.zeros(self.viapoints.shape)
        self.v[:, 1:] = np.diff(self.viapoints) / self.dt[:, np.newaxis]
        self.v[:, -1] = 0.

        self.a = np.zeros(self.viapoints.shape)
        self.a[:, 1:] = np.diff(self.v) / (self.bthalf[:-1] * 2.)


    def __call__(self, t):
        assert t >= 0. and t <= self.total_time

        # Find interval associated with t
        i = np.searchsorted(self.wayt, t)

        T = self.wayt[i]
        BTH = self.bthalf[i]

        isblend = (t >= T - BTH) and (t <= T + BTH)
        if isblend:
            x = self.viapoints[:, i] + self.v[:, i-1]*(t-T) + 0.5 * self.a[:, i] * (t - T + BTH)**2
        else:
            x = self.viapoints[:, i] + self.v[:, i]*(t-T)

        print(x)

t = LSPBTrajectory(np.array([
    [0, 5, 10],
    [0, 0, 0]
]), 1, 0.5)
t(0.1)
t(0.2)
t(0.3)
t(0.4)
t(0.6)
t(0.7)
t(0.8)
t(0.9)
t(0.95)
t(0.98)
t(0.99)
t(1.)


    