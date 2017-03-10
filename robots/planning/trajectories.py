import numpy as np
import math

class PolynomialTrajectory:
    """Computes a trajectory by fitting polynomials to a set of configuration samples.

    Given N via points in M coordinates, N-1 segment times, N target velocities at via points 
    and optionally N target accelerations at via points, this class computes a trajectory 
    by fitting third or fifth order polynomials.

    The trajectory will pass the via points exactly. The boundary conditions related to 
    via point velocitie and accelerations are also matched exactly. While the system is 
    matches the requirements at via points nicely, it might generate trajectories with 
    large velocities or accelerations in between the via points.

    PolynomialTrajectory precomputes its coefficients at construction and can therefore
    be queried efficiently. For each query time `t` the point, velocity and acceleration
    profile is returned.

        traj = PolynomialTrajectory(points=[10, 30], dt=[1], v=[0, 0], a=[0, 0])
        t = np.arange(-0.1, 1.1, 0.01)
        x, dx, ddx = traj(t)

    """

    def __init__(self, points, dt, v, a=None):
        points = np.asarray(points).astype(float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        nseg = points.shape[0] - 1
        naxis = points.shape[1]

        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
            if naxis > 1:
                v = np.tile(v, (1, naxis))
        
        if a is not None:
            a = np.asarray(a)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
                if naxis > 1:
                    a = np.tile(a, (1, naxis))

        assert nseg > 0
        assert len(dt) == nseg
        assert v.shape == points.shape or v.shape == (2, points.shape[1])
        assert a is None or (a.shape == points.shape or a.shape == (2, points.shape[1]))

        self.t = np.zeros(nseg + 1, dtype=float)
        self.t[1:] = np.cumsum(dt)
        self.total_time = np.sum(dt)

        self.order = 4 if a is None else 6    
        self.nseg = nseg

        def estimate_rates(x, dx, t):
            dxnew = np.zeros(x.shape)
            dxnew[0] = dx[0]
            dxnew[-1] = dx[-1]

            for i in range(1, len(dxnew) - 1):
                vk = (x[i] - x[i-1]) / (t[i] - t[i-1])
                vkn = (x[i+1] - x[i]) / (t[i+1] - t[i])
                mask = np.sign(vk) == np.sign(vkn)
                dxnew[i] = np.select([mask, ~mask], [0.5 * (vk + vkn), 0.])
            return dxnew

        if len(v) == 2:
            # Heuristically determine velocities
            v = estimate_rates(points, v, self.t)

        if a is not None and len(a) == 2:
            # Heuristically determine accelerations
            a = estimate_rates(v, a, self.t)
        
        self.coeff = np.zeros((nseg*self.order, naxis))
        if self.order == 4:            
            for i in range(naxis):                
                self.coeff[:, i] = PolynomialTrajectory.solve_cubic(points[:, i], self.t, v[:, i])
        else:
            for i in range(naxis):                
                self.coeff[:, i] = PolynomialTrajectory.solve_quintic(points[:, i], self.t, v[:, i], a[:, i])


    def __call__(self, t):
        scalar = np.isscalar(t)
        t = np.atleast_1d(t)        
        
        t = np.clip(t, 0., self.total_time)
        i = np.digitize(t, self.t) - 1
        i = np.clip(i, 0, self.nseg - 1)
        
        t = t.reshape(-1, 1) # To make elementwise ops working
        r = i * self.order

        if self.order == 4:
            x = self.coeff[r+0] + self.coeff[r+1]*t + self.coeff[r+2]*t**2 + self.coeff[r+3]*t**3
            dx = self.coeff[r+1] + 2 * self.coeff[r+2]*t + 3*self.coeff[r+3]*t**2
            ddx = 2 * self.coeff[r+2] + 6*self.coeff[r+3]*t            
        else:
            x = self.coeff[r+0] + self.coeff[r+1]*t + self.coeff[r+2]*t**2 + self.coeff[r+3]*t**3 + self.coeff[r+4]*t**4 + self.coeff[r+5]*t**5
            dx = self.coeff[r+1] + 2 * self.coeff[r+2]*t + 3*self.coeff[r+3]*t**2 + 4*self.coeff[r+4]*t**3 + 5*self.coeff[r+5]*t**4
            ddx = 2 * self.coeff[r+2] + 6*self.coeff[r+3]*t + 12*self.coeff[r+4]*t**2 + 20*self.coeff[r+5]*t**3
        
        return (x, dx, ddx) if not scalar else (x[0], dx[0], ddx[0])


    @staticmethod
    def solve_cubic(x, t, v):
        n = len(x)
        s = n - 1

        A = np.zeros((4*s, 4*s))
        b = np.zeros(4*s)

        for i in range(s):
            xi = x[i]; xf = x[i+1]
            vi = v[i]; vf = v[i+1]
            ti = t[i]; tf = t[i+1]

            r = 4 * i
            c = 4 * i

            A[r+0, c:c+4] = [1., ti, ti**2, 1*ti**3]; b[r+0] = xi # qi = c0 + c1*ti + c2*ti**2 + c3*ti**3
            A[r+1, c:c+4] = [1., tf, tf**2, 1*tf**3]; b[r+1] = xf # qf = c0 + c1*tf + c2*tf**2 + c3*tf**3
            A[r+2, c:c+4] = [0., 1., 2*ti, 3*ti**2];  b[r+2] = vi # vi = c1 + c2*2ti + c3*3ti**2
            A[r+3, c:c+4] = [0., 1., 2*tf, 3*tf**2];  b[r+3] = vf # vf = c1 + c2*2tf + c3*3tf**2
            
        c = np.linalg.solve(A, b)
        return c
    
    @staticmethod
    def solve_quintic(x, t, v, a):
        n = len(x)
        s = n - 1

        A = np.zeros((6*s, 6*s))
        b = np.zeros(6*s)

        for i in range(s):
            xi = x[i]; xf = x[i+1]
            vi = v[i]; vf = v[i+1]
            ai = a[i]; af = a[i+1]
            ti = t[i]; tf = t[i+1]

            r = 6 * i
            c = 6 * i

            A[r+0, c:c+6] = [1., ti, ti**2, 1*ti**3, 1*ti**4, 1*ti**5]; b[r+0] = xi 
            A[r+1, c:c+6] = [1., tf, tf**2, 1*tf**3, 1*tf**4, 1*tf**5]; b[r+1] = xf 
            A[r+2, c:c+6] = [0., 1., 2*ti, 3*ti**2, 4*ti**3, 5*ti**4];  b[r+2] = vi 
            A[r+3, c:c+6] = [0., 1., 2*tf, 3*tf**2, 4*tf**3, 5*tf**4];  b[r+3] = vf 
            A[r+4, c:c+6] = [0., 0., 2., 6*ti, 12*ti**2, 20*ti**3];     b[r+4] = ai 
            A[r+5, c:c+6] = [0., 0., 2., 6*tf, 12*tf**2, 20*tf**3];     b[r+5] = af 
            
        c = np.linalg.solve(A, b)
        return c

def lspb(p, dt, a, v, t):
    # x = x0 + t*v0 + 0.5*a*t**2
    # v = a*t -> t = v/a
    bt = v/a
    lt = dt - 2.*bt

    isacc = t < bt
    isdeacc = t > (dt - bt)
    islin = np.logical_and(~isacc, ~isdeacc)

    x = \
        (p[0] + 0.5*a*t**2) * isacc + \
        (p[0] + 0.5*a*bt**2 + (t-bt)*v) * islin + \
        (p[0] + 0.5*a*bt**2 + lt*v + (t-lt-bt)*v - 0.5*a*(t-lt-bt)**2 ) * isdeacc
    
    dx = \
        (a*t) * isacc + \
        (v) * islin + \
        (v - a*(t-lt-bt)) * isdeacc

    ddx = \
        (a) * isacc + \
        (0) * islin + \
        (-a) * isdeacc

    return x, dx, ddx

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.linspace(0, 5, 100)
    x, dx, ddx = lspb([0, 10], 5, 2, 3, t)

    fig, ax = plt.subplots()
    ax.plot(t, x, label='x')
    ax.plot(t, dx, label='dx')
    ax.step(t, ddx, label='ddx')
    ax.legend(loc='upper left')
    plt.show()


    """"
    traj = PolynomialTrajectory(np.array([10, 20, 0, 30, 40]), [2, 2, 4, 2], [0, 0, 0, 0, 0], [0, 0])
    t = np.linspace(-1, 11, 1000)
    x, dx, ddx = traj(t)


    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Position')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Velocity')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('Acceleration')

    #ax1.scatter([0, 2, 4, 8, 10], [10, 20, 0, 30, 40])
    ax1.plot(t, x[:, 0])
    ax2.plot(t, dx[:, 0])
    ax3.plot(t, ddx[:, 0])
    plt.tight_layout()
    plt.show()

    """


# from collections import namedtuple



# class LSPBTrajectory:

#     def __init__(self, viapoints, dt, acc):
#         """
#         Params
#         ------
#         viapoints : MxN array
#             Waypoints of path in configuration space of M dimensions. For Cartesian space provide a 2xN array.
#         dt: float or 1x(N-1) array
#             Time between two waypoints.
#         acc: float or 1xN array
#             Default magnitude of accelerations in distance units / s**2.

#         References
#         ----------
#         Kunz, Tobias, and Mike Stilman. Turning paths into trajectories using parabolic blends. Georgia Institute of Technology, 2011.
#         https://smartech.gatech.edu/bitstream/handle/1853/41948/ParabolicBlends.pdf    
#         """

#         self.pos = np.atleast_2d(viapoints).astype(float)
#         self.n = self.pos.shape[1]

#         if np.isscalar(dt):
#             dt = np.repeat(dt, self.n-1)
#         else:            
#             assert len(dt) == self.n-1
#             dt = np.copy(dt)

#         if np.isscalar(acc):
#             acc = np.tile([acc], self.pos.shape)
#         else:
#             acc = np.asarray(acc)
#             assert acc.shape == self.pos.shape
            
#         self.v = np.zeros((self.pos.shape[0], self.n-1))
#         self.lt = np.zeros((self.pos.shape[0], self.n-1))
#         self.a = np.zeros(self.pos.shape)
#         self.bt = np.zeros(self.pos.shape)
        
#         for i in range(self.pos.shape[0]): # For each axis
#             print(self.pos[i, :])
#             r = self.lspb(self.pos[i, :], dt, acc[i, :])
#             print(self.pos[i, :])
#             self.v[i, :] = r.v
#             self.a[i, :] = r.a
#             self.bt[i, :] = r.bt
#             self.lt[i, :] = r.lt


#         self.total_time = np.sum(dt)
#         self.t = np.zeros(len(dt) + 1)
#         self.t[1:] = np.cumsum(dt)

#     LSPBResult = namedtuple('LSPBResult', 'v, a, bt, lt')
#     def lspb(self, pos, dt, acc):
#         n = len(pos)

#         v = np.zeros(n-1, dtype=float)
#         a = np.zeros(n, dtype=float)
#         bt = np.zeros(n, dtype=float)
#         lt = np.zeros(n-1, dtype=float)


#         dpos = np.diff(pos)
#         aa = np.abs(acc)

#         # First segment
#         a[0] = np.sign(dpos[0]) * aa[0]
#         if a[0] != 0.:
#             bt[0] = dt[0] - math.sqrt(dt[0]**2 - (2. * dpos[0]) / a[0]) # what if a == 0?            
#         v[0] = dpos[0] / (dt[0] - 0.5 * bt[0])

#         # Last segment
#         a[-1] = np.sign(-dpos[-1]) * aa[-1]
#         if a[-1] != 0.:
#             bt[-1] = dt[-1] - math.sqrt(dt[-1]**2 + (2. * dpos[-1]) / a[-1])
#         v[-1] = dpos[-1] / (dt[-1] - 0.5 * bt[-1])

#         # Inner segments
#         v[1:-1] = dpos[1:-1] / dt[1:-1]
#         a[1:-1] = np.sign(np.diff(v)) * aa[1:-1]

#         with np.errstate(all='ignore'):
#             mask = a[1:-1] == 0.
#             bt[1:-1] = np.select([mask, ~mask], [0., np.diff(v) / a[1:-1]]) * 0.5

#         # Linear timings
#         lt[:] = dt - bt[:-1] - bt[1:]

#         return LSPBTrajectory.LSPBResult(v=v, a=a, bt=bt, lt=lt)


#     def __call__(self, t):
#         torig = t
#         t = np.atleast_1d(t)
#         assert (t >= 0.).all() and (t <= self.total_time).all()

#         # Find interval associated with t
        
#         #https://github.com/jhu-cisst/cisst/blob/2e90a6b69ac141a0ac386a933ad53d83327d448b/cisstRobot/code/robLSPB.cpp
#         #http://www.ee.nmt.edu/~wedeward/EE570/SP09/gentraj.html
#         i = np.digitize(t, self.t) - 1
#         tf = self.t[i+1] - self.t[i]
#         t = t - self.t[i]
        
#         # Only one of the following will be true for each axis
#         isacc = t < self.bt[:, i]
#         isdeacc = t >= (tf - self.bt[:, i+1])
#         islinear = np.logical_and(~isacc, ~isdeacc)

#         p = (self.pos[:, i] + 0.5 * self.a[:, i] * t**2) * isacc + \
#             (self.pos[:, i] + self.v[:, i] * (t - self.bt[:, i]/2)) * islinear + \
#             (self.pos[:, i+1] + self.a[:, i+1] * (tf-t)**2) * isdeacc

#         """
#         acct = np.minimum(t, self.bt[:, i])
#         lint = np.clip(t - self.bt[:, i], 0., self.lt[:, i])
#         dacct = np.clip(t - (tf - self.bt[:, i+1]), 0., self.bt[:, i+1])

#         # http://www-lar.deis.unibo.it/people/cmelchiorri/Files_Robotica/FIR_07_Traj_1.pdf
 
#         p = self.pos[:, i] + \
#             0.5 * self.a[:, i] * acct**2 + \
#             self.v[:, i] * lint + \
#             self.v[:, i] * dacct + 0.5 * self.a[:, i+1] * dacct**2
#         """
#         return p

# """
# traj = LSPBTrajectory(np.array([
#     [10, 35, 25, 10],
#     [0, 0, 0, 0]
# ]), [2, 1, 3], 50)
# """

# traj = LSPBTrajectory(np.array([10, 35, 25, 10]), [2, 1, 3], 50)


# import matplotlib.pyplot as plt
# t = np.arange(0, 2, 0.01)
# p = traj(t)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(t, p[0])
# ax2.plot(t[1:], np.diff(p[0]) * 0.01)
# #ax2.plot(t[2:], np.diff(p[0], 2) / 0.01**2)
# plt.show()