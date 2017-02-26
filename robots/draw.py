import math
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
import matplotlib.transforms as mplt

class BaseDrawer:
    def __init__(self):
        self.items = {}
        self.nextkey = 0

    def genkey(self):
        k = self.nextkey
        self.nextkey += 1
        return k

class RobotDrawer(BaseDrawer):

    def draw(self, robot, ax, **kwargs):
        key = kwargs.pop('key', self.genkey())

        radius = kwargs.pop('radius', 0.5)
        fc = kwargs.pop('fc', 'None')
        ec = kwargs.pop('ec', 'k')
        with_axis = kwargs.pop('with_axis', True)
        with_circle = kwargs.pop('with_circle', True)
        zorder = kwargs.pop('zorder', 1)
        
        if (ax, key) not in self.items:
            c = Circle((0,0), radius=radius, fc=fc, ec=ec, zorder=zorder)
            lx = Line2D((0,0),(0,0), color='r', zorder=zorder)
            ly = Line2D((0,0),(0,0), color='g', zorder=zorder)
            ax.add_artist(c)
            ax.add_artist(lx)
            ax.add_artist(ly)
            self.items[(ax, key)] = dict(c=c, lx=lx, ly=ly)

        updated = []
        d = self.items[(ax, key)]

        if with_circle:            
            d['c'].set_radius(radius)
            d['c'].center = robot.state[:2]
            d['c'].set_zorder(zorder)
            updated.append(d['c'])

        if with_axis:
            m = robot.robot_in_world()
            points = np.array([
                [0, radius, 0],
                [0, 0, radius],
                [1, 1, 1]
            ])
            p = np.dot(m, points)

            d['lx'].set_xdata([p[0, 0], p[0, 1]])
            d['lx'].set_ydata([p[1, 0], p[1, 1]])
            d['lx'].set_zorder(zorder)

            d['ly'].set_xdata([p[0, 0], p[0, 2]])
            d['ly'].set_ydata([p[1, 0], p[1, 2]])
            d['ly'].set_zorder(zorder)

            updated.append(d['lx'])
            updated.append(d['ly'])

        return updated

class LandmarkDrawer(BaseDrawer):
    def draw(self, landmarks, ax, **kwargs):
        key = kwargs.pop('key', self.genkey())
        
        size = kwargs.pop('size', 50)
        fc = kwargs.pop('fc', 'b')
        ec = kwargs.pop('ec', 'none')
        zorder = kwargs.pop('zorder', 3)

        if (ax, key) not in self.items:
            scat = ax.scatter(landmarks[0], landmarks[1], s=size, edgecolors=ec, facecolors=fc, zorder=zorder, marker=(5, 1))
            self.items[(ax, key)] = dict(scatter=scat)

        scat = self.items[(ax, key)]['scatter']
        scat.set_offsets(landmarks.T)
        scat.set_zorder(zorder)
        scat.set_facecolors(fc)
        scat.set_edgecolors(ec)
        return scat,

class LandmarkSensorDrawer(BaseDrawer):

    def draw(self, robot, sensor, ax, **kwargs):
        key = kwargs.pop('key', self.genkey())
        fc = kwargs.pop('fc', 'r')
        ec = kwargs.pop('ec', 'r')
        zorder = kwargs.pop('zorder', 2)

        if (ax, key) not in self.items:
            w = Wedge((0,0), min(sensor.maxdist, 100), -math.degrees(sensor.fov/2), math.degrees(sensor.fov/2), fc=fc, ec=ec, alpha=0.5, zorder=zorder)
            ax.add_artist(w)
            self.items[(ax, key)] = dict(w=w)

        d = self.items[(ax, key)]

        mtx = robot.robot_in_world()
        tr = mplt.Affine2D(matrix=mtx) + ax.transData
        d['w'].set_transform(tr)
        
        return d['w'],