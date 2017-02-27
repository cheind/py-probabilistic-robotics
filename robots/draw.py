import math
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection
import matplotlib.transforms as mplt
from robots import transforms

class BaseDrawer:
    def __init__(self):
        self.items = {}
        self.nextkey = 0

    def genkey(self):
        k = self.nextkey
        self.nextkey += 1
        return k

class Drawer(BaseDrawer):

    def draw_robot(self, robot, ax, **kwargs):
        key = kwargs.pop('key', self.genkey())

        radius = kwargs.pop('radius', 0.5)
        fc = kwargs.pop('fc', 'None')
        ec = kwargs.pop('ec', 'k')
        with_axis = kwargs.pop('with_axis', True)
        with_circle = kwargs.pop('with_circle', True)
        zorder = kwargs.pop('zorder', 2)
        
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

        tr = mplt.Affine2D(matrix=robot.transform_to_world) + ax.transData

        if with_circle:            
            d['c'].set_radius(radius)
            d['c'].set_zorder(zorder)
            d['c'].set_transform(tr)
            updated.append(d['c'])

        if with_axis:
            d['lx'].set_xdata([0., radius])
            d['lx'].set_ydata([0., 0])
            d['lx'].set_zorder(zorder)
            d['lx'].set_transform(tr)

            d['ly'].set_xdata([0., 0])
            d['ly'].set_ydata([0., radius])
            d['ly'].set_zorder(zorder)
            d['ly'].set_transform(tr)

            updated.append(d['lx'])
            updated.append(d['ly'])

        return updated

    def draw_points(self, points, ax, **kwargs):
        key = kwargs.pop('key', self.genkey())
        
        size = kwargs.pop('size', 80)
        fc = kwargs.pop('fc', 'b')
        ec = kwargs.pop('ec', 'none')
        with_labels = kwargs.pop('with_labels', False)
        marker = kwargs.pop('marker', (5, 1))
        zorder = kwargs.pop('zorder', 4)
        t = kwargs.pop('transform', None)

        if t is not None:
            points = transforms.transform(t, points, hvalue=1.)

        if (ax, key) not in self.items:
            scat = ax.scatter([], [], s=size, edgecolors=ec, facecolors=fc, zorder=zorder, marker=marker)                   
            self.items[(ax, key)] = dict(scatter=scat)

        updated=[]
        
        d = self.items[(ax, key)]
        scat = d['scatter']
        scat.set_offsets(points.T)
        scat.set_zorder(zorder)
        scat.set_facecolors(fc)
        scat.set_edgecolors(ec)

        updated.append(scat)

        if with_labels:
            if not 'ann' in d:
                d['ann'] = [ax.annotate(i, xy=(landmarks[0,i], landmarks[1,i])) for i in range(landmarks.shape[1])]                    
                updated.extend(d['ann'])
            else:
                ann = d['ann']
                for i,a in enumerate(ann):
                    a.set_position((landmarks[0,i], landmarks[1,i]))
                updated.extend(ann)
        return updated
        

    def draw_sensor(self, sensor, ax, **kwargs):
        key = kwargs.pop('key', self.genkey())
        fc = kwargs.pop('fc', 'r')
        ec = kwargs.pop('ec', 'r')
        zorder = kwargs.pop('zorder', 3)

        if (ax, key) not in self.items:
            w = Wedge((0,0), min(sensor.maxdist, 100), -math.degrees(sensor.fov/2), math.degrees(sensor.fov/2), fc=fc, ec=ec, alpha=0.5, zorder=zorder)
            ax.add_artist(w)
            self.items[(ax, key)] = dict(w=w)

        d = self.items[(ax, key)]

        tr = mplt.Affine2D(matrix=sensor.transform_to_world) + ax.transData
        d['w'].set_transform(tr)
        
        return d['w'],


    def draw_grid(self, grid, ax, **kwargs):
        key = kwargs.pop('key', self.genkey())
        cmap = kwargs.pop('cmap', 'gray_r')
        interp = kwargs.pop('interpolation', 'none')
        zorder = kwargs.pop('zorder', 1)
        alpha = kwargs.pop('alpha', 1)

        if (ax, key) not in self.items:
            bbox = grid.bbox
            im = ax.imshow(
                grid.values, 
                origin='lower', 
                interpolation=interp, 
                alpha=alpha, 
                cmap=cmap, 
                extent=[bbox.mincorner[0], bbox.maxcorner[0], bbox.mincorner[1], bbox.maxcorner[1]], 
                zorder=zorder)                
            self.items[(ax, key)] = dict(im=im)

        d = self.items[(ax, key)]

        tr = mplt.Affine2D(matrix=grid.transform_to_world) + ax.transData
        d['im'].set_data(grid.values)
        d['im'].set_transform(tr)

        return d['im'],