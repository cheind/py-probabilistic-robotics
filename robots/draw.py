import math
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection, EllipseCollection
from matplotlib.axes._axes import _AxesBase
import matplotlib.transforms as mplt


from robots import transforms
from robots.posenode import PoseNode

class BaseDrawer:
    def __init__(self):
        self.items = {}
        self.nextkey = 0

    def keyfor(self, *objs):
        hashable = []
        for obj in objs:
            if isinstance(obj, (PoseNode, np.ndarray, _AxesBase)):
                hashable.append(id(obj))
            else:
                k = self.nextkey
                hashable.append(k)
                self.nextkey += 1
        return tuple(hashable)

class Drawer(BaseDrawer):

    def draw_robot(self, robot, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(robot)))

        radius = kwargs.pop('radius', 0.5)
        fc = kwargs.pop('fc', 'None')
        ec = kwargs.pop('ec', 'k')
        with_axis = kwargs.pop('with_axis', True)
        with_circle = kwargs.pop('with_circle', True)
        zorder = kwargs.pop('zorder', 2)
        
        if key not in self.items:
            c = Circle((0,0), radius=radius, fc=fc, ec=ec, zorder=zorder)
            lx = Line2D((0,0),(0,0), color='r', zorder=zorder)
            ly = Line2D((0,0),(0,0), color='g', zorder=zorder)
            ax.add_artist(c)
            ax.add_artist(lx)
            ax.add_artist(ly)
            self.items[key] = dict(c=c, lx=lx, ly=ly)

        updated = []
        d = self.items[key]

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

    def draw_sensor(self, sensor, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(sensor)))

        fc = kwargs.pop('fc', 'r')
        ec = kwargs.pop('ec', 'r')
        zorder = kwargs.pop('zorder', 3)

        if key not in self.items:
            w = Wedge((0,0), min(sensor.maxdist, 100), -math.degrees(sensor.fov/2), math.degrees(sensor.fov/2), fc=fc, ec=ec, alpha=0.5, zorder=zorder)
            ax.add_artist(w)
            self.items[key] = dict(w=w)

        d = self.items[key]

        tr = mplt.Affine2D(matrix=sensor.transform_to_world) + ax.transData
        d['w'].set_transform(tr)
        
        return d['w'],


    def draw_grid(self, grid, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(grid)))

        cmap = kwargs.pop('cmap', 'gray_r')
        interp = kwargs.pop('interpolation', 'none')
        zorder = kwargs.pop('zorder', 1)
        alpha = kwargs.pop('alpha', 1)

        if key not in self.items:
            bbox = grid.bbox
            im = ax.imshow(
                grid.values, 
                origin='lower', 
                interpolation=interp, 
                alpha=alpha, 
                cmap=cmap, 
                extent=[bbox.mincorner[0], bbox.maxcorner[0], bbox.mincorner[1], bbox.maxcorner[1]], 
                zorder=zorder)                
            self.items[key] = dict(im=im)

        d = self.items[key]

        # The following requires at least matplotlib 2.x / qt4.8 for rotated grids to show correctly.
        tr = mplt.Affine2D(matrix=grid.transform_to_world) + ax.transData 
        d['im'].set_data(grid.values)
        d['im'].set_transform(tr)

        return d['im'],

    def draw_points(self, points, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(points)))
        
        size = kwargs.pop('size', 80)
        fc = kwargs.pop('fc', 'b')
        ec = kwargs.pop('ec', 'none')
        with_labels = kwargs.pop('with_labels', False)
        marker = kwargs.pop('marker', (5, 1))
        zorder = kwargs.pop('zorder', 4)
        t = kwargs.pop('transform', None)

        if t is not None:
            points = transforms.transform(t, points, hvalue=1.)

        if key not in self.items:
            scat = ax.scatter([], [], s=size, edgecolors=ec, facecolors=fc, zorder=zorder, marker=marker)                   
            self.items[key] = dict(scatter=scat)

        updated=[]
        
        d = self.items[key]
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

    def draw_line(self, segments, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(segments)))
        ec = kwargs.pop('ec', 'k')        
        t = kwargs.pop('transform', None)
        zorder = kwargs.pop('zorder', 1)

        if key not in self.items:
            l = LineCollection(segments)
            ax.add_collection(l)
            self.items[key] = dict(lines=l)

        d = self.items[key]
        l = d['lines']

        l.set_segments(segments)
        l.set_edgecolors(ec)
        if t is not None:
            l.set_transform(mplt.Affine2D(matrix=t) + ax.transData)

        # u += drawer.draw_line(
        #     np.array([
        #         [0,10],
        #         [0,10]
        #     ]).T.reshape(1,2,2),
        #     ax
        # )
        
        return l,

    def draw_confidence_ellipse(self, u, cov, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(u, cov)))
        fc = kwargs.pop('fc', 'none')
        ec = kwargs.pop('ec', 'k')        
        zorder = kwargs.pop('zorder', 1)

        chisquare_val = kwargs.pop('scale', 5.991) # 95% confidence area based on chi2(2dof, 0.05)
        
        # http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        # https://people.richland.edu/james/lecture/m170/tbl-chi.html

        if key in self.items:
            self.items[key].remove()

        u = np.asarray(u)
        cov = np.asarray(cov).reshape(-1, 2, 2)
        widths, heights, angles = self._compute_ellipse_parameters(cov, chisquare_val)        
        
        e = EllipseCollection(
            widths, 
            heights, 
            np.degrees(angles),
            units='y', 
            offsets=u.reshape(2,-1), 
            transOffset=ax.transData,
            facecolors=fc,
            edgecolors=ec,
            zorder=zorder,
            alpha=0.5
        )
        ax.add_collection(e)

        self.items[key] = e

        return e,

        


    def _compute_ellipse_parameters(self, cov, chi_square=5.991):
        cov = np.asarray(cov).reshape(-1, 2, 2)
        n = cov.shape[0]
        w, v = np.linalg.eig(cov)
        w = np.abs(w)
        major = np.argmax(w, axis=1)
        minor = (major + 1) % 2
    
        angles = np.zeros(n)
        widths = np.zeros(n)
        heights = np.zeros(n)
        for i in range(n):
            angles[i] = math.atan2(v[i, 1, major[i]], v[i, 0, major[i]])
            widths[i] = 2 * math.sqrt(w[i, major[i]] * chi_square)
            heights[i] = 2 * math.sqrt(w[i, minor[i]] * chi_square)

        angles[angles < 0.] += 2 * np.pi # -pi..pi -> 0..2pi
        return widths, heights, angles
