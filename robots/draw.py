import math
import numpy as np



from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection, EllipseCollection
from matplotlib.axes._axes import _AxesBase
import matplotlib.transforms as mplt

from collections import namedtuple

from robots import transforms
from robots.posenode import PoseNode

class BaseDrawer:
    def __init__(self):
        self.items = {}

    def keyfor(self, *objs):
        hashable = []
        for obj in objs:
            if isinstance(obj, (PoseNode, np.ndarray, _AxesBase)):
                hashable.append(id(obj))
            else:
                import uuid    
                hashable.append(uuid.uuid4())  
        return tuple(hashable)

    def get_artists(self, key, creator):
        if not key in self.items:
            artists = creator()
            self.items[key] = artists
        else:
            artists = self.items[key]
        
        return artists

class Drawer(BaseDrawer):
    
    RobotArtists = namedtuple('RobotArtists', 'circle, linex, liney')

    def draw_robot(self, robot, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(robot)))

        radius = kwargs.pop('radius', 0.5)
        fc = kwargs.pop('fc', 'None')
        ec = kwargs.pop('ec', 'k')
        zorder = kwargs.pop('zorder', 2)

        def create_artists():
            artists = Drawer.RobotArtists(
                circle=Circle((0,0), radius=radius, fc=fc, ec=ec, zorder=zorder),
                linex=Line2D([], [], color='r', zorder=zorder),
                liney=Line2D([], [], color='g', zorder=zorder)
            )
            for a in artists:
                ax.add_artist(a)
            return artists
        
        artists = self.get_artists(key, create_artists)
        tr = mplt.Affine2D(matrix=robot.transform_to_world) + ax.transData
         
        artists.circle.set_radius(radius)
        artists.circle.set_zorder(zorder)
        artists.circle.set_transform(tr)

        artists.linex.set_xdata([0., radius])
        artists.linex.set_ydata([0., 0])
        artists.linex.set_zorder(zorder)
        artists.linex.set_transform(tr)

        artists.liney.set_xdata([0., 0])
        artists.liney.set_ydata([0., radius])
        artists.liney.set_zorder(zorder)
        artists.liney.set_transform(tr)

        return artists

    SensorArtists = namedtuple('SensorArtists', 'wedge')

    def draw_sensor(self, sensor, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(sensor)))

        fc = kwargs.pop('fc', 'r')
        ec = kwargs.pop('ec', 'r')
        zorder = kwargs.pop('zorder', 3)

        def create_artists():
            artists = Drawer.SensorArtists(
                wedge=Wedge(
                    (0,0), 
                    min(sensor.maxdist, 1000), 
                    -math.degrees(sensor.fov/2), 
                    math.degrees(sensor.fov/2), 
                    fc=fc, ec=ec, alpha=0.5, zorder=zorder)
            )
            for a in artists:
                ax.add_artist(a)
            return artists

        artists = self.get_artists(key, create_artists)
        tr = mplt.Affine2D(matrix=sensor.transform_to_world) + ax.transData
        artists.wedge.set_transform(tr)
        
        return artists

    GridArtists = namedtuple('GridArtists', 'image')

    def draw_grid(self, grid, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(grid)))

        cmap = kwargs.pop('cmap', 'gray_r')
        interp = kwargs.pop('interpolation', 'none')
        zorder = kwargs.pop('zorder', 1)
        alpha = kwargs.pop('alpha', 1)

        def create_artists():
            artists = Drawer.GridArtists(
                image=ax.imshow(
                    grid.values, 
                    origin='lower', 
                    interpolation=interp, 
                    alpha=alpha, 
                    cmap=cmap, 
                    extent=[grid.bbox.mincorner[0], grid.bbox.maxcorner[0], grid.bbox.mincorner[1], grid.bbox.maxcorner[1]], 
                    zorder=zorder)
            )
            return artists

        artists = self.get_artists(key, create_artists)

        # The following requires at least matplotlib 2.x / qt4.8 for rotated grids to show correctly.
        tr = mplt.Affine2D(matrix=grid.transform_to_world) + ax.transData 
        artists.image.set_data(grid.values)
        artists.image.set_transform(tr)

        return artists

    PointArtists = namedtuple('PointArtists', 'scatter')

    def draw_points(self, points, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(points)))
        
        size = kwargs.pop('size', 80)
        fc = kwargs.pop('fc', 'b')
        ec = kwargs.pop('ec', 'none')
        marker = kwargs.pop('marker', (5, 1))
        zorder = kwargs.pop('zorder', 4)
        t = kwargs.pop('transform', None)

        def create_artists():
            artists = Drawer.PointArtists(
                scatter=ax.scatter([], [], s=size, edgecolors=ec, facecolors=fc, zorder=zorder, marker=marker)
            )
            return artists
        
        if t is not None:
            points = transforms.transform(t, points, hvalue=1.)

        artists = self.get_artists(key, create_artists)
        artists.scatter.set_offsets(points.T)
        artists.scatter.set_zorder(zorder)
        artists.scatter.set_facecolors(fc)
        artists.scatter.set_edgecolors(ec)
        return artists


    LineArtists = namedtuple('LineArtists', 'lines')
    def draw_lines(self, segments, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(segments)))
        ec = kwargs.pop('ec', 'k')        
        t = kwargs.pop('transform', None)
        zorder = kwargs.pop('zorder', 1)

        def create_artists():
            artists = Drawer.LineArtists(lines=LineCollection([]))
            ax.add_collection(artists.lines)
            return artists

        transposed_lines = []
        for l in segments:
            transposed_lines.append(l.T)
        
        artists = self.get_artists(key, create_artists)
        artists.lines.set_segments(transposed_lines)
        artists.lines.set_edgecolors(ec)
        return artists

    ConfidenceEllipseArtists = namedtuple('ConfidenceEllipseArtists', 'ellipses')

    def draw_confidence_ellipses(self, u, cov, ax, **kwargs):
        key = (ax, kwargs.pop('key', self.keyfor(u, cov)))
        fc = kwargs.pop('fc', 'none')
        ec = kwargs.pop('ec', 'k')        
        zorder = kwargs.pop('zorder', 1)

        chisquare_val = kwargs.pop('scale', 5.991) # 95% confidence area based on chi2(2dof, 0.05)
        
        # http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
        # https://people.richland.edu/james/lecture/m170/tbl-chi.html

        if key in self.items:
            artists = self.get_artists(key, None)
            for a in artists:
                a.remove()

        u = np.asarray(u)
        cov = np.asarray(cov).reshape(-1, 2, 2)
        widths, heights, angles = self._compute_ellipse_parameters(cov, chisquare_val)        
                
        artists = Drawer.ConfidenceEllipseArtists(
            ellipses=EllipseCollection(
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
        )
        for a in artists:
            ax.add_artist(a)
        
        self.items[key] = artists
        return artists

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
