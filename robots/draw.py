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
    """Base class for drawing utils.
    
    Each drawer holds a map from keys to artists to be drawn. 
    This allows effective redrawing by reusing keys instead of regenerating
    the artists every time frame.

    Keys consist of two hashable elements:
      - the canvas - matplotlib axis element
      - identifier - a hashable element that identifies the element to be drawn.
    """

    def __init__(self):
        self.items = {}

    def keyfor(self, *objs):
        hashable = []
        for obj in objs:
            if isinstance(obj, (PoseNode, _AxesBase)):
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

    def make_transform(self, m, ax):
        if m is None:
            return ax.transData
        elif isinstance(m, np.ndarray):
            if m.shape == (3,3):
                return mplt.Affine2D(matrix=m) + ax.transData
            else:
                return mplt.Affine2D(matrix=transforms.transform_from_pose(m)) + ax.transData
        elif isinstance(m, mplt.Affine2DBase):
            return m + ax.transdata
        else:
            raise TypeError('Unknown transform type.')

class Drawer(BaseDrawer):
    """Default drawing util.

    Provides high level drawing commands for various classes inheriting PoseNode 
    and low level routines for drawing basic items.
    """
    
    class RobotArtists(namedtuple('RobotArtists', 'circle, linex, liney')):
        """Artists associated with a robot draw command."""
        pass

    def draw_robot(self, robot, ax, **kwargs):
        """Draw or update robot.

        A robot is drawn using a circle and a xy-frame to determine its orientation.
        """

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
        t = self.make_transform(robot.transform_to_world, ax)
         
        artists.circle.set_radius(radius)
        artists.circle.set_zorder(zorder)
        artists.circle.set_transform(t)

        artists.linex.set_xdata([0., radius])
        artists.linex.set_ydata([0., 0])
        artists.linex.set_zorder(zorder)
        artists.linex.set_transform(t)

        artists.liney.set_xdata([0., 0])
        artists.liney.set_ydata([0., radius])
        artists.liney.set_zorder(zorder)
        artists.liney.set_transform(t)

        return artists

    class SensorArtists(namedtuple('SensorArtists', 'wedge')):
        """Artists associated with a sensor draw command."""
        pass

    def draw_sensor(self, sensor, ax, **kwargs):
        """Draw or update a sensor and its field of view.

        A sensor is drawn as a wedge indicating its field of view.
        """
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
        artists.wedge.set_transform(self.make_transform(sensor.transform_to_world, ax))
        
        return artists

    class GridArtists(namedtuple('GridArtists', 'image')):
        """Artists associated with a grid draw command."""
        pass

    def draw_grid(self, grid, ax, **kwargs):
        """Draw or update a Grid node."""        
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
        artists.image.set_data(grid.values)
        # The following requires at least matplotlib 2.x / qt4.8 for rotated grids to show correctly.
        artists.image.set_transform(self.make_transform(grid.transform_to_world, ax))

        return artists

    class PointArtists(namedtuple('PointArtists', 'scatter')):
        """Artists associated with a point list draw command."""
        pass

    def draw_points(self, points, ax, **kwargs):
        """Draw or update a set of 2D points."""
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
        artists.scatter.set_offsets(points)
        artists.scatter.set_zorder(zorder)
        artists.scatter.set_facecolors(fc)
        artists.scatter.set_edgecolors(ec)
        return artists


    class LineArtists(namedtuple('LineArtists', 'lines')):
        """Artists associated with a line draw command."""
        pass

    def draw_lines(self, segments, ax, **kwargs):
        """Draw or update a set of 2D lines."""
        key = (ax, kwargs.pop('key', self.keyfor(segments)))
        ec = kwargs.pop('ec', 'k')        
        t = kwargs.pop('transform', None)
        zorder = kwargs.pop('zorder', 1)

        def create_artists():
            artists = Drawer.LineArtists(lines=LineCollection([]))
            ax.add_collection(artists.lines)
            return artists
       
        artists = self.get_artists(key, create_artists)
        artists.lines.set_segments(segments)
        artists.lines.set_edgecolors(ec)
        artists.lines.set_transform(self.make_transform(t, ax))
        return artists

    class ConfidenceEllipseArtists(namedtuple('ConfidenceEllipseArtists', 'ellipses')):
        """Artists associated with a ellipse draw command."""
        pass

    def draw_confidence_ellipses(self, u, cov, ax, **kwargs):
        """Draw or update a set of confidence ellipses."""
        key = (ax, kwargs.pop('key', self.keyfor(u, cov)))
        fc = kwargs.pop('fc', 'none')
        ec = kwargs.pop('ec', 'k')        
        zorder = kwargs.pop('zorder', 1)
        t = kwargs.pop('transform', None)

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
                alpha=0.5,
                transform=self.make_transform(t, ax)
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
