import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

class Drawer:
    def __init__(self):
        self.items = {}

    def draw_robot(self, r, ax, **kwargs):
        radius = kwargs.pop('radius', 0.5)
        fc = kwargs.pop('fc', 'None')
        ec = kwargs.pop('ec', 'k')
        with_axis = kwargs.pop('with_axis', True)
        with_circle = kwargs.pop('with_circle', True)

        if (ax, r) not in self.items:
            c = Circle((0,0), radius=radius, fc=fc, ec=ec)
            lx = Line2D((0,0),(0,0), color='r')
            ly = Line2D((0,0),(0,0), color='g')
            ax.add_artist(c)
            ax.add_artist(lx)
            ax.add_artist(ly)
            self.items[(ax, r)] = dict(c=c, lx=lx, ly=ly)

        updated = []

        if with_circle:
            d = self.items[(ax, r)]
            d['c'].set_radius(radius)
            d['c'].center = r.state[:2]
            updated.append(d['c'])

        if with_axis:
            m = r.robot_in_world()
            o = np.dot(m, [0,0,1])
            x = np.dot(m, [radius,0,1])
            y = np.dot(m, [0,radius,1])

            d['lx'].set_xdata([o[0], x[0]])
            d['lx'].set_ydata([o[1], x[1]])
            d['ly'].set_xdata([o[0], y[0]])
            d['ly'].set_ydata([o[1], y[1]])

            updated.append(d['lx'])
            updated.append(d['ly'])

        return updated
