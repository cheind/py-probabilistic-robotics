
import numpy as np
import math

class Robot:
    def __init__(self, **kwargs):
        self.state = np.asarray(kwargs.pop('state', np.array([0.,0.,0.])))
        self.motion_err = np.asarray(kwargs.pop('err', np.array([0., 0.])))

    def move(self, motion, **kwargs):
        """Move robot by first turning and then driving along the robot's x-axis."""
        motion_err = kwargs.pop('err', self.motion_err)
        
        sigma = motion_err * np.abs(motion)
        e =  np.where(sigma == 0., 0., np.random.randn() * sigma)

        phi = self.state[2] + motion[0] + e[0]
        self.state[0] += math.cos(phi) * (motion[1] + e[1])
        self.state[1] += math.sin(phi) * (motion[1] + e[1])
        self.state[2] = phi
        
    def robot_in_world(self, **kwargs):
        """Returns a 2x3 matrix representing the robot's location in world space."""
        hom = kwargs.pop('hom', False)

        c = math.cos(self.state[2])
        s = math.sin(self.state[2])

        t = np.array([
            [c, -s, self.state[0]],
            [s, c, self.state[1]]
        ])

        if hom:
            m = np.eye(3)
            m[:2, :] = t
            return m
        else:
            return t

    def world_in_robot(self):
        """Returns a 2x3 matrix representing the world in robot space."""
        hom = kwargs.pop('hom', False)

        c = math.cos(self.state[2])
        s = math.sin(self.state[2])

        t = np.array([
            [c, s, -(c * self.state[0] + s * self.state[1])],
            [-s, c, -(-s * self.state[0] + c * self.state[1])]
        ])

        if hom:
            m = np.eye(3)
            m[:2, :] = t
            return m
        else:
            return t



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from draw import Drawer

    fig, ax = plt.subplots()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect('equal')

    rideal = Robot()
    rreal = Robot(err=[1., 0.1])
    d = Drawer()

    def update(i):    
        rideal.move([0.05, 0.1])
        rreal.move([0.05, 0.1])
        return d.draw_robot(rideal, ax) + d.draw_robot(rreal, ax)


    ani = animation.FuncAnimation(fig, update, 25, interval=50, blit=True)
    plt.show()

