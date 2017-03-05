
import numpy as np
import math

from robots.posenode import PoseNode

class XYRobot(PoseNode):

    def __init__(self, **kwargs):
        self.motion_err = kwargs.pop('err', 0.)

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(XYRobot, self).__init__(pose=pose)

    def move(self, motion, **kwargs):
        """Move robot by first turning and then driving along the robot's x-axis."""
        motion_err = kwargs.pop('err', self.motion_err)
        sigma = np.abs(motion) * motion_err
        e = np.expand_dims(np.random.randn(2), 1)
        e *= sigma

        self.pose[0] += motion[0] + e[0]
        self.pose[1] += motion[1] + e[1]

class XYPhiRobot(PoseNode):
    """A robot fully described by its position in x,y and heading phi.

    Attributes
    ----------
    motion_err : 1x2 array
        The error associated with a single unit of motion (turn, drive).
    """

    def __init__(self, **kwargs):        
        """Create a XYPhiRobot.

        Kwargs
        ------
        err : 1x2 array, optional
            Array describing the error associated with motion. The error model assumes
            a zero centered normal distribution having a standard deviation proportional to
            [error per radians, error per meter].
        pose : 1x3 array, optional
            1x3 pose vector. If omitted identity is assumed.
        """
        self.motion_err = np.array(kwargs.pop('err', [0., 0.]), dtype=float)

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(XYPhiRobot, self).__init__(pose=pose)
        
        self.right = True # used for angle representation

    def _normalized_phi(self, phi):
        return phi
        if self.right:
            if phi > 3 * np.pi / 4 or phi < -3 * np.pi / 4:
                phi = np.arctan2(np.sin(phi - np.pi), np.cos(phi - np.pi)) + np.pi # phi in [0..2pi]
                self.right = False
        elif phi < np.pi / 4 or phi > -7 * np.pi / 4:
                phi = np.arctan2(np.sin(phi), np.cos(phi))  # phi in [-pi, pi]
                self.right = True
        return phi

    def move(self, motion, **kwargs):
        """Move robot by first turning and then driving along the robot's new heading (x-axis).

        Params
        ------
        motion : 1x2 array
            Motion command [angle in radians, meter]        
              
        Kwargs
        ------
        err : 1x2 array, optional 
            Vector describing the error associated with motion. The error model assumes
            a zero centered normal distribution having a standard deviation proportional to
            [error per radians, error per meter].
        """
        motion_err = kwargs.pop('err', self.motion_err)
        
        sigma = motion_err * np.abs(motion)
        e = np.random.randn(2)
        e *= sigma

        phi = self.pose[2] + motion[0] + e[0]
        phi = np.arctan2(np.sin(phi), np.cos(phi))  # phi in [-pi, pi]
        #print('new phi robot ', phi)
        self.pose[0] += math.cos(phi) * (motion[1] + e[1])
        self.pose[1] += math.sin(phi) * (motion[1] + e[1])
        self.pose[2] = phi#self._normalized_phi(phi)
