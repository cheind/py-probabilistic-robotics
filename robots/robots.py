
import numpy as np
import math

from robots.posenode import PoseNode


class XYPhiRobot(PoseNode):
    """A robot determined by its position in x,y and heading phi."""

    def __init__(self, **kwargs):        
        """Create a XYPhiRobot.

        Kwargs
            err : 1x2 vector describing the error associated with motion. The error model assumes
                  a zero centered normal distribution having a standard deviation proportional to
                  [error per radians, error per meter].
            pose : (optional) 1x3 pose vector. If omitted identity is assumed.
        """
        self.motion_err = np.array(kwargs.pop('err', [0., 0.]), dtype=float)

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(XYPhiRobot, self).__init__(pose=pose)

    def move(self, motion, **kwargs):
        """Move robot by first turning and then driving along the robot's new heading (x-axis).

        Params
            motion : 1x2 motion vector [angle in radians, meter]          
              
        Kwargs
            err : 1x2 vector describing the error associated with motion. The error model assumes
                  a zero centered normal distribution having a standard deviation proportional to
                  [error per radians, error per meter].
        """
        motion_err = kwargs.pop('err', self.motion_err)
        
        sigma = motion_err * np.abs(motion)
        e = np.random.randn(2)
        e *= sigma

        phi = self.pose[2] + motion[0] + e[0]
        self.pose[0] += math.cos(phi) * (motion[1] + e[1])
        self.pose[1] += math.sin(phi) * (motion[1] + e[1])
        self.pose[2] = phi
        

