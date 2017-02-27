
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

    def __init__(self, **kwargs):        
        self.motion_err = np.array(kwargs.pop('err', [0., 0.]), dtype=float)

        pose = np.array(kwargs.pop('pose', [0.,0.,0.]), dtype=float)
        super(XYPhiRobot, self).__init__(pose=pose)

    def move(self, motion, **kwargs):
        """Move robot by first turning and then driving along the robot's x-axis."""
        motion_err = kwargs.pop('err', self.motion_err)
        
        sigma = motion_err * np.abs(motion)
        e = np.random.randn(2)
        e *= sigma

        phi = self.pose[2] + motion[0] + e[0]
        self.pose[0] += math.cos(phi) * (motion[1] + e[1])
        self.pose[1] += math.sin(phi) * (motion[1] + e[1])
        self.pose[2] = phi
        

