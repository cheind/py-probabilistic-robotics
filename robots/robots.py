
import numpy as np
import math

class RobotBase:

    @property 
    def pose(self):
        """Return the pose vector of the robot."""
        raise NotImplementedError

    def robot_in_world(self):
        """Returns a 3x3 matrix representing the robot's location in world space."""

        pose = self.pose
        c = math.cos(pose[2])
        s = math.sin(pose[2])

        return np.array([
            [c, -s, pose[0]],
            [s, c, pose[1]],
            [0, 0., 1]
        ])

    def world_in_robot(self, **kwargs):
        """Returns a 3x3 matrix representing the world in robot space."""
        
        pose = self.pose

        c = math.cos(pose[2])
        s = math.sin(pose[2])

        return np.array([
            [c, s, -(c * pose[0] + s * pose[1])],
            [-s, c, -(-s * pose[0] + c * pose[1])],
            [0, 0., 1]
        ])
    


class XYPhiRobot(RobotBase):

    def __init__(self, **kwargs):
        self.state = np.array(kwargs.pop('state', [0.,0.,0.]), dtype=float)
        self.motion_err = np.array(kwargs.pop('err', [0., 0.]), dtype=float)

    @property
    def pose(self):
        return self.state

    def move(self, motion, **kwargs):
        """Move robot by first turning and then driving along the robot's x-axis."""
        motion_err = kwargs.pop('err', self.motion_err)
        
        sigma = motion_err * np.abs(motion)
        e = np.random.randn(2)
        e *= sigma

        phi = self.state[2] + motion[0] + e[0]
        self.state[0] += math.cos(phi) * (motion[1] + e[1])
        self.state[1] += math.sin(phi) * (motion[1] + e[1])
        self.state[2] = phi
        

