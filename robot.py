
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

class Robot:
    def __init__(self, state=[0, 0, 0]):
        self.state = np.asarray(state, dtype=float)
        
    def move(self, dphi, dlen):
        a = self.state[2] + dphi
        
        # Angle wrap
        # math.atan2(math.sin(a), math.cos(a)) -> -pi..pi
        # math.atan2(math.sin(a-math.pi), math.cos(a-math.pi)) + math.pi -> 0..2pi
                
        self.state[2] = math.atan2(math.sin(a-math.pi), math.cos(a-math.pi)) + math.pi
        self.state[0] += math.cos(self.state[2]) * dlen
        self.state[1] += math.sin(self.state[2]) * dlen
        print(self.state)

    def draw(self, ax, radius=0.5):
        dx = math.cos(self.state[2]) * radius
        dy = math.sin(self.state[2]) * radius

        c = plt.Circle(self.state[:2], radius=radius, fc='none', ec='k')
        ax.add_artist(c)
        line,  = ax.plot([self.state[0], self.state[0] + dx], [self.state[1], self.state[1] + dy], color='k')
        return c, line

r = Robot()

fig, ax = plt.subplots()
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

def update(i):
    r.move(0.05, 0.1)
    return r.draw(ax)

ani = animation.FuncAnimation(fig, update, 25, interval=50, blit=True)
plt.show()


