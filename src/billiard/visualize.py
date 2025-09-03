import matplotlib.pyplot as plt
import numpy as np
from billiard.simulation import run
from billiard.state import State

# setup
a, b = 5, 3
c = np.sqrt(a**2 - b**2)  # focal distance
F1 = np.array([-c, 0.0])
F2 = np.array([ c, 0.0])
s0 = State(pos = np.array([5.0, 0.0]),
           dir = np.array([-1.0, 0.5]),
           speed = 1.0,
           time = 0.0)

# spustíme simulaci
states = run(s0, a, b, max_bounces=200)

# vyberem z každého stavu pozici
xs = [s.pos[0] for s in states]
ys = [s.pos[1] for s in states]

# elipsa
t = np.linspace(0, 2*np.pi, 400)
plt.plot(a*np.cos(t), b*np.sin(t), "k") # černá čára

# trajectory
plt.plot(xs, ys, "-o", markersize=3)

plt.gca().set_aspect("equal", adjustable="box")
plt.show()