import numpy as np
from billiard.state import State
from billiard.shapes import StadiumShape, EllipseShape
from billiard.simulation import run_shape
from billiard.visualize import animate_trajectory_shape, plot_poincare_shape

a, b = 5.0, 3.0 # poloosy elipsy
#R, L = 1.0, 2.0 # parametry stadionu
start = np.array([2.5, 0.0]) # start [x0, y0]
direction = np.array([0.5, 0.7]) # směr [vx, vy]
s0 = State(pos=start, dir=direction, speed=20.0, time=0.0)

#shape = StadiumShape(R, L)
shape = EllipseShape(a, b)
states, _ = run_shape(s0, shape, max_bounces=2000)

animate_trajectory_shape(states, shape, interval_ms=16)
plot_poincare_shape(states, shape)