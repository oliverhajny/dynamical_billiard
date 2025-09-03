from billiard.simulation import run
from billiard.state import State
import numpy as np
from billiard.visualize import animate_trajectory

a, b = 5, 3
c = np.sqrt(a**2 - b**2)  # focal distance
F1 = np.array([-c, 0.0])
F2 = np.array([ c, 0.0])
start = np.array([0.0, 0.0])
direction = np.array([1.0, 0.5])

s0 = State(pos = start,
           dir = direction,
           speed = 1.0,
           time = 0.0)

states = run(s0, a, b, max_bounces=200)

animate_trajectory(states, a, b, interval_ms=20)               # zobrazí okno
# animate_trajectory(states, a, b, interval_ms=20, save_path="traj.mp4")  # uloží soubor
