import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from billiard.simulation import run
from billiard.state import State
from billiard.physics import position_at_time

def animate_trajectory(states, a: float, b: float, *, interval_ms: int = 25, save_path: str | None = None):
    # 1) interpolate positions for smooth movement
    xs_anim = []
    ys_anim = []
    for i in range(len(states) - 1):
        s0 = states[i]
        s1 = states[i+1]
        dt = s1.time - s0.time
        steps = max(2, int(dt * 1000 // interval_ms))  # at least 2 steps per segment
        for j in range(steps):
            t = j * dt / steps
            pos = position_at_time(s0.pos, s0.dir, s0.speed, t)
            xs_anim.append(pos[0])
            ys_anim.append(pos[1])
    # Add final impact position
    xs_anim.append(states[-1].pos[0])
    ys_anim.append(states[-1].pos[1])

    # 2) figure + axes + ellipse
    fig, ax = plt.subplots()
    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(a*np.cos(t), b*np.sin(t), linewidth=1.2)   # ellipse outline

    path, = ax.plot([], [], linewidth=1.0)             # trajectory line
    dot,  = ax.plot([], [], "o")                       # current ball

    ax.set_aspect("equal", adjustable="box")   
    m = 1.1
    ax.set_xlim(-m*a, m*a)
    ax.set_ylim(-m*b, m*b)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Elliptic billiard")

    def init():
        path.set_data([], [])
        dot.set_data([], [])
        return path, dot

    def update(i):
        path.set_data(xs_anim[:i+1], ys_anim[:i+1])
        dot.set_data([xs_anim[i]], [ys_anim[i]])
        return path, dot

    ani = animation.FuncAnimation(fig, update, frames=len(xs_anim), init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)

    if save_path:
        ani.save(save_path, fps=max(1, 1000 // max(1, interval_ms)))
    else:
        plt.show()