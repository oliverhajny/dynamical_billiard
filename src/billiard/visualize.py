import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from billiard.simulation import run
from billiard.state import State
from billiard.physics import position_at_time

# animace

def animate_trajectory(states, a: float, b: float, *, interval_ms: int = 25, save_path: str | None = None):
    # 1) vytáhni pozice
    xs = np.array([s.pos[0] for s in states], dtype=float)
    ys = np.array([s.pos[1] for s in states], dtype=float)

    # 2) figure + axes + elipsa
    fig, ax = plt.subplots()
    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(a*np.cos(t), b*np.sin(t), linewidth=1.2)   # obrys elipsy

    path, = ax.plot([], [], linewidth=1.0)             # čára trajektorie
    dot,  = ax.plot([], [], "o")                       # aktuální bod

    ax.set_aspect("equal", adjustable="box")   
    m = 1.1
    ax.set_xlim(-m*a, m*a)
    ax.set_ylim(-m*b, m*b)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Elliptic billiard")

    # 3) init + update
    def init():
        path.set_data([], [])
        dot.set_data([], [])
        return path, dot

    def update(i):
        path.set_data(xs[:i+1], ys[:i+1])
        dot.set_data([xs[i]], [ys[i]])  # <-- wrap in list
        return path, dot

    ani = animation.FuncAnimation(fig, update, frames=len(xs), init_func=init,
                                  blit=True, interval=interval_ms, repeat=False)

    if save_path:
        # pro MP4 potřebuješ v systému ffmpeg; pro GIF pillow/imagio
        ani.save(save_path, fps=max(1, 1000 // max(1, interval_ms)))
    else:
        plt.show()