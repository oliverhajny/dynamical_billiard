import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from math import atan2, pi
from billiard.simulation import run
from billiard.state import State
from billiard.physics import position_at_time
from billiard.geometry import normal_at_point, normalize, dot

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


def birkhoff_coordinates(states: list[State], a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Birkhoff coordinates (s, p) for a billiard inside an ellipse, from a list of states.
    - s: boundary coordinate in [0, 1), derived from parameter phi: (x, y) = (a cos phi, b sin phi)
    - p: sin of the angle with the boundary tangent, p in [-1, 1]

    Notes:
    - The input states are assumed to be the post-impact states returned by simulation.run/next_step,
      i.e., positions on the boundary with directions already reflected (pointing inward).
    - If the first state is inside (not on boundary), it is skipped.
    """
    s_vals = []
    p_vals = []

    for st in states:
        x, y = float(st.pos[0]), float(st.pos[1])

        # Detect if on boundary (within tolerance); otherwise skip
        val = (x*x)/(a*a) + (y*y)/(b*b)
        if abs(val - 1.0) > 1e-6:
            continue

        # Parameter on ellipse via scaled atan2
        phi = atan2(y / b, x / a)
        if phi < 0:
            phi += 2 * pi
        s = phi / (2 * pi)  # normalized to [0,1)

        # p = sin(angle with tangent). With outward normal n_out, inward normal is -n_out.
        n_out = normal_at_point((x, y), a, b)
        vhat = normalize(st.dir)
        p = -dot(vhat, n_out)  # use inward normal

        s_vals.append(s)
        p_vals.append(p)

    return np.asarray(s_vals), np.asarray(p_vals)


def plot_poincare(
    states: list[State],
    a: float,
    b: float,
    *,
    show: bool = True,
    save_path: str | None = None,
    ax=None,
):
    """
    Scatter plot of Poincaré (Birkhoff) map for an ellipse.

    Args:
        states: list of post-impact states (as returned by run with collect_states=True)
        a, b: ellipse semi-axes
        show: call plt.show() at the end
    """
    s_vals, p_vals = birkhoff_coordinates(states, a, b)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(s_vals, p_vals, s=8, alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("s (boundary coordinate)")
    ax.set_ylabel("p = sin(ψ)")
    ax.set_title("Poincaré map (ellipse)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_poincare_groups(
    groups: list[list[State]],
    a: float,
    b: float,
    *,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    show: bool = True,
    save_path: str | None = None,
    ax=None,
):
    """
    Plot multiple Poincaré datasets with different colors/labels.

    Args:
        groups: list of state lists (one list per group)
        a, b: ellipse semi-axes
        labels: optional labels per group (same length as groups)
        colors: optional colors per group
        show: whether to call plt.show()
        save_path: optional path to save figure
        ax: optional existing axes to plot into
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if labels is None:
        labels = [f"group {i+1}" for i in range(len(groups))]
    if colors is None:
        # cycle of default matplotlib tab colors
        default = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
            "tab:olive", "tab:cyan",
        ]
        colors = [default[i % len(default)] for i in range(len(groups))]

    for states, label, color in zip(groups, labels, colors):
        if not states:
            continue
        s_vals, p_vals = birkhoff_coordinates(states, a, b)
        ax.scatter(s_vals, p_vals, s=8, alpha=0.7, label=label, color=color)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("s (boundary coordinate)")
    ax.set_ylabel("p = sin(ψ)")
    ax.set_title("Poincaré map (ellipse)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
