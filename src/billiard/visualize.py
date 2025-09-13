import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from billiard.state import State
from billiard.physics import position_at_time
from billiard.geometry import normalize, dot
from billiard.shapes import Shape
import os


def animate_trajectory_shape(
    states: list[State],
    shape: Shape,
    *,
    interval_ms: int = 25,
    save_path: str | None = None,
    color: str = "#39FF14",
):
    """
    Animate a trajectory for an arbitrary shape. Interpolates linearly between impacts
    (no collisions assumed within each segment) and draws the boundary via shape.draw(ax).
    """
    # 1) interpolate positions for smooth movement between impacts
    xs_anim: list[float] = []
    ys_anim: list[float] = []
    for i in range(len(states) - 1):
        s0 = states[i]
        s1 = states[i+1]
        dt = float(s1.time - s0.time)
        steps = max(2, int(dt * 1000 // interval_ms))
        for j in range(steps):
            t = j * dt / steps
            pos = position_at_time(s0.pos, s0.dir, s0.speed, t)
            xs_anim.append(float(pos[0]))
            ys_anim.append(float(pos[1]))
    xs_anim.append(float(states[-1].pos[0]))
    ys_anim.append(float(states[-1].pos[1]))

    # 2) figure + axes + boundary
    fig, ax = plt.subplots()
    # Background black, boundary white
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    shape.draw(ax, color="white")

    # Trajectory neon green (default color param)
    path, = ax.plot([], [], linewidth=1.0, color=color)
    dotp,  = ax.plot([], [], "o", color=color)

    ax.set_aspect("equal", adjustable="box")

    # Heuristic bounds: prefer known attributes; otherwise derive from data with margin
    m = 1.1
    if hasattr(shape, "a") and hasattr(shape, "b"):
        a = float(getattr(shape, "a")); b = float(getattr(shape, "b"))
        ax.set_xlim(-m*a, m*a)
        ax.set_ylim(-m*b, m*b)
    elif hasattr(shape, "L") and hasattr(shape, "R"):
        L = float(getattr(shape, "L")); R = float(getattr(shape, "R"))
        ax.set_xlim(-m*(L+R), m*(L+R))
        ax.set_ylim(-m*R, m*R)
    else:
        x_min = min(xs_anim); x_max = max(xs_anim)
        y_min = min(ys_anim); y_max = max(ys_anim)
        dx = x_max - x_min; dy = y_max - y_min
        ax.set_xlim(x_min - 0.1*max(1e-6, dx), x_max + 0.1*max(1e-6, dx))
        ax.set_ylim(y_min - 0.1*max(1e-6, dy), y_max + 0.1*max(1e-6, dy))

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Billiard trajectory")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    def init():
        path.set_data([], [])
        dotp.set_data([], [])
        return path, dotp

    def update(i):
        path.set_data(xs_anim[:i+1], ys_anim[:i+1])
        dotp.set_data([xs_anim[i]], [ys_anim[i]])
        return path, dotp

    if save_path:
        # Ensure parent dir exists
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        fps = max(1, 1000 // max(1, interval_ms))
        root, ext = os.path.splitext(save_path)
        ext = (ext or ".mp4").lower()

        Writer = None
        out_path = save_path
        if ext in (".gif", ".webp"):
            try:
                from matplotlib.animation import PillowWriter
                Writer = PillowWriter
            except Exception:
                # fallback to mp4
                from matplotlib.animation import FFMpegWriter
                Writer = FFMpegWriter
                out_path = root + ".mp4"
        else:
            try:
                from matplotlib.animation import FFMpegWriter
                Writer = FFMpegWriter
            except Exception:
                from matplotlib.animation import PillowWriter
                Writer = PillowWriter
                out_path = root + ".gif"

        writer = Writer(fps=fps)

        # Manual save loop with simple progress output
        total = len(xs_anim)
        step = max(1, total // 20)  # ~5% steps
        with writer.saving(fig, out_path, dpi=150):
            # initialize artists once
            init()
            for i in range(total):
                update(i)
                # ensure canvas is updated before grabbing
                fig.canvas.draw()
                writer.grab_frame()
                if (i + 1) % step == 0 or (i + 1) == total:
                    print(f"Saving animation: {i+1}/{total} frames -> {out_path}")
        plt.close(fig)
    else:
        ani = animation.FuncAnimation(
            fig, update, frames=len(xs_anim), init_func=init,
            blit=True, interval=interval_ms, repeat=False
        )
        plt.show()


def birkhoff_coordinates_shape(states: list[State], shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    s_vals = []
    p_vals = []
    # Skip the initial state (typically interior), use post-impact states
    for st in states[1:]:
        x, y = float(st.pos[0]), float(st.pos[1])
        # rely on shape.arc_param for boundary parameter. If the point is not on boundary,
        # arc_param may be undefined; skip if it errors.
        try:
            s = shape.arc_param((x, y))
        except Exception:
            continue
        n_out = shape.normal_at((x, y))
        vhat = normalize(st.dir)
        p = -dot(vhat, n_out)
        s_vals.append(s)
        p_vals.append(p)
    return np.asarray(s_vals), np.asarray(p_vals)


def plot_poincare_shape(
    states: list[State],
    shape: Shape,
    *,
    show: bool = True,
    save_path: str | None = None,
    ax=None,
):
    """
    Scatter plot of Poincaré (Birkhoff) map for an arbitrary shape.

    Args:
        states: list of post-impact states (as returned by run_shape with collect_states=True)
        shape: a Shape providing arc_param and normal_at
        show: call plt.show() at the end
        save_path: optional path to save figure
        ax: optional axes to draw into
    """
    s_vals, p_vals = birkhoff_coordinates_shape(states, shape)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(s_vals, p_vals, s=4, alpha=0.7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("s (boundary coordinate)")
    ax.set_ylabel("p = sin(ψ)")
    ax.set_title("Poincaré map")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_poincare_groups_shape(
    groups: list[list[State]],
    shape: Shape,
    *,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    show: bool = True,
    save_path: str | None = None,
    ax=None,
):
    """
    Plot multiple Poincaré datasets with different colors/labels for an arbitrary shape.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if labels is None:
        labels = [f"group {i+1}" for i in range(len(groups))]
    if colors is None:
        default = [
            "tab:blue", "tab:orange", "tab:green", "tab:red",
            "tab:purple", "tab:brown", "tab:pink", "tab:gray",
            "tab:olive", "tab:cyan",
        ]
        colors = [default[i % len(default)] for i in range(len(groups))]

    for states, label, color in zip(groups, labels, colors):
        if not states:
            continue
        s_vals, p_vals = birkhoff_coordinates_shape(states, shape)
        ax.scatter(s_vals, p_vals, s=4, alpha=0.7, label=label, color=color)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("s (boundary coordinate)")
    ax.set_ylabel("p = sin(ψ)")
    ax.set_title("Poincaré map")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
