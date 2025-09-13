from billiard.simulation import run_shape
from billiard.state import State
import numpy as np
import argparse
from typing import Iterable
from billiard.visualize import (
    animate_trajectory_shape,
    plot_poincare_shape,
    plot_poincare_groups_shape,
)
from billiard.shapes import EllipseShape, StadiumShape


def generate_start_positions(a: float, b: float, *, n_inside: int, n_outside: int) -> list[np.ndarray]:
    """
    Generate start positions along the major axis (y=0):
    - inside focus region: |x| < c
    - outside focus region: c < |x| < a
    Returns a list of 2D numpy arrays [x, 0].
    """
    c = float(np.sqrt(max(a*a - b*b, 0.0)))
    eps = 1e-6
    starts: list[np.ndarray] = []

    # inside foci (|x| < c)
    if n_inside > 0 and c > eps:
        xs_in = np.linspace(-0.9*c, 0.9*c, n_inside)
        for x in xs_in:
            starts.append(np.array([x, 0.0]))

    # outside foci (c < |x| < a)
    if n_outside > 0 and a - c > eps:
        half = max(1, n_outside // 2)
        # sample symmetrically away from the foci and boundary
        left = np.linspace(-a*0.95, -max(c*1.01, a*0.05), half)
        right = np.linspace(max(c*1.01, a*0.05), a*0.95, half)
        for x in np.concatenate([left, right]):
            if abs(x) < a:
                starts.append(np.array([float(x), 0.0]))

    if not starts:
        # fallback: center
        starts.append(np.array([0.0, 0.0]))

    return starts


def generate_directions(n_angles: int, deg_min: float, deg_max: float) -> Iterable[np.ndarray]:
    thetas = np.deg2rad(np.linspace(deg_min, deg_max, max(1, n_angles)))
    for th in thetas:
        yield np.array([np.cos(th), np.sin(th)])


def run_batch(a: float, b: float, *, n_inside: int, n_outside: int, n_angles: int,
              deg_min: float, deg_max: float, speed: float, bounces: int) -> dict[str, list[list[State]]]:
    starts = generate_start_positions(a, b, n_inside=n_inside, n_outside=n_outside)
    dirs = list(generate_directions(n_angles, deg_min, deg_max))

    c = float(np.sqrt(max(a*a - b*b, 0.0)))
    shape = EllipseShape(a, b)
    groups: dict[str, list[list[State]]] = {"inside": [], "outside": []}
    for start in starts:
        grp = "inside" if abs(float(start[0])) < c else "outside"
        for d in dirs:
            s0 = State(pos=start, dir=d, speed=speed, time=0.0)
            traj, _ = run_shape(s0, shape, max_bounces=bounces)
            groups[grp].append(traj)
    return groups


def generate_start_positions_stadium(R: float, L: float, *, n_starts: int) -> list[np.ndarray]:
    """Evenly spaced starts along the central line y=0, x in [-L*0.9, L*0.9]."""
    if n_starts <= 0:
        return [np.array([0.0, 0.0])]
    xs = np.linspace(-0.9*L, 0.9*L, n_starts)
    return [np.array([float(x), 0.0]) for x in xs]


def run_batch_stadium(R: float, L: float, *, n_starts: int, n_angles: int,
                      deg_min: float, deg_max: float, speed: float, bounces: int) -> dict[str, list[list[State]]]:
    starts = generate_start_positions_stadium(R, L, n_starts=n_starts)
    dirs = list(generate_directions(n_angles, deg_min, deg_max))
    shape = StadiumShape(R, L)
    groups: dict[str, list[list[State]]] = {"left": [], "right": []}
    for start in starts:
        grp = "left" if float(start[0]) < 0 else "right"
        for d in dirs:
            s0 = State(pos=start, dir=d, speed=speed, time=0.0)
            traj, _ = run_shape(s0, shape, max_bounces=bounces)
            groups[grp].append(traj)
    return groups


def main():
    p = argparse.ArgumentParser(description="Single or batch simulations and Poincaré plots for billiards")
    p.add_argument("--shape", choices=["ellipse", "stadium"], default="ellipse", help="Billiard shape")
    # Single-run toggle and initial conditions
    p.add_argument("--single", action="store_true", help="Run a single trajectory instead of a batch")
    p.add_argument("--x0", type=float, default=0.0, help="Single: initial x position")
    p.add_argument("--y0", type=float, default=0.0, help="Single: initial y position")
    p.add_argument("--vx", type=float, default=0.1, help="Single: initial direction x-component")
    p.add_argument("--vy", type=float, default=0.9, help="Single: initial direction y-component")
    # Ellipse params
    p.add_argument("--a", type=float, default=5.0, help="Major semi-axis a (ellipse)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--b", type=float, default=None, help="Minor semi-axis b (overrides ecc)")
    group.add_argument("--ecc", type=float, default=None, help="Eccentricity e in [0,1). If given, b=a*sqrt(1-e^2)")
    p.add_argument("--n-inside", type=int, default=2, help="Ellipse: starts inside foci (|x|<c) along y=0")
    p.add_argument("--n-outside", type=int, default=2, help="Ellipse: starts outside foci (c<|x|<a) along y=0")
    # Stadium params
    p.add_argument("--R", type=float, default=2.0, help="Stadium cap radius R")
    p.add_argument("--L", type=float, default=5.0, help="Stadium half-length L")
    p.add_argument("--n-starts", type=int, default=5, help="Stadium: number of start positions along y=0")
    # Common params
    p.add_argument("--n-angles", type=int, default=5, help="Number of initial angles")
    p.add_argument("--angle-min", type=float, default=10.0, help="Min angle in degrees")
    p.add_argument("--angle-max", type=float, default=170.0, help="Max angle in degrees")
    p.add_argument("--speed", type=float, default=20.0, help="Ball speed")
    p.add_argument("--bounces", type=int, default=500, help="Max bounces per trajectory")
    p.add_argument("--animate", action="store_true", help="Animate the first trajectory")
    p.add_argument("--anim-interval", type=int, default=25, help="Animation frame interval in ms (affects FPS and export time)")
    p.add_argument("--save-anim", type=str, default=None, help="Path to save animation (e.g., .gif, .mp4). In batch mode saves the first trajectory.")
    p.add_argument("--save-poincare", type=str, default=None, help="Path to save Poincaré scatter (PNG)")
    args = p.parse_args()


    if args.shape == "ellipse":
        a = float(args.a)
        if args.b is not None:
                b = float(args.b)
        elif args.ecc is not None:
                e = float(args.ecc)
                if not (0.0 <= e < 1.0):
                    raise SystemExit("eccentricity must be in [0,1)")
                b = a * float(np.sqrt(max(1.0 - e*e, 0.0)))
        else:
                # default minor axis if not given
                b = 0.6 * a

        if args.single:
            shape_obj = EllipseShape(a, b)
            s0 = State(pos=np.array([args.x0, args.y0]), dir=np.array([args.vx, args.vy]), speed=args.speed, time=0.0)
            states, _ = run_shape(s0, shape_obj, max_bounces=args.bounces)
            if args.save_anim:
                animate_trajectory_shape(states, shape_obj, interval_ms=args.anim_interval, save_path=args.save_anim)
            elif args.animate:
                animate_trajectory_shape(states, shape_obj, interval_ms=args.anim_interval)
            show = (args.save_poincare is None) and (not args.save_anim)
            plot_poincare_shape(states, shape_obj, show=show, save_path=args.save_poincare)
        else:
            groups = run_batch(
                a, b,
                n_inside=args.n_inside,
                n_outside=args.n_outside,
                n_angles=args.n_angles,
                deg_min=args.angle_min,
                deg_max=args.angle_max,
                speed=args.speed,
                bounces=args.bounces,
            )
            inside_states: list[State] = [s for traj in groups.get("inside", []) for s in traj]
            outside_states: list[State] = [s for traj in groups.get("outside", []) for s in traj]
            # pick a trajectory to animate, prefer inside group
            first_traj = None
            if groups.get("inside"):
                first_traj = groups["inside"][0]
            elif groups.get("outside"):
                first_traj = groups["outside"][0]
            if first_traj is not None:
                shape_obj = EllipseShape(a, b)
                if args.save_anim:
                    animate_trajectory_shape(first_traj, shape_obj, interval_ms=args.anim_interval, save_path=args.save_anim)
                elif args.animate:
                    animate_trajectory_shape(first_traj, shape_obj, interval_ms=args.anim_interval)
            show = (args.save_poincare is None) and (not args.save_anim)
            shape = EllipseShape(a, b)
            plot_poincare_groups_shape(
                [inside_states, outside_states], shape,
                labels=["inside |x|<c", "outside |x|>c"],
                colors=["tab:blue", "tab:orange"],
                show=show,
                save_path=args.save_poincare,
            )
    elif args.shape == "stadium":
        R = float(args.R); L = float(args.L)
        if args.single:
            shape = StadiumShape(R, L)
            s0 = State(pos=np.array([args.x0, args.y0]), dir=np.array([args.vx, args.vy]), speed=args.speed, time=0.0)
            states, _ = run_shape(s0, shape, max_bounces=args.bounces)
            if args.save_anim:
                animate_trajectory_shape(states, shape, interval_ms=args.anim_interval, save_path=args.save_anim)
            elif args.animate:
                animate_trajectory_shape(states, shape, interval_ms=args.anim_interval)
            show = (args.save_poincare is None) and (not args.save_anim)
            plot_poincare_shape(states, shape, show=show, save_path=args.save_poincare)
        else:
            groups = run_batch_stadium(
                R, L,
                n_starts=args.n_starts,
                n_angles=args.n_angles,
                deg_min=args.angle_min,
                deg_max=args.angle_max,
                speed=args.speed,
                bounces=args.bounces,
            )
            left_states: list[State] = [s for traj in groups.get("left", []) for s in traj]
            right_states: list[State] = [s for traj in groups.get("right", []) for s in traj]
            shape = StadiumShape(R, L)
            # pick a trajectory to animate, prefer left group
            first_traj = None
            if groups.get("left"):
                first_traj = groups["left"][0]
            elif groups.get("right"):
                first_traj = groups["right"][0]
            if first_traj is not None:
                if args.save_anim:
                    animate_trajectory_shape(first_traj, shape, interval_ms=args.anim_interval, save_path=args.save_anim)
                elif args.animate:
                    animate_trajectory_shape(first_traj, shape, interval_ms=args.anim_interval)
            # Plot without color grouping (merge all states)
            show = (args.save_poincare is None) and (not args.save_anim)
            all_states: list[State] = left_states + right_states
            plot_poincare_shape(all_states, shape, show=show, save_path=args.save_poincare)
