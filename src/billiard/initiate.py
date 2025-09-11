from billiard.simulation import run
from billiard.state import State
import numpy as np
import argparse
from typing import Iterable
from billiard.visualize import animate_trajectory, plot_poincare, plot_poincare_groups


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
    groups: dict[str, list[list[State]]] = {"inside": [], "outside": []}
    for start in starts:
        grp = "inside" if abs(float(start[0])) < c else "outside"
        for d in dirs:
            s0 = State(pos=start, dir=d, speed=speed, time=0.0)
            traj, _ = run(s0, a, b, max_bounces=bounces)
            groups[grp].append(traj)
    return groups


def main():
    p = argparse.ArgumentParser(description="Batch simulations and Poincaré plots for elliptic billiard")
    p.add_argument("--a", type=float, default=10.0, help="Major semi-axis a")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--b", type=float, default=None, help="Minor semi-axis b (overrides ecc)")
    group.add_argument("--ecc", type=float, default=None, help="Eccentricity e in [0,1). If given, b=a*sqrt(1-e^2)")
    group.add_argument("--ecc-scan", nargs=3, metavar=("E_MIN","E_MAX","N"), help="Scan eccentricity range; produces one plot per e")
    p.add_argument("--n-inside", type=int, default=2, help="Start positions inside foci (|x|<c) along y=0")
    p.add_argument("--n-outside", type=int, default=2, help="Start positions outside foci (c<|x|<a) along y=0")
    p.add_argument("--n-angles", type=int, default=10, help="Number of initial angles")
    p.add_argument("--angle-min", type=float, default=10.0, help="Min angle in degrees")
    p.add_argument("--angle-max", type=float, default=170.0, help="Max angle in degrees")
    p.add_argument("--speed", type=float, default=20.0, help="Ball speed")
    p.add_argument("--bounces", type=int, default=1500, help="Max bounces per trajectory")
    p.add_argument("--animate", action="store_true", help="Animate the first trajectory")
    p.add_argument("--save-poincare", type=str, default=None, help="Path to save Poincaré scatter (PNG) for single run")
    p.add_argument("--save-prefix", type=str, default=None, help="Prefix to save multiple plots when using --ecc-scan")
    args = p.parse_args()

    a = float(args.a)
    if args.ecc_scan is not None:
        e_min, e_max, n = args.ecc_scan
        e_min = float(e_min); e_max = float(e_max); n = int(float(n))
        if not (0.0 <= e_min < 1.0 and 0.0 <= e_max < 1.0 and n >= 1):
            raise SystemExit("--ecc-scan requires 0<=E_MIN,E_MAX<1 and N>=1")
        es = np.linspace(e_min, e_max, n)
        for e in es:
            b = a * float(np.sqrt(max(1.0 - e*e, 0.0)))
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
            save_path = None
            if args.save_prefix:
                save_path = f"{args.save_prefix}_e{e:.3f}.png"
            show = args.save_prefix is None
            plot_poincare_groups(
                [inside_states, outside_states], a, b,
                labels=["inside |x|<c", "outside |x|>c"],
                colors=["tab:blue", "tab:orange"],
                show=show,
                save_path=save_path,
            )
    else:
        if args.b is not None:
            b = float(args.b)
        elif args.ecc is not None:
            e = float(args.ecc)
            if not (0.0 <= e < 1.0):
                raise SystemExit("eccentricity must be in [0,1)")
            b = a * float(np.sqrt(max(1.0 - e*e, 0.0)))
        else:
            # default minor axis if not given
            b = 4.0 if a >= 4.0 else 0.9*a

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
        if args.animate and first_traj is not None:
            animate_trajectory(first_traj, a, b, interval_ms=16)
        show = args.save_poincare is None
        plot_poincare_groups(
            [inside_states, outside_states], a, b,
            labels=["inside |x|<c", "outside |x|>c"],
            colors=["tab:blue", "tab:orange"],
            show=show,
            save_path=args.save_poincare,
        )


if __name__ == "__main__":
    main()
