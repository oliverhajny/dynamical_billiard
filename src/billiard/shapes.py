from __future__ import annotations

from typing import Protocol, Tuple
import numpy as np
from math import atan2, pi

from billiard.geometry import (
    intersect_line_with_ellipse,
    normal_at_point,
    EPS,
    NUDGE,
    normalize,
)


class Shape(Protocol):
    def intersect_ray(self, point: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        ...

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        ...

    def arc_param(self, point: np.ndarray) -> float:
        ...

    def draw(self, ax) -> None:
        ...


class EllipseShape:
    """
    Shape wrapper for an ellipse x^2/a^2 + y^2/b^2 = 1.
    """

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)

    def intersect_ray(self, point: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        (impact, t) = intersect_line_with_ellipse(point, direction, self.a, self.b, return_t=True)
        return np.asarray(impact, dtype=float), float(t)

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        return normal_at_point(point, self.a, self.b)

    def arc_param(self, point: np.ndarray) -> float:
        # Parameter via scaled atan2, normalized to [0,1)
        x, y = float(point[0]), float(point[1])
        phi = atan2(y / self.b, x / self.a)
        if phi < 0:
            phi += 2 * pi
        return phi / (2 * pi)

    def draw(self, ax, *, color: str = "white") -> None:
        import numpy as np
        t = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(self.a * np.cos(t), self.b * np.sin(t), linewidth=1.2, color=color)


class StadiumShape:
    """
    Stadium billiard: two semicircles of radius R centered at (-L,0) and (L,0),
    connected by horizontal line segments at y=±R for x∈[-L, L].
    """

    def __init__(self, R: float, L: float):
        if R <= 0 or L < 0:
            raise ValueError("Stadium requires R>0 and L>=0")
        self.R = float(R)
        self.L = float(L)

    def _intersect_with_horizontal(self, point: np.ndarray, direction: np.ndarray, y_line: float):
        x0, y0 = float(point[0]), float(point[1])
        vx, vy = float(direction[0]), float(direction[1])
        if abs(vy) < EPS:
            return None
        t = (y_line - y0) / vy
        if t <= EPS:
            return None
        x = x0 + t * vx
        if x < -self.L - 1e-9 or x > self.L + 1e-9:
            return None
        return (x, y_line, float(t))

    def _circle_intersections(self, point: np.ndarray, direction: np.ndarray, cx: float, cy: float):
        x0, y0 = float(point[0]), float(point[1])
        vx, vy = float(direction[0]), float(direction[1])
        dx = x0 - cx
        dy = y0 - cy
        A = vx*vx + vy*vy
        B = 2.0 * (vx*dx + vy*dy)
        C = dx*dx + dy*dy - self.R*self.R
        D = B*B - 4*A*C
        if D < 0.0 and D > -1e-14:
            D = 0.0
        if D < 0.0 or A == 0.0:
            return []
        sqrtD = D**0.5
        t1 = (-B - sqrtD) / (2*A)
        t2 = (-B + sqrtD) / (2*A)
        out = []
        for t in (t1, t2):
            if t > EPS:
                x = x0 + t * vx
                y = y0 + t * vy
                out.append((x, y, float(t)))
        return out

    def intersect_ray(self, point: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        p = np.asarray(point, dtype=float)
        v = np.asarray(direction, dtype=float)
        if np.linalg.norm(v) == 0:
            raise ValueError("Direction vector must be non-zero.")

        candidates = []
        # Top and bottom lines
        top = self._intersect_with_horizontal(p, v, +self.R)
        if top is not None:
            candidates.append(top)
        bot = self._intersect_with_horizontal(p, v, -self.R)
        if bot is not None:
            candidates.append(bot)

        # Left and right circular caps
        for (cx, side) in [(-self.L, 'left'), (self.L, 'right')]:
            for x, y, t in self._circle_intersections(p, v, cx, 0.0):
                if side == 'left' and x <= -self.L + 1e-9:
                    candidates.append((x, y, t))
                if side == 'right' and x >= self.L - 1e-9:
                    candidates.append((x, y, t))

        if not candidates:
            # Nudge along direction and retry to avoid grazing/tangent starts
            nv = normalize(v)
            p2 = p + NUDGE * max(self.R, self.L, 1.0) * nv
            return self.intersect_ray(p2, v)

        x, y, tmin = min(candidates, key=lambda it: it[2])
        return np.array([x, y], dtype=float), float(tmin)

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        x, y = float(point[0]), float(point[1])
        # Straight segments
        if abs(y - self.R) <= 1e-6 and -self.L - 1e-6 <= x <= self.L + 1e-6:
            return np.array([0.0, 1.0])
        if abs(y + self.R) <= 1e-6 and -self.L - 1e-6 <= x <= self.L + 1e-6:
            return np.array([0.0, -1.0])
        # Circular caps: outward radial normal
        if x <= -self.L + 1e-6:
            cx = -self.L
        elif x >= self.L - 1e-6:
            cx = self.L
        else:
            # choose closer center
            cx = -self.L if (x + self.L)**2 + y*y < (x - self.L)**2 + y*y else self.L
        n = np.array([x - cx, y - 0.0], dtype=float)
        return normalize(n)

    def arc_param(self, point: np.ndarray) -> float:
        # Normalized perimeter coordinate starting at (L, R), counter-clockwise
        x, y = float(point[0]), float(point[1])
        R = self.R; L = self.L
        P = 4.0*L + 2.0*pi*R
        tol = 1e-6
        # top segment
        if abs(y - R) <= tol and -L - tol <= x <= L + tol:
            s = (L - x)
            return (s % P) / P
        # left cap
        if x <= -L + tol:
            phi = atan2(y, x + L)
            delta = (pi/2) - phi
            # wrap into [0, 2pi)
            while delta < 0:
                delta += 2*pi
            while delta > 2*pi:
                delta -= 2*pi
            if delta > pi:
                delta = 2*pi - delta
            s = 2.0*L + R*delta
            return (s % P) / P
        # bottom segment
        if abs(y + R) <= tol and -L - tol <= x <= L + tol:
            s = 2.0*L + pi*R + (x + L)
            return (s % P) / P
        # right cap
        if x >= L - tol:
            phi = atan2(y, x - L)
            delta = phi - (-pi/2)
            while delta < 0:
                delta += 2*pi
            while delta > 2*pi:
                delta -= 2*pi
            if delta > pi:
                delta = 2*pi - delta
            s = 2.0*L + pi*R + 2.0*L + R*delta
            return (s % P) / P
        # fallback
        return 0.0

    def draw(self, ax, *, color: str = "white") -> None:
        import numpy as np
        R = self.R; L = self.L
        xs = np.linspace(-L, L, 200)
        ax.plot(xs, np.full_like(xs, R), linewidth=1.2, color=color)
        ax.plot(xs, np.full_like(xs, -R), linewidth=1.2, color=color)
        tL = np.linspace(pi/2, 3*np.pi/2, 200)
        ax.plot(-L + R*np.cos(tL), 0.0 + R*np.sin(tL), linewidth=1.2, color=color)
        tR = np.linspace(-pi/2, pi/2, 200)
        ax.plot(L + R*np.cos(tR), 0.0 + R*np.sin(tR), linewidth=1.2, color=color)
