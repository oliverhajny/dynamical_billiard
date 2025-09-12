from __future__ import annotations

from typing import Protocol, Tuple
import numpy as np
from math import atan2, pi

from billiard.geometry import (
    intersect_line_with_ellipse,
    normal_at_point,
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

    def draw(self, ax) -> None:
        import numpy as np
        t = np.linspace(0.0, 2.0 * np.pi, 400)
        ax.plot(self.a * np.cos(t), self.b * np.sin(t), linewidth=1.2)

