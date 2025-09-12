import numpy as np
import math
from typing import Tuple, Union
from billiard.geometry import dot, normalize, normal_at_point, intersect_line_with_ellipse
from billiard.shapes import Shape

speed = 1.0

def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Calculates the reflected vector after an impact.
    """
    normal = normalize(normal)
    direction = normalize(direction)
    dot_prod = dot(direction, normal)
    reflection = direction - 2 * dot_prod * normal
    return normalize(reflection)

def advance(
    point: np.ndarray,
    direction: np.ndarray,
    a: float,
    b: float,
    *,
    return_t: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
    # 1) průsečík + vzdálenost k dopadu
    impact_point, t = intersect_line_with_ellipse(point, direction, a, b)

    # 2) normála v bodě dopadu
    n_hat = normal_at_point(impact_point, a, b)

    # 3) nový směr
    v = np.asarray(direction, dtype=float)
    v = v / np.linalg.norm(v)
    new_dir = reflect(v, n_hat)

    impact_point = np.asarray(impact_point, dtype=float)

    if return_t:
        return impact_point, new_dir, t
    else:
        return impact_point, new_dir

def advance_with_time(
    point: np.ndarray,
    direction: np.ndarray,
    speed: float,
    a: float,
    b: float,
    *,
    return_t: bool = True,
) -> Union[Tuple[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, float, float]]:
    if speed <= 0:
        raise ValueError("speed must be > 0")

    out = advance(point, direction, a, b, return_t=True)
    impact_point, new_dir, t = out  # t = geometrická vzdálenost
    dt = t / float(speed)

    if return_t:
        return impact_point, new_dir, dt, t  # vracím i t (užitečné pro debug)
    else:
        return impact_point, new_dir

def position_at_time(
    point: np.ndarray,
    direction: np.ndarray,
    speed: float,
    t: float
) -> np.ndarray:
    """
    Returns the position of the ball after time t, moving from 'point' in 'direction' at 'speed'.
    Assumes straight-line motion (no impact).
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    displacement = direction * speed * t
    return np.asarray(point, dtype=float) + displacement


# Shape-agnostic variants (backward-compatible extension)

def advance_shape(
    point: np.ndarray,
    direction: np.ndarray,
    shape: Shape,
    *,
    return_t: bool = True,
):
    impact_point, t = shape.intersect_ray(point, direction)
    n_out = shape.normal_at(impact_point)
    v = np.asarray(direction, dtype=float)
    v = v / np.linalg.norm(v)
    new_dir = reflect(v, n_out)
    if return_t:
        return impact_point, new_dir, t
    else:
        return impact_point, new_dir


def advance_with_time_shape(
    point: np.ndarray,
    direction: np.ndarray,
    speed: float,
    shape: Shape,
    *,
    return_t: bool = True,
):
    if speed <= 0:
        raise ValueError("speed must be > 0")
    impact_point, new_dir, t = advance_shape(point, direction, shape, return_t=True)
    dt = t / float(speed)
    if return_t:
        return impact_point, new_dir, dt, t
    else:
        return impact_point, new_dir
