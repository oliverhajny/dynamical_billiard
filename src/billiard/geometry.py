import numpy as np
import math
from typing import Tuple

EPS = 1e-12
NUDGE = 1e-9
speed = 2.0

# Vectors handling

def length(v: np.ndarray) -> float:
    v = np.array(v)
    return np.linalg.norm(v)

def normalize(v: np.ndarray) -> Tuple[float, float]:
    v = np.array(v)
    l = length(v)
    if l == 0:
        raise ValueError("Zero-length vector")
    return v / l

def dot(u: np.ndarray, v: np.ndarray) -> float:
    u = np.array(u)
    v = np.array(v)
    return float(np.dot(u, v))

# Ellipse

def normal_at_point(point: np.ndarray, a: float, b: float) -> Tuple[float, float]:
    """
    Returns the normal vector (normalized gradient) in the point "point"
    on an ellipse with axes "a" and "b"
    """
    if a == 0 or b == 0:
        raise ValueError("Ellipse axes 'a' and 'b' must be non-zero.")
    point = np.array(point)
    x, y = point
    gradient = np.array([2*x / (a**2), 2*y / (b**2)])
    return normalize(gradient)

def snap_to_ellipse(x: float, y: float, a: float, b: float) -> Tuple[float, float]:
    denom = math.sqrt((x*x)/(a*a) + (y*y)/(b*b))
    if denom < EPS:
        return x, y
    s = 1.0 / denom
    return x * s, y * s

def intersect_line_with_ellipse(
        point: np.ndarray, 
        direction: np.ndarray, 
        a: float, 
        b: float,
        *,
        eps: float = EPS,
        return_t: bool = True,
        ) -> Tuple[Tuple[float, float], float] | Tuple[float, float]:
    """
    Najdi nejbližší průsečík přímky (point + t*direction, t>0) s elipsou x^2/a^2 + y^2/b^2 = 1.
    - Ošetří tečnu (D ~ 0) i start na hraně (t ~ 0) jemným posunem.
    - Výsledek přisnapne zpět na elipsu (omezení numerické chyby).

    Args:
        point: (x0, y0) start (uvnitř nebo na hraně elipsy)
        direction: (vx, vy) směr (nemusí být jednotkový)
        a, b: poloosy elipsy
        eps: tolerance
        return_t: vrátit i čas/vzdálenost t

    Returns:
        ([x, y], t)
    """
    point = np.array(point, dtype=float)
    direction = np.array(direction, dtype=float)
    x0, y0 = point
    vx, vy = direction

    if a == 0 or b == 0:
        raise ValueError("Ellipse axes 'a' and 'b' must be non-zero.")
    if abs(vx) < eps and abs(vy) < eps:
        raise ValueError("Direction vector must be non-zero.")
    
    attempts = 2
    for _ in range(attempts + 1):
        A = (vx**2 / a**2) + (vy**2 / b**2)
        B = 2*((x0*vx / a**2) + (y0*vy / b**2))
        C = (x0**2 / a**2) + (y0**2 / b**2) - 1.0   
        
        D = B**2.0 - 4.0*A*C
        if D < 0.0 and D > -1e-14:
            D = 0.0 # numerické zaokrouhlení tečny
        if D < 0.0:
            raise ValueError("No real intersection: discriminant is negative.")

        sqrtD = math.sqrt(D)
        denom = 2.0 * A
        t1 = (-B + sqrtD) / denom
        t2 = (-B - sqrtD) / denom
        
        candidates = [t for t in (t1, t2) if t > eps]

        if candidates:
            t = min(candidates)
            x = x0 + vx * t
            y = y0 + vy * t
            x, y = snap_to_ellipse(x, y, a, b)
            return ((x, y), t) if return_t else (x, y)

        if any(abs(t) <= eps for t in (t1, t2)):
            nvx, nvy = normalize(direction)
            x0 += nvx * NUDGE * max(a, b)
            y0 += nvy * NUDGE * max(a, b)
            continue

        raise ValueError("No valid forward intersection")
    
    raise RuntimeError("Intersection failed after nudging attempts")

# Reflection

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
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, float]:
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
) -> Tuple[np.ndarray, np.ndarray, float] | Tuple[np.ndarray, np.ndarray, float, float]:
    if speed <= 0:
        raise ValueError("speed must be > 0")

    out = advance(point, direction, a, b, return_t=True)
    impact_point, new_dir, t = out  # t = geometrická vzdálenost
    dt = t / float(speed)

    if return_t:
        return impact_point, new_dir, dt, t  # vracím i t (užitečné pro debug)
    else:
        return impact_point, new_dir