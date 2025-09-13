import numpy as np
import math
from typing import Tuple

EPS = 1e-12
NUDGE = 1e-9

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
        ) -> tuple[tuple[float, float], float] | tuple[float, float]:
    """
    Najdi nejbližší průsečík přímky (point + t*direction, t>0) s elipsou x^2/a^2 + y^2/b^2 = 1.
    - Ošetří tečnu (D ~ 0) i start na hraně (t ~ 0) jemným posunem (nudge)
    - Výsledek "přisnapne" zpět na elipsu (omezení numerické chyby).
-   - Vrací i hodnotu parametru t (geometrická vzdálenost k průsečíku), pokud je return_t=True.
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

        # Nudge in case of a position really close to the tangent
        if any(abs(t) <= eps for t in (t1, t2)):
            nvx, nvy = normalize(direction)
            x0 += nvx * NUDGE * max(a, b)
            y0 += nvy * NUDGE * max(a, b)
            continue

        raise ValueError("No valid forward intersection")
    
    raise ValueError("No valid forward intersection (tangent/edge)")