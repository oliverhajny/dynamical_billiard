import numpy as np
import math

# Vectors handling

def length(v) -> float:
    """Returns length of a vector"""
    return np.linalg.norm(v)
def normalize(v) -> np.array:
    """normalize a vector"""
    return v / length(v)
def dot(u, v) -> float:
    """Return the dot product of vectors u and v"""
    return np.dot(u,v)

# Ellipse
 
def normal_at_point(point:np.array, a:float, b:float) -> np.array:
    """
    Returns the normal vector (normalized gradient) in the point "point"
    on an ellipse with axes "a" and "b"
    """
    if a == 0 or b == 0:
        raise ValueError("Ellipse axes 'a' and 'b' must be non-zero.")
    x, y = point
    # gradient at the point
    gradient = np.array([2*x / (a**2), 2*y / (b**2)])
    # normalization
    return normalize(gradient)

def intersect_line_with_ellipse(point:np.array, direction:np.array, a:float, b:float) -> tuple:
    """
    Return the coordinates of the nearest intersection point on the ellipse (point of impact).
    Handles the case when starting exactly on the boundary and moving perpendicular to the tangent.
    """
    x0, y0 = point
    vx, vy = direction

    if a == 0 or b == 0:
        raise ValueError("Ellipse axes 'a' and 'b' must be non-zero.")
    if abs(vx) < 1e-12 and abs(vy) < 1e-12:
        raise ValueError("Direction vector must be non-zero.")

    A = (vx**2 / a**2) + (vy**2 / b**2)
    B = 2*((x0*vx / a**2) + (y0*vy / b**2))
    C = (x0**2 / a**2) + (y0**2 / b**2) - 1

    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        raise ValueError("No real intersection: discriminant is negative.")

    sqrt_disc = math.sqrt(discriminant)
    denom = 2*A
    t1 = (-B + sqrt_disc) / denom
    t2 = (-B - sqrt_disc) / denom
    roots = [t1, t2]

    # If any root is very close to zero, return the starting point
    for t in roots:
        if abs(t) < 1e-10:
            return (x0, y0)

    candidates = [t for t in roots if t > 1e-12]
    if not candidates:
        raise ValueError("No valid intersection: both roots are negative or too close to zero.")
    t = min(candidates)
    x = x0 + vx*t
    y = y0 + vy*t
    return (x, y)

# Reflection
