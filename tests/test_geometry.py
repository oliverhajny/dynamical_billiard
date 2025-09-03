import math
from billiard.geometry import normal_at_point
from billiard.geometry import intersect_line_with_ellipse

def almost_equal(v1, v2, eps=1e-6):
    return all(abs(a - b) < eps for a, b in zip(v1, v2))

def test_circle_normals():
    a = b = 1.0  # kružnice
    # bod (1,0)
    n = normal_at_point((1,0), a, b)
    assert almost_equal(n, (1.0, 0.0))
    # bod (0,1)
    n = normal_at_point((0,1), a, b)
    assert almost_equal(n, (0.0, 1.0))
    # bod (sqrt(2)/2, sqrt(2)/2)
    s = math.sqrt(2) / 2
    n = normal_at_point((s,s), a, b)
    assert almost_equal(n, (s, s))  # směr (0.707,0.707)

def test_ellipse_normals():
    a, b = 2.0, 1.0
    # bod (2,0) → normála doprava
    n = normal_at_point((2,0), a, b)
    assert almost_equal(n, (1.0, 0.0))
    # bod (0,1) → normála nahoru
    n = normal_at_point((0,1), a, b)
    assert almost_equal(n, (0.0, 1.0))

def test_normal_is_unit_length():
    a, b = 2.0, 1.0
    points = [(2,0), (0,1), (math.sqrt(2), 0.5)]
    for p in points:
        n = normal_at_point(p, a, b)
        length = math.sqrt(n[0]**2 + n[1]**2)
        assert abs(length - 1.0) < 1e-6

def test_intersect_line_with_circle():
    a = b = 1.0
    # Start at origin, direction (1,0)
    pt = (0.0, 0.0)
    dir = (1.0, 0.0)
    x, y = intersect_line_with_ellipse(pt, dir, a, b)
    assert almost_equal((x, y), (1.0, 0.0))
    # Start at (0.5, 0), direction (1,0)
    pt = (0.5, 0.0)
    dir = (1.0, 0.0)
    x, y = intersect_line_with_ellipse(pt, dir, a, b)
    assert almost_equal((x, y), (1.0, 0.0))
    # Start at (0,0.5), direction (0,1)
    pt = (0.0, 0.5)
    dir = (0.0, 1.0)
    x, y = intersect_line_with_ellipse(pt, dir, a, b)
    assert almost_equal((x, y), (0.0, 1.0))

def test_intersect_line_with_ellipse():
    a, b = 2.0, 1.0
    # Start at origin, direction (1,0)
    pt = (0.0, 0.0)
    dir = (1.0, 0.0)
    x, y = intersect_line_with_ellipse(pt, dir, a, b)
    assert almost_equal((x, y), (2.0, 0.0))
    # Start at origin, direction (0,1)
    pt = (0.0, 0.0)
    dir = (0.0, 1.0)
    x, y = intersect_line_with_ellipse(pt, dir, a, b)
    assert almost_equal((x, y), (0.0, 1.0))
    # Start at (1,0), direction (1,1)
    pt = (1.0, 0.0)
    dir = (1.0, 1.0)
    x, y = intersect_line_with_ellipse(pt, dir, a, b)
    # Should hit ellipse, check if result satisfies ellipse equation
    assert abs((x/a)**2 + (y/b)**2 - 1.0) < 1e-6

def test_intersect_line_with_ellipse_boundary_start():
    a, b = 2.0, 1.0
    # Start at boundary, direction outward
    pt = (2.0, 0.0)
    dir = (1.0, 0.0)
    x, y = intersect_line_with_ellipse(pt, dir, a, b)
    # Should be the same point (or very close)
    assert almost_equal((x, y), (2.0, 0.0))

#