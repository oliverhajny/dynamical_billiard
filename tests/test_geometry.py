import math
import pytest
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

# pomocná funkce: bod leží na elipse?
def on_ellipse(x, y, a, b, eps=1e-9):
    val = (x**2)/(a**2) + (y**2)/(b**2)
    return abs(val - 1.0) < eps

def test_simple_horizontal_hit():
    a, b = 2.0, 1.0
    point = (0.0, 0.0)
    direction = (1.0, 0.0)
    (x, y), t = intersect_line_with_ellipse(point, direction, a, b)
    assert on_ellipse(x, y, a, b)
    assert pytest.approx(x, rel=1e-9) == 2.0
    assert pytest.approx(y, rel=1e-9) == 0.0
    assert t > 0

def test_simple_vertical_hit():
    a, b = 2.0, 1.0
    point = (0.0, 0.0)
    direction = (0.0, 1.0)
    (x, y), t = intersect_line_with_ellipse(point, direction, a, b)
    assert on_ellipse(x, y, a, b)
    assert pytest.approx(x, rel=1e-9) == 0.0
    assert pytest.approx(y, rel=1e-9) == 1.0

def test_diagonal_hit_circle():
    a, b = 1.0, 1.0
    point = (0.0, 0.0)
    direction = (1.0, 1.0)
    (x, y), t = intersect_line_with_ellipse(point, direction, a, b)
    # na kružnici by měl být průsečík (√2/2, √2/2)
    s = math.sqrt(2)/2
    assert on_ellipse(x, y, a, b)
    assert pytest.approx(x, rel=1e-9) == s
    assert pytest.approx(y, rel=1e-9) == s

def test_start_on_boundary_and_go_inside():
    a, b = 2.0, 1.0
    point = (2.0, 0.0)  # start na hraně
    direction = (-1.0, 0.5)   # míří šikmo dovnitř
   # směr dovnitř
    (x, y), t = intersect_line_with_ellipse(point, direction, a, b)
    assert on_ellipse(x, y, a, b)
    assert t > 0

def test_start_on_boundary_and_go_outside_raises():
    a, b = 2.0, 1.0
    point = (2.0, 0.0)       # bod na hraně
    direction = (1.0, 0.0)   # směr ven
    with pytest.raises(ValueError):
        intersect_line_with_ellipse(point, direction, a, b)

def test_tangent_hit():
    a, b = 2.0, 1.0
    point = (0.0, 0.0)
    direction = (2.0, 1.0)   # míří skoro tangentně
    (x, y), t = intersect_line_with_ellipse(point, direction, a, b)
    assert on_ellipse(x, y, a, b)
    assert t > 0
