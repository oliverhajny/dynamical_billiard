import math
import pytest
import numpy as np
from billiard.geometry import normal_at_point
from billiard.geometry import intersect_line_with_ellipse
from billiard.geometry import advance, advance_with_time

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
    direction = (2.0, 1.0)   # míří skoro po tečně
    (x, y), t = intersect_line_with_ellipse(point, direction, a, b)
    assert on_ellipse(x, y, a, b)
    assert t > 0

def on_ellipse(x, y, a, b, eps=1e-9):
    return abs((x*x)/(a*a) + (y*y)/(b*b) - 1.0) < eps

def is_unit(v, eps=1e-9):
    return abs(np.linalg.norm(v) - 1.0) < eps

def test_advance_simple_right_hit():
    a, b = 2.0, 1.0
    p0 = np.array([0.0, 0.0])
    d0 = np.array([1.0, 0.0])  # směrem doprava
    impact, new_dir, t = advance(p0, d0, a, b, return_t=True)

    # dopad na (2,0), vzdálenost t = 2
    assert on_ellipse(impact[0], impact[1], a, b)
    assert impact[0] == pytest.approx(2.0, rel=1e-9)
    assert impact[1] == pytest.approx(0.0, rel=1e-9)
    assert t == pytest.approx(2.0, rel=1e-9)

    # odraz od svislé „stěny“ → směr se otočí do leva
    assert is_unit(new_dir)
    assert new_dir[0] == pytest.approx(-1.0, rel=1e-9)
    assert new_dir[1] == pytest.approx(0.0, rel=1e-9)

def test_advance_diagonal_on_unit_circle():
    a = b = 1.0
    p0 = np.array([0.0, 0.0])
    d0 = np.array([1.0, 1.0]) / math.sqrt(2.0)
    impact, new_dir, t = advance(p0, d0, a, b, return_t=True)

    s = math.sqrt(2.0) / 2.0
    assert on_ellipse(impact[0], impact[1], a, b)
    assert impact[0] == pytest.approx(s, rel=1e-9)
    assert impact[1] == pytest.approx(s, rel=1e-9)

    # u kružnice při radiálním zásahu je v' = -v
    assert is_unit(new_dir)
    assert new_dir[0] == pytest.approx(-d0[0], rel=1e-9)
    assert new_dir[1] == pytest.approx(-d0[1], rel=1e-9)

def test_advance_start_on_boundary_and_go_inside():
    a, b = 2.0, 1.0
    p0 = np.array([2.0, 0.0])                 # na hraně
    d0 = np.array([-1.0, 0.25])               # šikmo dovnitř
    d0 = d0 / np.linalg.norm(d0)
    impact, new_dir, t = advance(p0, d0, a, b, return_t=True)

    assert on_ellipse(impact[0], impact[1], a, b)
    assert t > 0
    assert is_unit(new_dir)

def test_advance_return_without_t_flag():
    a, b = 2.0, 1.0
    p0 = np.array([0.0, 0.0])
    d0 = np.array([0.0, 1.0])
    out = advance(p0, d0, a, b, return_t=False)
    assert isinstance(out, tuple) and len(out) == 2
    impact, new_dir = out
    assert on_ellipse(impact[0], impact[1], a, b)
    assert is_unit(new_dir)

def test_advance_with_time_dt_matches_t_over_speed():
    a, b = 2.0, 1.0
    p0 = np.array([0.0, 0.0])
    d0 = np.array([1.0, 0.0])
    speed = 2.0

    # z advance získáme geometrickou vzdálenost t
    impact_a, new_dir_a, t_geom = advance(p0, d0, a, b, return_t=True)
    # z advance_with_time získáme dt = t/speed
    impact_b, new_dir_b, dt, t_echo = advance_with_time(p0, d0, speed, a, b, return_t=True)

    assert on_ellipse(impact_b[0], impact_b[1], a, b)
    assert np.allclose(impact_a, impact_b)
    assert np.allclose(new_dir_a, new_dir_b)
    assert dt == pytest.approx(t_geom / speed, rel=1e-12)
    assert t_echo == pytest.approx(t_geom, rel=1e-12)

def test_advance_with_time_speed_must_be_positive():
    a, b = 2.0, 1.0
    p0 = np.array([0.0, 0.0])
    d0 = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        _ = advance_with_time(p0, d0, 0.0, a, b, return_t=True)

def test_start_on_boundary_pure_tangent_raises():
    a, b = 2.0, 1.0
    p0 = np.array([2.0, 0.0])      # na hraně
    d0 = np.array([0.0, 1.0])      # čistě tečně (svisle nahoru)
    with pytest.raises(ValueError):
        _ = advance(p0, d0, a, b, return_t=True)

def test_start_on_boundary_outward_raises():
    a, b = 2.0, 1.0
    p0 = np.array([2.0, 0.0])      # na hraně
    d0 = np.array([1.0, 0.0])      # ven
    with pytest.raises(ValueError):
        _ = advance(p0, d0, a, b, return_t=True)

def test_near_tangent_from_inside_still_hits_forward():
    a, b = 2.0, 1.0
    # malinko uvnitř u (2,0): posuň se dovnitř o epsilon po normále
    eps = 1e-6
    p0 = np.array([2.0 - eps, 0.0])
    # téměř tečný směr (malá složka dovnitř)
    d0 = np.array([-1e-3, 1.0])
    d0 = d0 / np.linalg.norm(d0)

    impact, new_dir, t = advance(p0, d0, a, b, return_t=True)
    # očekáváme dopředný zásah s kladným t (může být malé)
    assert t > 0
    # směr po odrazu je jednotkový
    assert abs(np.linalg.norm(new_dir) - 1.0) < 1e-9

def test_advance_with_time_tangent_raises_too():
    a, b = 2.0, 1.0
    p0 = np.array([2.0, 0.0])
    d0 = np.array([0.0, 1.0])      # tečně
    with pytest.raises(ValueError):
        _ = advance_with_time(p0, d0, speed=1.0, a=a, b=b, return_t=True)