import numpy as np
import pytest

from billiard.shapes import StadiumShape
from billiard.physics import advance_with_time_shape
from billiard.simulation import run_shape
from billiard.state import State
from billiard.visualize import birkhoff_coordinates_shape


def test_stadium_top_segment_hit_and_reflection_time():
    shape = StadiumShape(R=1.0, L=2.0)
    point = np.array([0.0, 0.0])
    direction = np.array([0.0, 1.0])  # unit up
    speed = 2.0

    impact, new_dir, dt, t = advance_with_time_shape(point, direction, speed, shape)

    assert np.allclose(impact, [0.0, 1.0])
    assert np.allclose(new_dir, [0.0, -1.0])  # reflect from top, go down
    assert dt == pytest.approx(1.0 / speed)   # distance 1, speed 2
    assert t == pytest.approx(1.0)


def test_stadium_right_cap_hit_and_reflection():
    shape = StadiumShape(R=1.0, L=2.0)
    point = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])  # unit right
    speed = 1.0

    impact, new_dir, dt, t = advance_with_time_shape(point, direction, speed, shape)

    # Expect rightmost point on right cap at (L+R, 0)
    assert np.allclose(impact, [3.0, 0.0])
    # Normal at that point is (1,0), reflection flips x
    assert np.allclose(new_dir, [-1.0, 0.0])
    assert dt == pytest.approx(3.0)
    assert t == pytest.approx(3.0)


def test_stadium_two_steps_accumulate_time():
    shape = StadiumShape(R=1.0, L=2.0)
    s0 = State(pos=np.array([0.0, 0.0]), dir=np.array([0.0, 1.0]), speed=1.0, time=0.0)

    states, bounces = run_shape(s0, shape, max_bounces=2)

    # s1: (0,0)->(0,1), dt=1
    # s2: (0,1)->(0,-1), dt=2 (across vertical span)
    assert len(states) == 3
    assert np.allclose(states[1].pos, [0.0, 1.0])
    assert np.allclose(states[2].pos, [0.0, -1.0])
    assert states[2].time == pytest.approx(3.0)


def test_birkhoff_coordinates_shape_normal_incidence():
    shape = StadiumShape(R=1.0, L=2.0)
    s0 = State(pos=np.array([0.0, 0.0]), dir=np.array([0.0, 1.0]), speed=1.0, time=0.0)
    states, _ = run_shape(s0, shape, max_bounces=3)

    s_vals, p_vals = birkhoff_coordinates_shape(states, shape)

    # First two impacts are vertical head-on at top and bottom: p should be ~ +1
    assert len(p_vals) >= 2
    assert p_vals[0] == pytest.approx(1.0, abs=1e-9)
    assert p_vals[1] == pytest.approx(1.0, abs=1e-9)
    # s should be within [0,1)
    assert np.all((s_vals >= 0.0) & (s_vals < 1.0))

