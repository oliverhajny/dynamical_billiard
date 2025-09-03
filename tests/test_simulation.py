import numpy as np
import pytest

from billiard.simulation import next_step, run
from billiard.state import State


def test_step_unit_circle_axis_hit():
    """
    Kružnice (a=b=1). Start v (0,0) se směrem +x.
    První náraz bude v (1,0); normála je (1,0); odraz dá směr (-1,0).
    Čas do nárazu = vzdálenost 1 / speed.
    """
    a = b = 1.0
    speed = 2.0
    s0 = State(pos=np.array([0.0, 0.0]), dir=np.array([1.0, 0.0]), speed=speed, time=0.0)

    s1 = next_step(s0, a, b)

    assert np.allclose(s1.pos, [1.0, 0.0])
    assert np.allclose(s1.dir, [-1.0, 0.0])
    assert s1.speed == speed
    assert s1.time == pytest.approx(1.0 / speed)  # 0.5


def test_step_does_not_mutate_input_state():
    """Funkce next_step musí vracet NOVÝ stav a nezměnit s0 ani jeho numpy pole."""
    a = b = 1.0
    pos0 = np.array([0.0, 0.0])
    dir0 = np.array([1.0, 0.0])
    s0 = State(pos=pos0, dir=dir0, speed=1.0, time=0.0)

    s1 = next_step(s0, a, b)

    # vstupy beze změny
    assert np.allclose(pos0, [0.0, 0.0])
    assert np.allclose(dir0, [1.0, 0.0])
    assert np.allclose(s0.pos, [0.0, 0.0])
    assert np.allclose(s0.dir, [1.0, 0.0])
    assert s0.time == 0.0

    # výstup je jiný objekt/stav
    assert not np.allclose(s1.pos, s0.pos) or not np.allclose(s1.dir, s0.dir)


def test_two_steps_on_circle_accumulate_time_and_positions():
    """
    Po dvou krocích na kružnici:
    s0: (0,0) -> s1: (1,0), dir=(-1,0), dt=1/speed
    s1: (1,0) -> s2: (-1,0), dir=(+1,0), dt=2/speed (přes celý průměr)
    Celkový čas: 3/speed.
    """
    a = b = 1.0
    speed = 2.0
    s0 = State(np.array([0.0, 0.0]), np.array([1.0, 0.0]), speed=speed, time=0.0)

    s1 = next_step(s0, a, b)
    s2 = next_step(s1, a, b)

    assert np.allclose(s1.pos, [1.0, 0.0])
    assert np.allclose(s1.dir, [-1.0, 0.0])
    assert s1.time == pytest.approx(1.0 / speed)

    assert np.allclose(s2.pos, [-1.0, 0.0])
    assert np.allclose(s2.dir, [1.0, 0.0])
    assert s2.time == pytest.approx((1.0 + 2.0) / speed)  # 1.5 pro speed=2


def test_step_on_ellipse_major_axis():
    """
    Elipsa a=2, b=1. Start v (0,0), směr +x.
    První zásah v (2,0); normála je (1,0); odraz -> směr (-1,0).
    Čas do nárazu = 2 / speed.
    """
    a, b = 2.0, 1.0
    speed = 1.0
    s0 = State(np.array([0.0, 0.0]), np.array([1.0, 0.0]), speed=speed, time=0.0)

    s1 = next_step(s0, a, b)

    assert np.allclose(s1.pos, [2.0, 0.0])
    assert np.allclose(s1.dir, [-1.0, 0.0])
    assert s1.time == pytest.approx(2.0 / speed)


def test_run_returns_states_list():
    a = b = 1.0
    speed = 2.0
    s0 = State(pos=np.array([0.0, 0.0]), dir=np.array([1.0, 0.0]), speed=speed, time=0.0)
    states = run(s0, a, b, max_bounces=3)
    assert isinstance(states, list)
    assert len(states) == 4  # initial + 3 bounces
    assert np.allclose(states[0].pos, [0.0, 0.0])
    assert np.allclose(states[1].pos, [1.0, 0.0])
    assert np.allclose(states[2].pos, [-1.0, 0.0])
    assert np.allclose(states[3].pos, [1.0, 0.0])


def test_run_stops_at_max_time():
    a = b = 1.0
    speed = 1.0
    s0 = State(pos=np.array([0.0, 0.0]), dir=np.array([1.0, 0.0]), speed=speed, time=0.0)
    states = run(s0, a, b, max_bounces=100, max_time=1.0)
    # Should not exceed max_time
    assert all(s.time <= 1.0 + 1e-9 for s in states)


def test_run_returns_last_state_when_collect_states_false():
    a = b = 1.0
    speed = 1.0
    s0 = State(pos=np.array([0.0, 0.0]), dir=np.array([1.0, 0.0]), speed=speed, time=0.0)
    last_state = run(s0, a, b, max_bounces=5, collect_states=False)
    assert isinstance(last_state, State)
