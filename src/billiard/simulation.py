from billiard.physics import advance_with_time, advance_with_time_shape
from billiard.state import State
from typing import List, Optional
from billiard.shapes import EllipseShape, Shape

def next_step(state: State, a: float, b: float) -> State:
    """
    Executes one step of the simulation: moves the ball from the current state to the impact point
    and returns the new state.
    """
    impact, new_dir, dt, t = advance_with_time(state.pos, state.dir, state.speed, a, b)

    return State(
        pos = impact,
        dir = new_dir,
        speed = state.speed,
        time = state.time + dt,
    )


def next_step_shape(state: State, shape: Shape) -> State:
    impact, new_dir, dt, t = advance_with_time_shape(state.pos, state.dir, state.speed, shape)
    return State(
        pos=impact,
        dir=new_dir,
        speed=state.speed,
        time=state.time + dt,
    )


def run(initial: State, a: float, b: float, *,
        max_bounces: int = 1000,
        max_time: Optional[float] = None,
        collect_states: bool = True) -> tuple[List[State], int]:
    """
    Spustí simulaci od počátečního stavu.
    Vrátí seznam stavů (nebo jen výslednou pozici, podle collect_states) a počet odrazů.
    """
    return run_shape(initial, EllipseShape(a, b),
                     max_bounces=max_bounces,
                     max_time=max_time,
                     collect_states=collect_states)


def run_shape(initial: State, shape: Shape, *,
              max_bounces: int = 1000,
              max_time: Optional[float] = None,
              collect_states: bool = True) -> tuple[List[State], int]:
    bounces = 0
    states: List[State] = [initial] if collect_states else []
    current_state = initial

    while bounces < max_bounces:
        if max_time is not None and current_state.time >= max_time:
            break

        current_state = next_step_shape(current_state, shape)
        if collect_states:
            states.append(current_state)

        bounces += 1

    if not collect_states:
        return current_state, bounces

    return states, bounces
