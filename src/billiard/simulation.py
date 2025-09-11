from billiard.physics import advance_with_time
from billiard.state import State
from typing import List, Optional

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



def run(initial: State, a: float, b: float, *,
        max_bounces: int = 1000,
        max_time: Optional[float] = None,
        collect_states: bool = True) -> tuple[List[State], int]:
    """
    Spustí simulaci od počátečního stavu.
    Vrátí seznam stavů (nebo jen výslednou pozici, podle collect_states) a počet odrazů.
    """
    bounces = 0
    states: List[State] = [initial] if collect_states else []
    current_state = initial
    
    while bounces < max_bounces:
        if max_time is not None and current_state.time >= max_time:
            break
        
        current_state = next_step(current_state, a, b)
        if collect_states:
            states.append(current_state)
        
        bounces += 1
    
    if not collect_states:
        return current_state, bounces
    
    return states, bounces
