from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class State:
    pos: np.ndarray     # current position
    dir: np.ndarray     # normalized direction
    speed: float = 1.0
    time: float = 0.0

