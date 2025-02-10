
from dataclasses import dataclass


@dataclass(init = True, repr = True)
class GDParams:
    max_step_size: float = 0.2
    step_size: float = 0.01
    max_iterations: int = 200
    momentum: float = 0.8
    
    force_threshold: float = 0.001
    
    momentum_only_good_directions: bool = False