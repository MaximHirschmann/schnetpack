
from dataclasses import dataclass


@dataclass(init = True, repr = True)
class GDParams:
    max_step_size: float = 0.2
    step_size: float = 0.01