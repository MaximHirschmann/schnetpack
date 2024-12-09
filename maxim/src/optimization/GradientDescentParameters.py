
from dataclasses import dataclass


@dataclass(init = True, repr = True)
class GDParams:
    tolerance: float = 5e-3
    max_step_size: float = 0.01
    rho_pos: float = 1.2
    rho_neg: float = 0.5
    rho_ls: float = 1e-4
    
    autodiff_muh: float = 1