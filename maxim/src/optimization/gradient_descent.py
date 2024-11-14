
import os
import sys

from ase import Atoms
from matplotlib import pyplot as plt

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
import schnetpack.transform as trn

sys.path.insert(1, schnetpack_dir + "\\maxim\\src")
from strategies import AutoDiffHessianStrategy, StrategyBase
from Utils import load_data, timed

from dataclasses import dataclass
import torch
from time import time



@dataclass(init = True, repr = True)
class GradientDescentParameters:
    tolerance: float = 5e-3
    max_step_size: float = 0.05
    rho_pos: float = 1.2
    rho_neg: float = 0.5
    rho_ls: float = 1e-4
    
    autodiff_muh: float = 1
    
@dataclass(init = True, repr = True)
class BacktrackingLineSearchParams:
    active: bool = False
    
    alpha: float = 0.1
    beta: float = 0.5
    

class GradientDescentResult:
    def __init__(self, score_history: list, time_history: list, final_atom: Atoms) -> None:
        self.score_history = score_history
        self.time_history = time_history
        self.final_atom = final_atom
        
        self.total_time = time_history[-1]
        self.total_steps = len(score_history)

    def plot_score(self):
        plt.plot(self.time_history, self.score_history, "ro-")
        plt.show()

@timed
def gradient_descent(
    atoms, 
    strategy: StrategyBase,
    gradientDescentParams: GradientDescentParameters = GradientDescentParameters(),
    lineSearchParams: BacktrackingLineSearchParams = BacktrackingLineSearchParams()
    ) -> GradientDescentResult:
    score_history = []
    time_history = []
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    inputs = converter(atoms)
    
    step_size = gradientDescentParams.max_step_size
    
    i = 0
    counter = 0
    t0 = time()
    while True:
        energy, forces, direction = strategy.get_direction(inputs.copy())
        
        score_history.append(energy.item())
        time_history.append(time() - t0)
        
        inputs[spk.properties.R] = inputs[spk.properties.R] + step_size * direction
        
        if lineSearchParams.active:
            while True:
                new_energy, _, _ = strategy.get_direction(inputs.copy())
                if new_energy < energy + lineSearchParams.alpha * step_size * torch.dot(forces, direction):
                    break
                step_size *= lineSearchParams.beta
                inputs[spk.properties.R] = inputs[spk.properties.R] - step_size * direction
        
        # break if the energy has not improved in the last 10 steps
        if i > 30 and score_history[-1] > score_history[-30]:
            break
        
        # counter += 1
        i += 1
    
    time_history[0] = 0
    final_atom = atoms
    final_atom.positions = inputs[spk.properties.R].detach().numpy()
    
    return GradientDescentResult(score_history, time_history, final_atom)
    
    
device = "cpu"

if __name__ == "__main__":
    data = load_data()
    strategy = AutoDiffHessianStrategy(data.test_dataset[0])
    
    structure = data.test_dataset[0]
    atoms = Atoms(
        numbers=structure[spk.properties.Z], 
        positions=structure[spk.properties.R]
    )
    
    res, t = gradient_descent(atoms, strategy)
    
    res.plot_score()
    
    print("END")