
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
    tolerance: float = 1e-2
    max_step_size: float = 0.05
    rho_pos: float = 1.2
    rho_neg: float = 0.5
    rho_ls: float = 1e-4
    
    autodiff_muh: float = 1
    

class GradientDescentResult:
    def __init__(self, score_history: list, time_history: list):
        self.score_history = score_history
        self.time_history = time_history
        
        self.total_time = time_history[-1]
        self.total_steps = len(score_history)

    def plot_score(self):
        plt.plot(self.time_history, self.score_history, "ro-")
        plt.show()

@timed
def gradient_descent(
    atoms, 
    strategy: StrategyBase,
    params: GradientDescentParameters = GradientDescentParameters()
    ) -> GradientDescentResult:
    score_history = []
    time_history = []
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    inputs = converter(atoms)
    
    step_size = params.max_step_size
    
    i = 0
    counter = 0
    t0 = time()
    while True:
        energy, forces, direction = strategy.get_direction(inputs.copy())
        
        score_history.append(energy.item())
        time_history.append(time() - t0)
        
        if i > 100:
            break
        elif step_size < params.tolerance:
            counter += 1
            if counter == 10:
                break
        else:
            counter = 0
            
        # line search
        if strategy.line_search:
            # modify step size 
            while True:
                new_inputs = inputs.copy()
                new_inputs[spk.properties.R] = inputs[spk.properties.R] + step_size * direction
                
                new_energy, new_forces = strategy.prepare_energy_and_forces(new_inputs)
                if new_energy < energy + params.rho_ls * step_size * torch.dot(forces.flatten().to(dtype=torch.float32), direction.flatten()):
                    break
                step_size = step_size * params.rho_neg
                if step_size < 1e-5:
                    break
            
        inputs[spk.properties.R] = inputs[spk.properties.R] + step_size * direction
        if strategy.line_search:
            step_size = min(step_size * params.rho_pos, params.max_step_size)
        
        if not strategy.line_search:
            # break if energy hasnt changed by 0.001 in the last 10 steps
            if len(score_history) > 30 and score_history[-1] - score_history[-10] >= -0.0001:
                break
            
        counter += 1
        i += 1
    
    time_history[0] = 0
    return GradientDescentResult(score_history, time_history)
    
    
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