
import os
import sys
from ase import Atoms
from matplotlib import pyplot as plt
from dataclasses import dataclass
import torch
from typing import List
import numpy as np
from time import time

from Utils import load_data, timed
from datatypes import GradientDescentResult
from .strategies import StrategyBase
from .GradientDescentParameters import GDParams

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
import schnetpack.transform as trn
    
@dataclass(init = True, repr = True)
class BacktrackingLineSearchParams:
    active: bool = False
    
    alpha: float = 1e-3
    beta: float = 0.5
    

@timed
def gradient_descent(
    atoms, 
    strategy: StrategyBase,
    lineSearchParams: BacktrackingLineSearchParams = BacktrackingLineSearchParams()
    ) -> GradientDescentResult:
    score_history = []
    time_history = []
    position_history = []
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    inputs = converter(atoms)
    
    gradientDescentParams = strategy.gradient_descent_params
    step_size = gradientDescentParams.step_size
    max_step_size = gradientDescentParams.max_step_size
    
    i = 0
    counter = 0
    t0 = time()
    while True:
        energy, forces, direction = strategy.get_direction(inputs.copy())
        direction = direction.to(dtype=torch.float32)
        
        # logging
        score_history.append({
            "basic": energy.item(),
            "force_norm": torch.linalg.norm(forces).item(),
            "direction_norm": torch.linalg.norm(direction).item()
        })
        time_history.append(time() - t0)
        position_history.append(inputs[spk.properties.R].detach().numpy())
        
        # update position
        delta_positions = determine_step(max_step_size, step_size * direction)
        inputs[spk.properties.R] = inputs[spk.properties.R] + delta_positions
         
        # Line search logic
        # if lineSearchParams.active:
        #     line_search_counter = 0  # Optional: to limit line search iterations
        #     while True:
        #         new_energy, _, _ = strategy.get_direction(inputs.copy())
        #         improvement = lineSearchParams.alpha * step_size * torch.dot(
        #             forces.flatten(), direction.flatten().to(dtype=torch.float64)
        #         )
        #         if new_energy < energy + improvement:
        #             break
        #         step_size *= lineSearchParams.beta
        #         inputs[spk.properties.R] -= step_size * direction
                
        #         line_search_counter += 1
        #         if line_search_counter > 50:  # Prevent infinite loop
        #             print("Line search iteration limit reached.")
        #             break
        
        # break if the energy has not improved in the last 10 steps
        if i > 30 and score_history[-1]["basic"] > score_history[-30]["basic"]:
            break
        if i > 100:
            break
        
        # counter += 1
        i += 1
    
    # start should be 0
    time_history[0] = 0
    
    # end with the best score
    best_idx = min(range(len(score_history)), key = lambda i: score_history[i]["basic"])
    score_history.append(score_history[best_idx])
    time_history.append(time() - t0)
    position_history.append(position_history[best_idx])
    
    return GradientDescentResult(score_history, position_history, time_history)
    
def determine_step(max_step: float, delta_positions: torch.Tensor) -> torch.Tensor:
    step_lengths = (delta_positions**2).sum(1)**0.5 # l2-distance each atom would move
    longest_step = torch.max(step_lengths)
    if longest_step >= max_step:
        delta_positions *= max_step / longest_step

    return delta_positions
    
device = "cpu"

if __name__ == "__main__":
    data = load_data()
    # strategy = AutoDiffHessianStrategy(data.test_dataset[0])
    
    # structure = data.test_dataset[0]
    # atoms = Atoms(
    #     numbers=structure[spk.properties.Z], 
    #     positions=structure[spk.properties.R]
    # )
    
    # res, t = gradient_descent(atoms, strategy)
    
    # res.plot_score()
    
    # print("END")