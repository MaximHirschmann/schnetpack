
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
    
    def log(energy, forces, direction):
        score_history.append({
            "basic": energy.item(),
            # "force_norm": torch.linalg.norm(forces).item(),
            # "direction_norm": torch.linalg.norm(direction).item()
        })
        time_history.append(time() - t0)
        position_history.append(inputs[spk.properties.R].detach().numpy())
    
    mom_direction = torch.zeros_like(inputs[spk.properties.R])
    energy, forces, direction = strategy.get_direction(inputs.copy())
    log(energy, forces, direction)
    while True:
        direction = direction.to(dtype=torch.float32)
        
        new_mom_direction = gradientDescentParams.momentum * mom_direction + direction
        
        delta_positions = determine_step(max_step_size, step_size * new_mom_direction)
        
        # update position
        inputs[spk.properties.R] = inputs[spk.properties.R] + delta_positions
        
        new_energy, forces, direction = strategy.get_direction(inputs.copy())
        if strategy.gradient_descent_params.momentum_only_good_directions:
            if new_energy < energy:
                mom_direction = new_mom_direction
            else:
                mom_direction *= gradientDescentParams.momentum
        else:
            mom_direction = new_mom_direction
        energy = new_energy
        
        # logging
        log(energy, forces, direction)
        
        if torch.linalg.norm(direction) < gradientDescentParams.force_threshold:
            break
        
        if i > gradientDescentParams.max_iterations:
            break
        
        # counter += 1
        i += 1
    
    # start should be 0
    time_history[0] = 0
    
    # end with the best score
    if energy != 0:
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