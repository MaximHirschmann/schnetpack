
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
from .strategies import AutoDiffHessianStrategy, StrategyBase

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
import schnetpack.transform as trn





@dataclass(init = True, repr = True)
class GradientDescentParameters:
    tolerance: float = 5e-3
    max_step_size: float = 1e-2
    rho_pos: float = 1.2
    rho_neg: float = 0.5
    rho_ls: float = 1e-4
    
    autodiff_muh: float = 1
    
@dataclass(init = True, repr = True)
class BacktrackingLineSearchParams:
    active: bool = False
    
    alpha: float = 0.1
    beta: float = 0.5
    

@timed
def gradient_descent(
    atoms, 
    strategy: StrategyBase,
    gradientDescentParams: GradientDescentParameters = GradientDescentParameters(),
    lineSearchParams: BacktrackingLineSearchParams = BacktrackingLineSearchParams()
    ) -> GradientDescentResult:
    score_history = []
    time_history = []
    position_history = []
    
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
        
        score_history.append({
            "basic": energy.item(),
        })
        time_history.append(time() - t0)
        position_history.append(inputs[spk.properties.R].detach().numpy())
        
        inputs[spk.properties.R] = inputs[spk.properties.R] + step_size * direction
        
        if lineSearchParams.active:
            while True:
                new_energy, _, _ = strategy.get_direction(inputs.copy())
                if new_energy < energy + lineSearchParams.alpha * step_size * torch.dot(forces, direction):
                    break
                step_size *= lineSearchParams.beta
                inputs[spk.properties.R] = inputs[spk.properties.R] - step_size * direction
        
        # break if the energy has not improved in the last 10 steps
        if i > 30 and score_history[-1]["basic"] > score_history[-30]["basic"]:
            break
        if i > 300:
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