"""
Gradient Descent with Gradient vs Newton's Step
"""

from dataclasses import dataclass
import sys
import os 
from plotting import plot, plot_structure, plot_all_histories, plot_average, plot_hessian
from Utils import load_model, load_data, timed

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator

import torch
import pytorch_lightning as pl
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
from typing import Dict, List, Tuple
from tabulate import tabulate



class StrategyBase:
    def __init__(self, name: str = "", line_search: bool = True) -> None:
        self.name = name
        self.line_search = line_search
        
    def prepare_energy_and_forces(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_copy = inputs.copy()
        inputs_copy = energy_model(inputs_copy)
        
        return (
            inputs_copy["energy"], 
            inputs_copy["forces"]
            )
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
class ForcesStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Forces", line_search)
        
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        norm: torch.Tensor = torch.linalg.norm(forces)
        direction: torch.Tensor = forces / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class HessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = hessian_model(inputs_copy)
        hessian: torch.Tensor = inputs_copy["hessian"]
        
        newton_step: torch.Tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class NewtonStepStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Newton Step", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = newton_step_model(inputs_copy)
        newton_step: torch.Tensor = inputs_copy["newton_step"]
        
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = -1 * newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class InvHessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Inv Hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = inv_hessian_model(inputs_copy)
        inv_hessian: torch.Tensor = inputs_copy["inv_hessian"]
        
        newton_step: torch.Tensor = (inv_hessian @ forces.flatten()).reshape(forces.shape)
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class AvgHessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Avg Hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        newton_step: torch.tensor = torch.linalg.solve(avg_hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.tensor = torch.linalg.norm(newton_step)
        direction: torch.tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
    
class AutoDiffWrapper(torch.nn.Module):
    def __init__(self, model, inputs_template):
        super(AutoDiffWrapper, self).__init__()
        self.model = model
        self.inputs_template = inputs_template
        
    def forward(self, positions):
        inputs = self.inputs_template.copy()
        inputs[spk.properties.R] = positions
        return self.model(inputs)["energy"]
    
class AutoDiffHessianStrategy(StrategyBase):
    def __init__(self, 
                 example_structure, 
                 line_search: bool = True,
                 muh: float = 1) -> None:
        super().__init__("AutoDiff Hessian", line_search)
        
        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
        )
        
        example_atoms = Atoms(
            numbers=example_structure[spk.properties.Z],
            positions=example_structure[spk.properties.R],
        )
        inputs_template = converter(example_atoms)
        
        self.muh = muh
        self.wrapper = AutoDiffWrapper(energy_model, inputs_template)
        
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        hessian = torch.autograd.functional.hessian(self.wrapper, inputs_copy[spk.properties.R])
        hessian = hessian.to(dtype=torch.float64)
        hessian = torch.reshape(hessian, (27, 27))
        hessian += self.muh * torch.eye(27, device = device, dtype = torch.float64)
        
        newton_step: torch.tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.tensor = torch.linalg.norm(newton_step)
        direction: torch.tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
        
@dataclass(init = True, repr = True)
class GradientDescentParameters:
    tolerance: float = 1e-2
    max_step_size: float = 0.05
    rho_pos: float = 1.2
    rho_neg: float = 0.5
    rho_ls: float = 1e-4
    

@timed
def gradient_descent(
    atoms, 
    strategy: StrategyBase,
    params: GradientDescentParameters = GradientDescentParameters()
    ):
    history = []
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    inputs = converter(atoms)
    
    step_size = params.max_step_size
    
    i = 0
    counter = 0
    while True:
        energy, forces, direction = strategy.get_direction(inputs.copy())
        
        history.append(energy.item())
        
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
            if len(history) > 30 and history[-1] - history[-10] >= -0.0001:
                break
            
        counter += 1
        i += 1
    
    return history
    

class StrategyEvalutionResult:
    strategy: StrategyBase
    gradient_descent_params: GradientDescentParameters
    avg_score: float
    avg_steps: float
    avg_time: float
    

def strategy_evaluation(data, strategy: StrategyBase, N: int = 100, 
                        gradient_descent_params: GradientDescentParameters = GradientDescentParameters()):
    t0 = time()
    steps = 0
    avg_score = 0
    for i in random.sample(range(len(data.test_dataset)), N):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z],
            positions=structure[spk.properties.R],
        )
        
        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
        )
        
        history, t = gradient_descent(atoms, strategy, gradient_descent_params)
        steps += len(history)
        avg_score += history[-1]
    
    avg_time = (time() - t0) / N
    avg_score /= N
    steps /= N
    
    result = StrategyEvalutionResult()
    result.strategy = strategy
    result.gradient_descent_params = gradient_descent_params
    result.avg_score = avg_score
    result.avg_steps = steps
    result.avg_time = avg_time
    
    print(f"Strategy {strategy.name} N: {N} finished with avg score {avg_score} in {steps} steps in {t} seconds")
    
    return result
    
def hyperparameter_search(data):
    params = [
        GradientDescentParameters(),
        GradientDescentParameters(tolerance = 1e-3),
        GradientDescentParameters(tolerance = 2e-2),
        GradientDescentParameters(max_step_size = 0.1),
        GradientDescentParameters(max_step_size = 0.01),
        GradientDescentParameters(rho_pos = 1.1),
        GradientDescentParameters(rho_pos = 1.6),
        GradientDescentParameters(rho_neg = 0.6),
        GradientDescentParameters(rho_neg = 0.3),
        GradientDescentParameters(rho_ls = 1e-3),
        GradientDescentParameters(rho_ls = 1e-5)
    ]
    results = []
    for param in params:
        strategy = HessianStrategy()
        results.append(strategy_evaluation(data, strategy, N = 100, gradient_descent_params = param))
    
    table = [] 
    for result in results:
        table.append([result.strategy.name, result.gradient_descent_params, result.avg_score, result.avg_steps, result.avg_time])
        
    print(tabulate(table, headers=["Strategy", "Muh", "Avg Score", "Avg Steps", "Avg Time"]))
    
    
def main():
    data = load_data()
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    histories = []
    
    strategies = [
        ForcesStrategy(),
        HessianStrategy(),
        NewtonStepStrategy(),
        InvHessianStrategy(),
        AvgHessianStrategy(),
        AutoDiffHessianStrategy(data.test_dataset[0])
    ]

    for i in random.sample(range(len(data.test_dataset)), 4):
        histories.append([])
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], 
            positions=structure[spk.properties.R]
        )

        energy_0, _ = StrategyBase().prepare_energy_and_forces(converter(atoms))
        print(f"Initial energy: {energy_0.item()}")
        for strategy in strategies:
            history, t = gradient_descent(atoms.copy(), strategy)
            print(f"Result {strategy.name}: {history[-1]} in {len(history)} steps in {t} seconds")
            
            histories[-1].append(history)
        print()

    labels = [strategy.name for strategy in strategies]
    plot_average(histories, labels)
    plot_all_histories(histories, labels)
    


if __name__ == "__main__":
    device = "cpu"
    energy_model = load_model("energy_model", device=device)
    hessian_model = load_model("hessian1", device=device)
    newton_step_model = load_model("newton_step", device=device)
    inv_hessian_model = load_model("inv_hessian2", device=device)
    avg_hessian = torch.tensor(np.load(os.getcwd() + "\\maxim\\data\\avg_hessian.npy"), dtype=torch.float64, device=device)

    # main()
    hyperparameter_search(load_data())