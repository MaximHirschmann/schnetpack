"""
Gradient Descent with Gradient vs Newton's Step
"""

import sys
import os 
from plotting import plot, plot_structure, plot_all_histories, plot_average
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



class StrategyBase:
    def __init__(self, name: str = "", line_search: bool = True) -> None:
        self.name = name
        self.line_search = line_search
        
    def prepare_energy_and_forces(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.array, np.array]:
        inputs_copy = inputs.copy()
        inputs_copy = energy_model(inputs_copy)
        
        return (
            inputs_copy["energy"].detach().cpu().numpy(), 
            inputs_copy["forces"].detach().cpu().numpy()
            )
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.array, np.array, np.array]:
        raise NotImplementedError
    
class ForcesStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Forces", line_search)
        
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.array, np.array, np.array]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        norm: np.array = np.linalg.norm(forces)
        direction: np.array = forces / norm
        
        return energy, forces, direction
    
class HessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.array, np.array, np.array]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = hessian_model(inputs_copy)
        hessian: np.array = inputs_copy["hessian"].detach().cpu().numpy()
        
        newton_step: np.array = np.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: np.array = np.linalg.norm(newton_step)
        direction: np.array = newton_step / norm
        
        return energy, forces, direction
    
class NewtonStepStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Newton Step", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.array, np.array, np.array]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = newton_step_model(inputs_copy)
        newton_step: np.array = inputs_copy["newton_step"].detach().cpu().numpy()
        
        norm: np.array = np.linalg.norm(newton_step)
        direction: np.array = -1 * newton_step / norm
        
        return energy, forces, direction
    
class InvHessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Inv Hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.array, np.array, np.array]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = inv_hessian_model(inputs_copy)
        inv_hessian: np.array = inputs_copy["inv_hessian"].detach().cpu().numpy()
        
        newton_step: np.array = np.dot(inv_hessian, forces.flatten()).reshape(forces.shape)
        norm: np.array = np.linalg.norm(newton_step)
        direction: np.array = newton_step / norm
        
        return energy, forces, direction
    
class AvgHessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Avg Hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[np.array, np.array, np.array]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        newton_step: np.array = np.dot(avg_hessian, forces.flatten()).reshape(forces.shape)
        norm: np.array = np.linalg.norm(newton_step)
        direction: np.array = newton_step / norm
        
        return energy, forces, direction


@timed
def gradient_descent(atoms, strategy: StrategyBase):
    history = []
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    
    tolerance = 5e-3
    step_size = 0.05
    rho_pos = 1.2
    rho_neg = 0.5
    rho_ls = 1e-4
    
    i = 0
    counter = 0
    while True:
        energy, forces, direction = strategy.get_direction(converter(atoms))
        
        history.append(energy.item())
        
        if i > 100:
            break
        elif step_size < tolerance:
            counter += 1
            if counter == 10:
                break
        else:
            counter = 0
            
        # line search
        if strategy.line_search:
            # modify step size 
            while True:
                new_atoms = Atoms(
                    numbers=atoms.numbers, 
                    positions=atoms.positions + step_size * direction
                )
                new_energy, new_forces = strategy.prepare_energy_and_forces(converter(new_atoms))
                if new_energy < energy + rho_ls * step_size * np.dot(forces.flatten(), direction.flatten()):
                    break
                step_size = step_size * rho_neg
                if step_size < 1e-5:
                    break
        
            
        atoms.positions = atoms.positions + step_size * direction
        if strategy.line_search:
            step_size = min(step_size * rho_pos, 0.05)
        
        if not strategy.line_search:
            # break if energy hasnt changed by 0.001 in the last 10 steps
            if len(history) > 30 and history[-1] - history[-10] >= -0.0001:
                break
            
        counter += 1
        i += 1
    
    return history
    

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
        AvgHessianStrategy()
    ]

    for i in random.sample(range(len(data.test_dataset)), 10):
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
    avg_hessian = np.load(os.getcwd() + "\\maxim\\data\\avg_hessian.npy")

    main()