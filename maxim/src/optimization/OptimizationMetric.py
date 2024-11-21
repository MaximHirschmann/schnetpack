from abc import ABC, abstractmethod

import os
import sys
from ase import Atoms
import numpy as np
import torch
from typing import Dict

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
import schnetpack.transform as trn


from ase.io.extxyz import read_xyz
import matplotlib.pyplot as plt

class OptimizationMetricInterface(ABC):
    def __init__(self, name, *args, **kwarg):
        self.name = name
    
    @abstractmethod
    def calculate_energy(self, atom_position: np.array) -> float:
        pass
    
    
class EnergyMetric(OptimizationMetricInterface):
    def __init__(self, energy_model, name: str, molecule_numbers, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(name)
        self.energy_model = energy_model
        
        self.converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
        )
        
        self.molecule_numbers = molecule_numbers
        
    def calculate_energy(self, atom_position: np.array) -> float:
        atom = Atoms(
            numbers=self.molecule_numbers,
            positions=atom_position
        )
        
        inputs: Dict[str, torch.Tensor] = self.converter(atom)
        
        return float(self.energy_model(inputs)['energy'].item())

class ClosestMinimumMetric(OptimizationMetricInterface):
    def __init__(self):
        super().__init__("ClosestMinimum")
        
        # load conformations from .xyz file
        with open("./maxim/src/optimization/found_conformers_run0.xyz", "r") as f:
            self.conformers = list(read_xyz(f, index = 0)) # three
        
    def calculate_energy(self, atom_position: np.array) -> float:
        min_distance = float("inf")
        
        for conformer in self.conformers:
            distance = np.linalg.norm(atom_position - conformer.positions)
            if distance < min_distance:
                min_distance = distance
                
        return min_distance
    
    
    
def plot_positions(positions, numbers):
    
    colors = {6: "black", 8: "red", 1: "blue"}
    colors = [colors[n] for n in numbers]
    charges = {6: "C", 8: "O", 1: "H"}
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # draw the bond between atom (0, 2) and (2, 8)
    for i, j in [(0, 2), (2, 8), (0, 1), (0, 3), (0, 4), (1, 7), (1, 5), (1, 6)]:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        z = [positions[i][2], positions[j][2]]
        ax.plot(x, y, z, c="gray")
        
    for i, (x, y, z) in enumerate(positions):
        ax.scatter(x, y, z, c=colors[i], label=i, s=100)
        ax.text(x, y, z + 0.15, f"{charges[numbers[i]]}", color='black')  # Shift the label slightly above the point
    
    
    # remove the axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    
    # rotate
    ax.view_init(elev=30, azim=20)
    
    plt.savefig("ethanol.png", dpi=300)
    
    # plt.legend()
    plt.show()