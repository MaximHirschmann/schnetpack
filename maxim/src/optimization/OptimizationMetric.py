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

import ase
from ase.io.extxyz import read_xyz
from ase.visualize import view
import matplotlib.pyplot as plt
from itertools import permutations

class OptimizationMetricInterface(ABC):
    def __init__(self, name, y_axis, *args, **kwarg):
        self.name = name
        self.y_axis = y_axis
    
    @abstractmethod
    def calculate_energy(self, atom_position: np.array) -> float:
        pass
    
    
class EnergyMetric(OptimizationMetricInterface):
    def __init__(self, energy_model, name: str, molecule_numbers, device = "cpu"):
        super().__init__(name, "Energy in Hartree")
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
    def __init__(self, molecule_numbers):
        super().__init__("ClosestMinimum", "Distance to closest minimum")
        self.molecule_numbers = molecule_numbers
        
        # Load conformations from .xyz file
        with open("./maxim/src/optimization/found_conformers_run0.xyz", "r") as f:
            self.conformers = ase.io.read(f, index=":")
        print("Loaded conformers")
        
    def calculate_energy(self, atom_position: np.array) -> float:
        min_distance = float("inf")
        # the atoms (3, 4) may and (5, 6, 7) may be swapped 
        # we need to check all permutations
        for perm1 in permutations(range(3, 5)):
            for perm2 in permutations(range(5, 8)):
                atom_positions2 = atom_position[[0, 1, 2, *perm1, *perm2, 8]]
                atom = Atoms(
                    numbers=self.molecule_numbers,
                    positions=atom_positions2
                )
                for conformer in self.conformers:
                    ase.build.minimize_rotation_and_translation(conformer, atom)
                    distance = np.linalg.norm(conformer.positions - atom.positions)
                    min_distance = min(min_distance, distance)
                    
        # ase.build.minimize_rotation_and_translation(conformer, atom)
        return min_distance
    
    def modify_positions(self, atom_positions: np.array) -> np.array:
        """
        Modify atom positions so that:
        - The first atom is at (0, 0, 0).
        - The second atom is along the first axis.
        - The third atom lies on the plane of the first and second axes.
        - The rest of the atoms are translated and rotated to maintain relative positions.

        Args:
            atom_positions (np.array): Array of shape (N, 3) containing the positions of N atoms.

        Returns:
            np.array: Modified positions of the atoms.
        """
        if len(atom_positions) < 3:
            raise ValueError("At least three atoms are required to define the transformations.")

        # Translate the first atom to the origin
        translation = -atom_positions[0]
        translated_positions = atom_positions + translation

        # Define the vector from the first atom to the second atom
        direction_vector = translated_positions[1]

        # Normalize the direction vector to create a rotation matrix
        direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
        x_axis = direction_vector_normalized

        # Define the vector from the first atom to the third atom
        vector_to_third_atom = translated_positions[2]

        # Project the third atom onto the plane defined by the first axis (x_axis)
        projection_onto_x = np.dot(vector_to_third_atom, x_axis) * x_axis
        projection_on_plane = vector_to_third_atom - projection_onto_x

        # Normalize to find the y-axis
        y_axis = projection_on_plane / np.linalg.norm(projection_on_plane)

        # Compute the z-axis as orthogonal to both x and y axes
        z_axis = np.cross(x_axis, y_axis)

        # Construct the rotation matrix
        rotation_matrix = np.array([x_axis, y_axis, z_axis])

        # Apply the rotation to all translated positions
        modified_positions = np.dot(translated_positions, rotation_matrix.T)

        return modified_positions
    
    def plot_all(self, atom):
        atom2 = atom.copy()
        atom2.positions = self.modify_positions(atom.positions)
        self.plot_atom(atom2, "Atom")
        for i, conformer in enumerate(self.conformers):
            conformer.positions = self.modify_positions(conformer.positions)
            self.plot_atom(conformer, f"Conformer {i}")
            
    def plot_atom(self, atom, title = ""):
        positions = atom.positions
        numbers = atom.numbers
        
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
            ax.text(x, y, z + 0.15, f"{i}", color='black')  # Shift the label slightly above the point
        
        # remove the axis
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        
        
        if title != "":
            plt.title(title)
        plt.show()
            
    
class ForceNormMetric(OptimizationMetricInterface):
    def __init__(self, name: str, device = "cpu"):
        super().__init__(name, "Norm of Forces")
        
    def calculate_energy(self, atom_position: np.array) -> float:
        return 0
    
    
    
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