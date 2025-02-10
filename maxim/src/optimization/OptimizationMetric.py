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
        super().__init__(name, "Energy in kcal/mol")
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
        super().__init__("ClosestMinimum", "Euclidean distance to closest minimum in Angstrom")
        self.molecule_numbers = molecule_numbers
        
        # Load conformations from .xyz file
        with open("./maxim/src/optimization/found_conformers_run0.xyz", "r") as f:
            self.conformers = ase.io.read(f, index=":")
        print("Loaded conformers")
        
    def calculate_energy(self, atom_position: np.array) -> float:
        min_distance = float("inf")
        # the atoms (3, 4) may and (5, 6, 7) may be swapped 
        # we need to check all permutations
        for perm1 in [(3, 4)]: # permutations(range(3, 5)):
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
            
    def plot_conformers(self):
        fig = plt.figure(figsize=(14, 6), dpi=300)
        fig_positions = [131, 132, 133]
        titles = ["Local Optimum 1", "Global Optimum", "Local Optimum 2"]
        
        for j, i in enumerate([1, 0, 2]):
            conformer = self.conformers[i]
            ax = fig.add_subplot(fig_positions[j], projection='3d')
            self.plot_atom(conformer, ax, titles[j])
        
        plt.tight_layout()
        plt.savefig("conformers.svg", bbox_inches='tight')
        plt.savefig("conformers.png", dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_atom(self, atom, ax=None, title=""):
        from matplotlib.lines import Line2D
        positions = atom.positions
        numbers = atom.numbers

        # Define visual properties
        colors = {6: "black", 8: "red", 1: "green"}

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Draw bonds
        for i, j in [(0, 2), (2, 8), (0, 1), (0, 3), (0, 4), (1, 7), (1, 5), (1, 6)]:
            x = [positions[i][0], positions[j][0]]
            y = [positions[i][1], positions[j][1]]
            z = [positions[i][2], positions[j][2]]
            ax.plot(x, y, z, c="#A9A9A9", linewidth=5)

        # Plot atoms
        for i, (x, y, z) in enumerate(positions):
            ax.scatter(x, y, z, c=colors[numbers[i]], s=400, edgecolor="k")

        # Remove axis ticks and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        # Set zoomed-in axis limits
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlim([-1.5, 1.5])
    
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='C', markersize=16),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='O', markersize=16),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='H', markersize=16)
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=16)
        
        # Title and view angle
        ax.set_title(title, fontsize=20, pad=15)
        ax.view_init(elev=20, azim=20)
                
    
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