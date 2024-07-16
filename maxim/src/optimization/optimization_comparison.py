"""
Gradient Descent with Gradient vs Newton's Step
"""

import sys
import os 

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator

sys.path.insert(1, schnetpack_dir + "\\maxim\\src")
from plotting  import plot_average, plot_all_histories
from Utils import load_data
from strategies import *
from gradient_descent import GradientDescentParameters, gradient_descent

import torch
from ase import Atoms
import matplotlib.pyplot as plt
import random


    
def compare():
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

    N = 3
    for i in random.sample(range(len(data.test_dataset)), N):
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
    compare()
    # hyperparameter_search(load_data())