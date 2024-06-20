"""
Gradient Descent with Gradient vs Newton's Step
"""

import sys
import os 
from time import time
from typing import List
from plotting import plot, plot_structure
from Utils import load_model, load_data, timed

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator

import torch
import torchmetrics
import pytorch_lightning as pl
from ase import Atoms
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time


def evaluate_atom(inputs, properties: List[str]):
    outputs = {}
    if "energy" in properties or "forces" in properties:
        temp = inputs.copy()
        temp = energy_model(temp)
        outputs["energy"] = temp["energy"].detach().cpu().numpy()
        outputs["forces"] = temp["forces"].detach().cpu().numpy()
    if "hessian" in properties:
        temp = inputs.copy()
        temp = hessian_model(temp)
        outputs["hessian"] = temp["hessian"].detach().cpu().numpy()
    if "newton_step" in properties:
        temp = inputs.copy()
        temp = newton_step_model(temp)
        outputs["newton_step"] = temp["newton_step"].detach().cpu().numpy()
    return outputs
    
def get_direction(outputs, strategy):
    if strategy == "forces":
        forces = outputs["forces"]
        norm = np.linalg.norm(forces)
        
        direction = forces / norm
    
    elif strategy == "hessian":
        hessian = outputs["hessian"]
        forces = outputs["forces"]
        newton_step = np.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm = np.linalg.norm(newton_step)
        
        direction = newton_step / norm
        
    elif strategy == "newton_step":
        newton_step = outputs["newton_step"]
        norm = np.linalg.norm(newton_step)
        
        direction = -1 * newton_step / norm
        
    else:
        raise ValueError("Invalid strategy")
    
    return direction
    
def get_energy(outputs):
    return outputs["energy"]

def get_forces(outputs):
    return outputs["forces"]

@timed
def gradient_descent(atoms, strategy, line_search=True):
    history = []
    
    properties = ["energy", "forces"]
    if strategy == "hessian":
        properties.append("hessian")
    elif strategy == "newton_step":
        properties.append("newton_step")
        
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cuda"
    )
    
    tolerance = 5e-3
    step_size = 0.05 if line_search else 0.05
    rho_pos = 1.2
    rho_neg = 0.5
    rho_ls = 1e-4
    
    counter = 0
    while True:
        outputs = evaluate_atom(converter(atoms), properties)
        direction = get_direction(outputs, strategy)
        energy = get_energy(outputs)
        forces = get_forces(outputs)
        
        history.append(energy.item())
        
        if step_size < tolerance:
            counter += 1
            if counter == 10:
                break
        else:
            counter = 0
            
        # line search
        if line_search:
            # modify step size 
            while True:
                new_atoms = Atoms(
                    numbers=atoms.numbers, 
                    positions=atoms.positions + step_size * direction
                )
                outputs = evaluate_atom(converter(new_atoms), ["energy", "forces"])
                if get_energy(outputs) < energy + rho_ls * step_size * np.dot(forces.flatten(), direction.flatten()):
                    break
                step_size = step_size * rho_neg
        
            
        atoms.positions = atoms.positions + step_size * direction
        if line_search:
            step_size = min(step_size * rho_pos, 0.05)
        
        if not line_search:
            # break if energy hasnt changed by 0.001 in the last 10 steps
            if len(history) > 30 and history[-1] - history[-10] >= -0.0001:
                break
            
        counter += 1
    
    return history
    

def plot_average(histories: List[List[str]], labels, title = "Average Energy History"):
    # convert each history to numpy array and make them the same length
    max_length = max([len(history) for history in histories[0]])
    np_histories = np.zeros((len(histories[0]), max_length))
    for i in range(len(histories[0])):
        for j in range(len(histories)):
            fitted = histories[j][i][:max_length]
            if len(fitted) < max_length:
                fitted += [fitted[-1]] * (max_length - len(fitted))
            np_histories[i] += fitted
    np_histories /= len(histories)    

    # plot average
    plt.figure()
    
    for i, label in enumerate(labels):
        plt.plot(np_histories[i], label=label)
    
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_all_histories(histories: List[List[str]], labels, title = "All Energy Histories"):
    cols = 6
    fig, axs = plt.subplots(int(np.ceil(len(histories) / cols)), cols, figsize=(15, 15))
    
    for i, history in enumerate(histories):
        if len(histories) <= cols:
            ax = axs[i]
        else:
            ax = axs[i // cols, i % cols]
        for j, strategy_history in enumerate(histories[i]):
            ax.plot(strategy_history, label=labels[j])
            
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.suptitle(title)
    plt.show()
        

def main():
    data = load_data()
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cuda"
    )

    histories = []
    
    for i in random.sample(range(len(data.test_dataset)), 3):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], 
            positions=structure[spk.properties.R]
        )

        energy_0 = evaluate_atom(converter(atoms), ["energy"])["energy"]
        history_1, t1 = gradient_descent(atoms.copy(), "forces", line_search=False)
        history_2, t2 = gradient_descent(atoms.copy(), "hessian", line_search=False)
        history_3, t3 = gradient_descent(atoms.copy(), "newton_step", line_search=False)
        
        history_4, t4 = gradient_descent(atoms.copy(), "forces", line_search=True)
        history_5, t5 = gradient_descent(atoms.copy(), "hessian", line_search=True)
        history_6, t6 = gradient_descent(atoms.copy(), "newton_step", line_search=True)
                
        print(f"Structure {i}")
        print(f"T0 Energy: {energy_0}")
        print(f"Result Forces: {history_1[-1]} in {len(history_1)} steps in {t1} seconds")
        print(f"Result Hessian: {history_2[-1]} in {len(history_2)} steps in {t2} seconds")
        print(f"Result Newton Step: {history_3[-1]} in {len(history_3)} steps in {t3} seconds")
        print(f"Result Forces with Line Search: {history_4[-1]} in {len(history_4)} steps in {t4} seconds")
        print(f"Result Hessian with Line Search: {history_5[-1]} in {len(history_5)} steps in {t5} seconds")
        print(f"Result Newton Step with Line Search: {history_6[-1]} in {len(history_6)} steps in {t6} seconds")
        print()
        
        histories.append([history_1, history_2, history_3, history_4, history_5, history_6])
        
    labels = ["Forces", "Hessian", "Newton Step", "Forces LS", "Hessian LS", "Newton Step LS"]
    plot_average(histories, labels)
    plot_all_histories(histories, labels)
    


if __name__ == "__main__":
    energy_model = load_model("energy_model")
    hessian_model = load_model("hessian1")
    newton_step_model = load_model("newton_step")

    main()