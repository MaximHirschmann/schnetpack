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
        # temp = inv_hessian_model(temp)
        outputs["hessian"] = temp["hessian"].detach().cpu().numpy()
    if "newton_step" in properties:
        temp = inputs.copy()
        temp = newton_step_model(temp)
        outputs["newton_step"] = temp["newton_step"].detach().cpu().numpy()
    if "inv_hessian" in properties:
        temp = inputs.copy()
        temp = inv_hessian_model(temp)
        outputs["inv_hessian"] = temp["inv_hessian"].detach().cpu().numpy()
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
    elif strategy == "inv_hessian":
        inv_hessian = outputs["inv_hessian"]
        forces = outputs["forces"]
        newton_step = np.dot(inv_hessian, forces.flatten()).reshape(forces.shape)
        norm = np.linalg.norm(newton_step)
        direction = newton_step / norm
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
    
    match strategy:
        case "forces":
            properties = ["energy", "forces"]
        case "hessian": 
            properties = ["energy", "forces", "hessian"]
        case "newton_step":
            properties = ["energy", "forces", "newton_step"]
        case "inv_hessian":
            properties = ["energy", "forces", "inv_hessian"]
        
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    
    tolerance = 5e-3
    step_size = 0.05 if line_search else 0.05
    rho_pos = 1.2
    rho_neg = 0.5
    rho_ls = 1e-4
    
    i = 0
    counter = 0
    while True:
        outputs = evaluate_atom(converter(atoms), properties)
        direction = get_direction(outputs, strategy)
        energy = get_energy(outputs)
        forces = get_forces(outputs)
        
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
        i += 1
    
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
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )

    histories = []
    
    runs = [
        ["forces", True], ["hessian", True], ["newton_step", True], ["inv_hessian", True],
    ]

    for i in random.sample(range(len(data.test_dataset)), 4):
        histories.append([])
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], 
            positions=structure[spk.properties.R]
        )

        energy_0 = evaluate_atom(converter(atoms), ["energy"])["energy"]
        print(f"Initial energy: {energy_0.item()}")
        for strategy, line_search in runs:
            history, t = gradient_descent(atoms.copy(), strategy, line_search=line_search)
            print(f"Result {strategy} LS {line_search}: {history[-1]} in {len(history)} steps in {t} seconds")
            histories[-1].append(history)
        print()

    labels = [f"{strategy} LS" if line_search else f"{strategy}" for strategy, line_search in runs]
    
    plot_average(histories, labels)
    plot_all_histories(histories, labels)
    


if __name__ == "__main__":
    device = "cpu"
    energy_model = load_model("energy_model", device=device)
    hessian_model = load_model("hessian1", device=device)
    newton_step_model = load_model("newton_step", device=device)
    inv_hessian_model = load_model("inv_hessian2", device=device)

    main()