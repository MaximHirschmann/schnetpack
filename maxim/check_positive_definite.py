"""
Checks if our hessian matrices are positive definite
"""


import os
import sys

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

from ase.io import read
import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import time

np.set_printoptions(suppress=True)

def get_targets(raw_data_dir, idx):
    energies_dir = os.path.join(raw_data_dir, "energies_ethanol")
    forces_dir = os.path.join(raw_data_dir, "gradients_ethanol")
    hessians_dir = os.path.join(raw_data_dir, "hessians_ethanol")

    target_energies = np.load(os.path.join(energies_dir, f"rmd17_ethanol_{idx}_energy.npy"))
    gradients = np.load(os.path.join(forces_dir, f"rmd17_ethanol_{idx}_grad.npy"))
    target_forces = -gradients
    target_hessians = np.load(os.path.join(hessians_dir, f"rmd17_ethanol_{idx}_hess.npy"))

    # convert shape NxNx3x3 to 3Nx3N
    hessian_rows = []
    for row in target_hessians:
        hessian_rows.append(
            np.concatenate([_ for _ in row], axis=1)
        )
    target_hessians = np.concatenate(hessian_rows, axis=0)

    return target_energies, target_forces, target_hessians


def get_indices(raw_data_dir):
    energies_files = os.listdir(os.path.join(raw_data_dir, "energies_ethanol"))
    indices = [int(file.split("_")[2]) for file in energies_files]
    indices.sort()
    return indices


def create_databases():
    # get the indices of all files in energies_dirbi
    print("Getting indices...")
    indices = get_indices(raw_data_dir)

    print("Reading atoms...")
    ats = read(rmd17_data_path, index=":")
    ats = [ats[idx] for idx in indices]

    print("Checking hessians...")
    i = 0
    hessian_list = []
    norms = []
    forces_norms = []
    for sample_idx, at in zip(indices, ats):
        energies, forces, hessian = get_targets(raw_data_dir, sample_idx)


        eigvals, eigvectors = np.linalg.eigh(hessian)
        print(eigvals)
        
        hessian2 = levenberg_marquardt(hessian)
        # hessian3 = eigenvalue_modification(hessian)
        # hessian4 = modified_cholesky(hessian)
        
        # hessians = []
        # hessians.append(hessian)
        # hessians.append(hessian2)
        # hessians.append(hessian3)
        # hessians.append(hessian4)
        
        newton_step = np.linalg.solve(hessian2, -forces.flatten()).reshape(forces.shape)
        
        # hessian_list.append(hessian2)
        
        norm = np.linalg.norm(newton_step)
        norms.append(norm)
        
        forces_norm = np.linalg.norm(forces)
        forces_norms.append(forces_norm)
        
        # plot_hessians_and_newtonSteps(hessians, forces, [
        #     "Hessian", 
        #     "Levenberg-Marquardt modification", 
        #     "Eigenvalue modification", 
        #     "Modified Cholesky"
        # ])
        
        i += 1
        
        if i == 10:
            i = 0
            # plot_100_inverse_hessians(hessian_list)
            hessian_list = []
            # break
            
    # plot norms in histogram in same plot
    plt.hist(norms, bins=50, alpha=0.5, label="Newton step norms")
    plt.hist(forces_norms, bins=50, alpha=0.5, label="Force norms")
    plt.legend()
    
    plt.show()
        

def levenberg_marquardt(hessian):
    eigvals = np.linalg.eigvalsh(hessian)
    min_eigval = np.min(eigvals)
    
    if min_eigval > 0:
        return hessian
    
    factor = -min_eigval + 0.1
    return hessian + factor * np.eye(hessian.shape[0])

def eigenvalue_modification(hessian):
    eigvals, eigvectors = np.linalg.eigh(hessian)
    
    min_eigval = np.min(eigvals)
    
    # modified_eigvals = eigvals + np.abs(min_eigval) + 1e-3
    modified_eigvals = np.maximum(eigvals, 1e-3)
    
    return eigvectors.T @ np.diag(modified_eigvals) @ eigvectors


def modified_cholesky(H, beta=1e-8):
    """
    Perform a modified Cholesky decomposition on matrix H to ensure positive definiteness.
    H: Input Hessian matrix (must be symmetric)
    beta: Small positive constant to ensure positive definiteness
    Returns: L such that H ≈ L @ L.T and L is lower triangular
    """
    n = H.shape[0]
    L = np.zeros_like(H)
    D = np.zeros(n)
    
    for j in range(n):
        dj = H[j, j] - np.sum(L[j, :j]**2 * D[:j])
        D[j] = max(abs(dj), beta)
        L[j, j] = 1.0

        for i in range(j + 1, n):
            L[i, j] = (H[i, j] - np.sum(L[i, :j] * L[j, :j] * D[:j])) / D[j]

    H_new = L @ np.diag(D) @ L.T
    return H_new



def plot_hessians_and_newtonSteps(hessians, forces, names):
    inv_hessians = [np.linalg.inv(hessian) for hessian in hessians]
    
    forces_flat = forces.flatten()
    newton_steps = [np.linalg.solve(hessian, -forces_flat).reshape(forces.shape) for hessian in hessians]
    
    fig, axs = plt.subplots(3, len(hessians), figsize=(15, 10))
    
    for i in range(len(hessians)):
        hessian, inv_hessian, newton_step = hessians[i], inv_hessians[i], newton_steps[i]
        
        axs[0, i].imshow(hessian)
        axs[0, i].set_title(names[i])
        plt.colorbar(axs[0, i].imshow(hessian), ax=axs[0, i])
        
        axs[1, i].imshow(inv_hessian)
        axs[1, i].set_title("Inverse " + names[i])
        plt.colorbar(axs[1, i].imshow(inv_hessian), ax=axs[1, i])
        
        axs[2, i].imshow(newton_step)
        axs[2, i].set_title("Newton step " + names[i])
        plt.colorbar(axs[2, i].imshow(newton_step), ax=axs[2, i])
    
    plt.tight_layout()
    plt.show()
    

def plot_100_inverse_hessians(hessians):
    inv_hessians = [np.linalg.inv(hessian) for hessian in hessians]
    
    fig, axs = plt.subplots(10, 10, figsize=(15, 15))
    
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            inv_hessian = inv_hessians[idx]
            
            axs[i, j].imshow(inv_hessian)
            #axs[i, j].set_title(f"Sample {idx}")
            plt.colorbar(axs[i, j].imshow(inv_hessian), ax=axs[i, j])
    
    plt.tight_layout()
    plt.show()


data_directory = os.getcwd() + "\\maxim\\data\\"
raw_data_dir = data_directory + "ene_grad_hess_1000eth"
rmd17_data_path = data_directory + "rMD17\\rMD17.db"
database_file = data_directory + "custom_database.db"
model_file = os.getcwd() + "\\maxim\\best_inference_model"

model = torch.load(model_file)

create_databases()

