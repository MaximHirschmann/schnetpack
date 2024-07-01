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

    # mask out the upper triangle of the hessian
    ##mask = np.ones(target_hessians.shape, dtype=bool)
    ##mask = np.triu(mask).flatten()

    ##target_hessians = target_hessians.flatten()
    ##target_hessians = target_hessians[mask]
    #target_hessians = target_hessians.reshape(9, -1)

    return target_energies, target_forces, target_hessians


def get_indices(raw_data_dir):
    energies_files = os.listdir(os.path.join(raw_data_dir, "energies_ethanol"))
    indices = [int(file.split("_")[2]) for file in energies_files]
    indices.sort()
    return indices


def levenberg_marquardt(hessian):
    eigvals = np.linalg.eigvalsh(hessian)
    min_eigval = np.min(eigvals)
    
    if min_eigval > 0:
        return hessian
    
    factor = -min_eigval + 0.1
    return hessian + factor * np.eye(hessian.shape[0])


def create_databases():
    # get the indices of all files in energies_dirbi
    indices = get_indices(raw_data_dir)

    ats = read(rmd17_data_path, index=":")
    ats = [ats[idx] for idx in indices]

    # initialize new db dataset
    new_dataset = ASEAtomsData.create(
        database_file,
        distance_unit="Ang",
        property_unit_dict={
            "energy": "Hartree",
            "forces": "Hartree/Bohr",
            "hessian": "Hartree/Bohr/Bohr",
            "inv_hessian": "Bohr/Hartree/Bohr",
            "newton_step": "Hartree/Bohr",
            },
    )

    for sample_idx, at in zip(indices, ats):
        energies, forces, hessians = get_targets(raw_data_dir, sample_idx)

        hessians = levenberg_marquardt(hessians)
        
        newton_step = np.linalg.solve(hessians, -forces.flatten()).reshape(forces.shape)
        
        inv_hessian = np.linalg.inv(hessians)
        
        properties = {
            "energy": energies[None],
            "forces": forces,
            "hessian": hessians,
            "newton_step": newton_step,
            "inv_hessian": inv_hessian
        }

        new_dataset.add_systems([properties], [at])
        
        print(sample_idx)
        
        
def get_best_direction(atom, force):
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cuda"
    )
    
    x = atom.positions.copy()
    
    atom.positions = x + force
    inputs = converter(atom)
    energy = model(inputs)["energy"]
    
    best_energy = energy
    best_delta = force
    
    # scalar - the norm of the force
    force_norm = np.linalg.norm(force)
    
    energy_history = [best_energy]
    #t0 = time()
    # get the best direction to minimize the energy
    for i in range(3):
        for j in range(2 * x.shape[0]):
            dim = j % x.shape[0]
            muh = (-1) ** (j // x.shape[0]) * 0.01
            while True:
                # TODO check dimensions
                new_delta = best_delta.copy()
                new_delta[dim] += muh
                # adjust norm to that of the force
                new_delta = new_delta * force_norm / np.linalg.norm(new_delta)
                
                new_x = x + new_delta
                
                atom.positions = new_x
                inputs = converter(atom)
                new_energy = model(inputs)["energy"]
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_delta = new_delta
                else:
                    break
        energy_history.append(best_energy)
        
    
    atom.positions = x
    
    return best_delta
    
    
    

        
        

data_directory = os.getcwd() + "\\maxim\\data\\"
raw_data_dir = data_directory + "ene_grad_hess_1000eth"
rmd17_data_path = data_directory + "rMD17\\rMD17.db"
database_file = data_directory + "custom_database.db"
model_file = os.getcwd() + "\\maxim\\best_inference_model"

model = torch.load(model_file)

create_databases()