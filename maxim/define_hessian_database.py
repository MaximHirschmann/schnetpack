import matplotlib.pyplot as plt
import numpy as np
import os
from ase.io import read
from schnetpack.data import ASEAtomsData


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


def create_databases():
    # get the indices of all files in energies_dirbi
    indices = get_indices(raw_data_dir)

    ats = read(rmd17_data_path, index=":")
    ats = [ats[idx] for idx in indices]

    # initialize new db dataset
    new_dataset = ASEAtomsData.create(
        database_file,
        distance_unit="Ang",
        property_unit_dict={"energy": "Hartree",
                            "forces": "Hartree/Bohr",
                            "hessian": "Hartree/Bohr/Bohr",
                            "newton_step": "Bohr"},
    )

    hessian_determinants = []
    for sample_idx, at in zip(indices, ats):
        print(sample_idx)
        energies, forces, hessians = get_targets(raw_data_dir, sample_idx)

        forces_flattened = forces.flatten()
        newton_step = -np.linalg.solve(hessians, forces_flattened).reshape(9, 3)

        properties = {
            "energy": energies[None],
            "forces": forces,
            "hessian": hessians,
            "newton_step": newton_step
        }

        new_dataset.add_systems([properties], [at])
        
    

data_directory = os.getcwd() + "\\maxim\\data\\"
raw_data_dir = data_directory + "ene_grad_hess_1000eth"
rmd17_data_path = data_directory + "rMD17\\rMD17.db"
database_file = data_directory + "custom_database.db"

create_databases()