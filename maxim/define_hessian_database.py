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
        os.path.join(raw_data_dir, "data.db"),
        distance_unit="Ang",
        property_unit_dict={"energy": "Hartree",
                            "forces": "Hartree/Bohr",
                            "hessian": "Hartree/Bohr/Bohr"},
    )
    new_dataset_no_hessian = ASEAtomsData.create(
        os.path.join(raw_data_dir, "data-no-hessian.db"),
        distance_unit="Ang",
        property_unit_dict={"energy": "Hartree",
                            "forces": "Hartree/Bohr",
                            #"hessian": "Hartree/Bohr/Bohr"
                            },
    )


    hessian_determinants = []
    for sample_idx, at in zip(indices, ats):
        print(sample_idx)
        energies, forces, hessians = get_targets(raw_data_dir, sample_idx)

        properties = {
            "energy": energies[None],
            "forces": forces,
        }

        new_dataset_no_hessian.add_systems([properties], [at])
        
        properties["hessian"] = hessians
        
        new_dataset.add_systems([properties], [at])
        
    

data_directory = os.getcwd() + "\\maxim\\data\\"
raw_data_dir = data_directory + "ene_grad_hess_1000eth"
rmd17_data_path = data_directory + "rMD17\\rMD17.db"

indices = get_indices(raw_data_dir)

hessians = []
inv_hessians = []

n = 5
for i in range(n):
    print(i)
    idx = indices[i]
    energies, forces, hessian = get_targets(raw_data_dir, idx)
    inv_hessian = np.linalg.inv(hessian)
    
    hessians.append(hessian)
    inv_hessians.append(inv_hessian)
    
    
# plot and compare hessians
fig, axs = plt.subplots(n, 2, figsize=(2, 20))

axs[0, 0].set_title("Hessian")
axs[0, 1].set_title("Inverse Hessian")

for i in range(n):
    axs[i, 0].imshow(hessians[i])
    axs[i, 1].imshow(inv_hessians[i])

    axs[i, 0].get_xaxis().set_ticks([])
    axs[i, 0].get_yaxis().set_ticks([])
    axs[i, 1].get_xaxis().set_ticks([])
    axs[i, 1].get_yaxis().set_ticks([])

plt.show()
