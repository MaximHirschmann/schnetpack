from Utils import load_model, load_data, timed
from plotting import plot_hessian
import os
import sys

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator
from ase import Atoms
import random

import torch
import numpy as np


class WrapperModel(torch.nn.Module):
    def __init__(self, model, inputs_template):
        super(WrapperModel, self).__init__()
        self.model = model
        self.inputs_template = inputs_template
        
    def forward(self, positions):
        inputs = self.inputs_template.copy()
        inputs[spk.properties.R] = positions
        return self.model(inputs)["energy"]

def main():
    device = "cpu"
    model = load_model("energy_model", device=device)

    data = load_data()

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    
    avg_hessian = torch.zeros((27, 27))
    n = 100
    for i in random.sample(range(len(data.test_dataset)), n):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z].detach().cpu(),
            positions=structure[spk.properties.R].detach().cpu(),
        )
        
        inputs_template = converter(atoms)

        wrapper_model = WrapperModel(model, inputs_template)

        hessian = torch.autograd.functional.hessian(wrapper_model, structure[spk.properties.R])

        avg_hessian += torch.reshape(hessian, (27, 27))
    
    avg_hessian /= n
    
    plot_hessian(avg_hessian)
    
    avg_hessian = avg_hessian.detach().numpy()
    
    np.save("avg_hessian.npy", avg_hessian)
    
    print("END")


if __name__ == "__main__":
    main()