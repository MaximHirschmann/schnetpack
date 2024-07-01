from Utils import load_model, load_data, timed
import os
import sys

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator
from ase import Atoms

import torch


class WrapperModel(torch.nn.Module):
    def __init__(self, model, converter, atom_numbers):
        super(WrapperModel, self).__init__()
        self.model = model
        self.converter = converter
        self.atom_numbers = atom_numbers
        

    def forward(self, positions):
        atoms = Atoms(
            numbers=self.atom_numbers, 
            positions=positions
        )
        inputs = self.converter(atoms)
        return self.model(inputs)["energy"]

def main():
    device = "cpu"
    model = load_model("energy_model", device=device)

    data = load_data()

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    
    structure = data.test_dataset[0]
    # atoms = Atoms(
    #     numbers=structure[spk.properties.Z], 
    #     positions=structure[spk.properties.R]
    # )

    wrapper_model = WrapperModel(model, converter, structure[spk.properties.Z])

    hessian = torch.autograd.functional.hessian(wrapper_model, structure[spk.properties.R])

    print("END")

if __name__ == "__main__":
    main()