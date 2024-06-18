import sys
import os 

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")


from ase.io import read, write
from ase import Atoms
import numpy as np
import torch
from schnetpack.transform.neighborlist import MatScipyNeighborList
from schnetpack.interfaces import SpkCalculator
from ase.optimize import LBFGS
from ase.visualize import view

from plotting import plot_atoms


data_dir = schnetpack_dir + "\\tests\\testdata\\"
atom_file = data_dir + "md_ethanol.xyz"
model_file = data_dir + "md_ethanol.model"
output_file = data_dir + "md_ethanol_output.xyz"


at = read(atom_file)
pos = at.get_positions()
at_nums = at.get_atomic_numbers()

at_new = Atoms(positions=pos, numbers=at_nums)
at_new.pbc = at.pbc
at_new.cell = at.cell

calculator = SpkCalculator(
    model_file=model_file,
    neighbor_list=MatScipyNeighborList(cutoff=5.0),
    energy_unit="kcal/mol",
    device=torch.device("cpu"),
)
at_new.calc = calculator
# run optimization
dyn = LBFGS(at_new)
dyn.run(fmax=1e-4)

write(output_file, [at, at_new], format="extxyz")

plot_atoms([at, at_new])

view(at_new)

# root mean squared deviaton - difference between geometries