import sys
import os 

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator

sys.path.insert(1, schnetpack_dir + "\\maxim\\src")
from Utils import load_model

from typing import Dict, Tuple
import torch
import numpy as np
from ase import Atoms


class StrategyBase:
    def __init__(self, name: str = "", line_search: bool = True) -> None:
        self.name = name
        self.line_search = line_search
        
    def prepare_energy_and_forces(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_copy = inputs.copy()
        inputs_copy = energy_model(inputs_copy)
        
        return (
            inputs_copy["energy"], 
            inputs_copy["forces"]
            )
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    
class ForcesStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Forces", line_search)
        
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        norm: torch.Tensor = torch.linalg.norm(forces)
        direction: torch.Tensor = forces / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
    
class HessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = hessian_model(inputs_copy)
        hessian: torch.Tensor = inputs_copy["hessian"]
        hessian = hessian / torch.linalg.norm(hessian)
        
        newton_step: torch.Tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
    
class NewtonStepStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Newton Step", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = newton_step_model(inputs_copy)
        newton_step: torch.Tensor = inputs_copy["newton_step"]
        
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = -1 * newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class InvHessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Inv Hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = inv_hessian_model(inputs_copy)
        inv_hessian: torch.Tensor = inputs_copy["inv_hessian"]
        inv_hessian = inv_hessian / torch.linalg.norm(inv_hessian)
        
        newton_step: torch.Tensor = (inv_hessian @ forces.flatten()).reshape(forces.shape)
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class AvgHessianStrategy(StrategyBase):
    def __init__(self, line_search: bool = True) -> None:
        super().__init__("Avg Hessian", line_search)
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        hessian = avg_hessian
        hessian = hessian / torch.linalg.norm(hessian)
        newton_step: torch.tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.tensor = torch.linalg.norm(newton_step)
        direction: torch.tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
    
class AutoDiffWrapper(torch.nn.Module):
    def __init__(self, model, inputs_template):
        super(AutoDiffWrapper, self).__init__()
        self.model = model
        self.inputs_template = inputs_template

    def forward(self, positions):
        inputs = self.inputs_template.copy()
        inputs[spk.properties.R] = positions
        return self.model(inputs)["energy"]

class AutoDiffHessianStrategy(StrategyBase):
    def __init__(self, 
                 example_structure, 
                 line_search: bool = True,
                 muh: float = 1) -> None:
        super().__init__("AutoDiff Hessian", line_search)

        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
        )

        example_atoms = Atoms(
            numbers=example_structure[spk.properties.Z],
            positions=example_structure[spk.properties.R],
        )
        inputs_template = converter(example_atoms)

        self.muh = muh
        self.wrapper = AutoDiffWrapper(energy_model, inputs_template)

    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)

        inputs_copy = inputs.copy()
        hessian = torch.autograd.functional.hessian(self.wrapper, inputs_copy[spk.properties.R])
        hessian = hessian / torch.linalg.norm(hessian)
        hessian = hessian.to(dtype=torch.float64)
        hessian = torch.reshape(hessian, (27, 27))
        hessian += self.muh * torch.eye(27, device = device, dtype = torch.float64)

        newton_step: torch.tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.tensor = torch.linalg.norm(newton_step)
        direction: torch.tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)

        return energy, forces, direction
    
"""
--------------------------  GLOBAL VARIABLES -------------------------
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

energy_model = load_model("energy_model", device=device)
# energy_model = load_model("jonas_all_forces", device=device)
# energy_model = load_model("jonas_forces_500_scfpy_loose", device=device)
# energy_model = load_model("jonas_hessian_500_loose", device=device)
hessian_model = load_model("hessian1", device=device)
newton_step_model = load_model("newton_step", device=device)
inv_hessian_model = load_model("inv_hessian2", device=device)
avg_hessian = torch.tensor(np.load(os.getcwd() + "\\maxim\\data\\avg_hessian.npy"), dtype=torch.float64, device=device)
