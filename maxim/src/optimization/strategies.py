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
import plotting

from typing import Dict, Tuple
import torch
import numpy as np
from ase import Atoms


class StrategyBase:
    def __init__(self, base_model, name: str = "", line_search: bool = True) -> None:
        self.base_model = base_model
        self.name = name
        self.line_search = line_search
        
    def prepare_energy_and_forces(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_copy = inputs.copy()
        inputs_copy = self.base_model(inputs_copy)
        
        return (
            inputs_copy["energy"], 
            inputs_copy["forces"]
            )
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    
class ForcesStrategy(StrategyBase):
    def __init__(self, base_model, line_search: bool = True) -> None:
        super().__init__(base_model, "Forces", line_search)
        
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        norm: torch.Tensor = torch.linalg.norm(forces)
        direction: torch.Tensor = forces / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
    
class HessianStrategy(StrategyBase):
    def __init__(self, base_model, hessian_model, name = "hessian", line_search: bool = True, make_pd = False, model2 = None) -> None:
        super().__init__(base_model, name, line_search)
        self.model = hessian_model
        self.make_pd = make_pd
        self.model2 = model2
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        model_output = self.model(inputs_copy)
        hessian: torch.Tensor = model_output["hessian"]
        if self.make_pd:
            smallest_eigval = torch.min(torch.linalg.eigvals(hessian).real)
            if smallest_eigval < 0:
                hessian = hessian + torch.eye(hessian.shape[0], device="cpu") * (-smallest_eigval + 1e-7)
            
        hessian = hessian / torch.linalg.norm(hessian)
        
        newton_step: torch.Tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
    
class OriginalHessianStrategy(StrategyBase):
    def __init__(self, base_model, hessian_model, name = "original_hessian", line_search: bool = True, tau = 0.1, modify_eig = False) -> None:
        super().__init__(base_model, name, line_search)
        self.model = hessian_model
        self.tau = tau
        self.modify_eig = modify_eig
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        model_output = self.model(inputs_copy)
        hessian: torch.Tensor = model_output["original_hessian"]
        if self.modify_eig:
            eigenvalues, eigenvectors = torch.linalg.eig(hessian)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            inv_eigenvalues = torch.where(eigenvalues > 1e-5, 1/eigenvalues, torch.tensor(1e-5, device = eigenvalues.device))
            inv_hessian = eigenvectors @ torch.diag(inv_eigenvalues) @ eigenvectors.T
            newton_step: torch.Tensor = (inv_hessian @ forces.flatten()).reshape(forces.shape)
        else:
            hessian += torch.eye(hessian.shape[0], device="cpu") * self.tau
            newton_step: torch.Tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
            
        direction: torch.Tensor = newton_step / torch.linalg.norm(newton_step)
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
    
class NewtonStepStrategy(StrategyBase):
    def __init__(self, base_model, newton_step_model, line_search: bool = True) -> None:
        super().__init__(base_model, "Newton Step", line_search)
        
        self.newton_step_model = newton_step_model
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = self.newton_step_model(inputs_copy)
        newton_step: torch.Tensor = inputs_copy["newton_step"]
        
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = -1 * newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class InvHessianStrategy(StrategyBase):
    def __init__(self, base_model, inv_hessian_model, name = "inv_hessian", line_search: bool = True) -> None:
        super().__init__(base_model, name, line_search)
        self.inv_hessian_model = inv_hessian_model
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = self.inv_hessian_model(inputs_copy)
        inv_hessian: torch.Tensor = inputs_copy["inv_hessian"]
        inv_hessian = inv_hessian / torch.linalg.norm(inv_hessian)
        
        newton_step: torch.Tensor = (inv_hessian @ forces.flatten()).reshape(forces.shape)
        norm: torch.Tensor = torch.linalg.norm(newton_step)
        direction: torch.Tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction
    
class AvgHessianStrategy(StrategyBase):
    def __init__(self, base_model, line_search: bool = True) -> None:
        super().__init__(base_model, "Avg Hessian", line_search)
        self.avg_hessian = torch.tensor(np.load(os.getcwd() + "\\maxim\\data\\avg_hessian.npy"), dtype=torch.float64, device="cpu")
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        hessian = self.avg_hessian
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
                 base_model,
                 example_structure, 
                 create_graph: bool = False,
                 vectorize: bool = False,
                 line_search: bool = True,
                 muh: float = 1,
                 name: str = "AutoDiff Hessian") -> None:
        super().__init__(base_model, name, line_search)

        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cpu"
        )

        example_atoms = Atoms(
            numbers=example_structure[spk.properties.Z],
            positions=example_structure[spk.properties.R],
        )
        inputs_template = converter(example_atoms)

        self.muh = muh
        self.wrapper = AutoDiffWrapper(base_model, inputs_template)
        self.create_graph = create_graph
        self.vectorize = vectorize

    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)

        inputs_copy = inputs.copy()
        hessian = torch.autograd.functional.hessian(self.wrapper, inputs_copy[spk.properties.R], create_graph=self.create_graph, vectorize=self.vectorize)
        hessian = hessian / torch.linalg.norm(hessian)
        hessian = hessian.to(dtype=torch.float64)
        hessian = torch.reshape(hessian, (27, 27))
        hessian += self.muh * torch.eye(27, device = "cpu", dtype = torch.float64)

        newton_step: torch.tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        norm: torch.tensor = torch.linalg.norm(newton_step)
        direction: torch.tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)

        return energy, forces, direction
    
class DiagonalStrategy(StrategyBase):
    def __init__(self, base_model, line_search: bool = True) -> None:
        super().__init__(base_model, "Diagonal", line_search)
        self.diagonal_model = load_model("diagonal2", device="cpu")
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        inputs_copy = self.diagonal_model(inputs_copy)
        diagonal: torch.Tensor = inputs_copy["original_diagonal"]
        inv_diagonal: torch.Tensor = 1 / diagonal
        
        newton_step: torch.Tensor = (inv_diagonal * forces.flatten()).reshape(forces.shape)
        norm = torch.linalg.norm(newton_step)
        direction: torch.Tensor = newton_step / norm
        direction = direction.to(dtype=torch.float32)
        
        return energy, forces, direction