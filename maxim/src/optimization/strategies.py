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
from .GradientDescentParameters import GDParams

from typing import Callable, Dict, Tuple
import torch
import numpy as np
from ase import Atoms

class StrategyBase:
    def __init__(self, 
                 base_model, 
                 gd_params: GDParams = GDParams(), 
                 name: str = ""
                 ) -> None:
        self.base_model = base_model
        self.name = name
        self.gradient_descent_params = gd_params
        
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
    def __init__(self, 
                 base_model, 
                 gd_params = GDParams(),
                 normalize: bool = False, 
                 name: str = "Forces"
                 ) -> None:
        super().__init__(base_model, gd_params, name)
        self.normalize = normalize
        
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        direction: torch.Tensor = forces
        if self.normalize:
            direction = direction / torch.linalg.norm(forces)
            
        return energy, forces, direction
    
    
class HessianStrategy(StrategyBase):
    def __init__(self, 
                 base_model, 
                 hessian_model, 
                 gd_params = GDParams(), 
                 tau: float = 50, 
                 hessian_key: str = "hessian",
                 normalize: bool = False,
                 name = "Hessian Strategy"
                 ) -> None:
        super().__init__(base_model, gd_params, name)
        self.model = hessian_model
        self.tau = tau
        self.hessian_key = hessian_key
        self.normalize = normalize
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        model_output = self.model(inputs_copy)
        hessian: torch.Tensor = model_output[self.hessian_key]
        hessian += self.tau * torch.eye(hessian.shape[0], device="cpu")
        direction: torch.Tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        if self.normalize:
            direction = direction / torch.linalg.norm(direction)
            
        return energy, forces, direction
    
class InvHessianStrategy(StrategyBase):
    def __init__(self, 
                 base_model, 
                 inv_hessian_model, 
                 gd_params = GDParams(), 
                 inv_hessian_key: str = "inv_hessian",
                 normalize: bool = False,
                 name = "Inv Hessian Strategy"
                 ) -> None:
        super().__init__(base_model, gd_params, name)
        self.model = inv_hessian_model
        self.inv_hessian_key = inv_hessian_key
        self.normalize = normalize
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        model_output = self.model(inputs_copy)
        inv_hessian: torch.Tensor = model_output[self.inv_hessian_key]
        direction: torch.Tensor = (inv_hessian @ forces.flatten()).reshape(forces.shape)
        if self.normalize:
            direction = direction / torch.linalg.norm(direction)
            
        return energy, forces, direction
    
class NewtonStepStrategy(StrategyBase):
    def __init__(self, 
                 base_model, 
                 newton_step_model, 
                 gd_params = GDParams(), 
                 newton_step_key: str = "newton_step_pd",
                 normalize: bool = False,
                 name = "Newton-Step Strategy"
                 ) -> None:
        super().__init__(base_model, gd_params, name)
        self.model = newton_step_model
        self.newton_step_key = newton_step_key
        self.normalize = normalize
    
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # energy, forces = super().prepare_energy_and_forces(inputs)
        # we dont need them
        energy = torch.zeros((1))
        forces = torch.zeros(inputs[spk.properties.R].shape)
        
        inputs_copy = inputs.copy()
        model_output = self.model(inputs_copy)
        direction: torch.Tensor = -1 * model_output[self.newton_step_key]
        if self.normalize:
            direction = direction / torch.linalg.norm(direction)
            
        return energy, forces, direction
    
class DiagonalStrategy(StrategyBase):
    def __init__(self, 
                 base_model, 
                 diagonal, 
                 gd_params = GDParams(), 
                 diagonal_key: str = "diagonal",
                 normalize: bool = False,
                 name = "Diagonal Strategy"
                 ) -> None:
        super().__init__(base_model, gd_params, name)
        self.model = diagonal
        self.diagonal_key = diagonal_key
        self.normalize = normalize
        
    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)
        
        inputs_copy = inputs.copy()
        model_output = self.model(inputs_copy)
        diagonal: torch.Tensor = model_output[self.diagonal_key]
        inv_diagonal: torch.Tensor = 1 / diagonal
        direction: torch.Tensor = (inv_diagonal * forces.flatten()).reshape(forces.shape)
        if self.normalize:
            direction = direction / torch.linalg.norm(direction)
            
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
                 gd_params = GDParams(),
                 create_graph: bool = False,
                 vectorize: bool = False,
                 tau: float = 1,
                 normalize: bool = False,
                 name: str = "AutoDiff Hessian") -> None:
        super().__init__(base_model, gd_params, name)

        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cpu"
        )

        example_atoms = Atoms(
            numbers=example_structure[spk.properties.Z],
            positions=example_structure[spk.properties.R],
        )
        inputs_template = converter(example_atoms)

        self.tau = tau
        self.wrapper = AutoDiffWrapper(base_model, inputs_template)
        self.create_graph = create_graph
        self.vectorize = vectorize
        self.normalize = normalize

    def get_direction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces = super().prepare_energy_and_forces(inputs)

        inputs_copy = inputs.copy()
        hessian = torch.autograd.functional.hessian(self.wrapper, inputs_copy[spk.properties.R], create_graph=self.create_graph, vectorize=self.vectorize)
        hessian = torch.reshape(hessian, (27, 27))
        hessian = hessian + self.tau * torch.eye(27, device = "cpu")
        hessian = hessian.to(dtype=torch.float64)

        direction: torch.tensor = torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        if self.normalize:
            direction = direction / torch.linalg.norm(direction)

        return energy, forces, direction
    