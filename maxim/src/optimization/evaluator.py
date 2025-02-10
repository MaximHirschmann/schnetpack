import sys
import os
from typing import List 
from ase import Atoms
import torch
import numpy as np
from plotting import plot_hessian2
from time import time

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
import schnetpack.transform as trn

class AutoDiffWrapper(torch.nn.Module):
    def __init__(self, model, inputs_template):
        super(AutoDiffWrapper, self).__init__()
        self.model = model
        self.inputs_template = inputs_template

    def forward(self, positions):
        inputs = self.inputs_template.copy()
        inputs[spk.properties.R] = positions
        return self.model(inputs)["energy"]
    
class Evaluator:
    def __init__(self, forces_model, tau = 50):
        self.forces_model = forces_model
        self.tau = tau
        
    def evaluate(self, model: torch.nn.Module, test_data: List, property: str) -> dict:
        total_mse = 0
        total_mape = 0
        total_cos_sim = 0
        ns_total_mse = 0
        ns_total_mape = 0
        ns_total_cos_sim = 0
        total_time = 0
        
        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cuda"
        )
        
        if property == "autodiff":
            structure = test_data[0]
            
            atoms = Atoms(
                numbers=structure[spk.properties.Z], 
                positions=structure[spk.properties.R]
            )
            
            autodiff_model = AutoDiffWrapper(model, inputs_template=converter(atoms))
            
        for structure in test_data:
            atoms = Atoms(
                numbers=structure[spk.properties.Z], 
                positions=structure[spk.properties.R]
            )
            
            atoms.pbc = np.array([False, False, False])
            
            inputs = converter(atoms)
            
            t0 = time()
            if property == "autodiff":
                hessian = torch.autograd.functional.hessian(autodiff_model, inputs[spk.properties.R], vectorize=True)
                hessian = torch.reshape(hessian, (27, 27)).to(dtype=torch.float64)
                # 200 is the usual norm for the hessian in our databases
                hessian *= 211 / torch.linalg.norm(hessian)
                
                forces = self.forces_model(inputs.copy())["forces"]
                newton_step = -torch.linalg.solve(hessian + self.tau * torch.eye(27, device = "cuda"), forces.flatten()).reshape(forces.shape)
                
                a, b = hessian, structure["hessian"]
            elif property == "forces":
                forces = self.forces_model(inputs.copy())["forces"]
                newton_step = -forces

                a, b = forces, structure["forces"]
            else:
                output = model(inputs.copy())
                
                newton_step = self.calculate_newton_step(inputs, output, property)
                
                a, b = output[property], structure[property]
            t1 = time()
            
            a, b = clean(a), clean(b)
            newton_step, newton_step_target = clean(newton_step), clean(structure["newton_step_pd"])
            
            total_cos_sim += cosine_similarity(a, b)
            ns_total_cos_sim += cosine_similarity(newton_step, newton_step_target)
            # total_mse += mse(a, b)
            # ns_total_mse += mse(newton_step, newton_step_target)
            # total_mape += mape(a, b)
            # ns_total_mape += mape(newton_step, newton_step_target)
        
            total_time += t1 - t0
            
        avg_mse = total_mse / len(test_data)
        avg_cos_sim = total_cos_sim / len(test_data)
        avg_mape = total_mape / len(test_data)
        ns_avg_mse = ns_total_mse / len(test_data)
        ns_avg_cos_sim = ns_total_cos_sim / len(test_data)
        ns_avg_mape = ns_total_mape / len(test_data)
        avg_time = total_time / len(test_data)
        
        return {
            #"mse": avg_mse,
            "cos_sim": avg_cos_sim,
            #"mape": avg_mape,
            #"newton_step_mse": ns_avg_mse,
            "newton_step_cos_sim": ns_avg_cos_sim,
            #"newton_step_mape": ns_avg_mape,
            "time": avg_time
        }
    
    def calculate_newton_step(self, input, output, property):
        if property == "energy" or property == "forces":
            return 0
        elif property == "newton_step" or property == "newton_step_pd":
            return output[property]
        
        forces = self.forces_model(input.copy())["forces"]
        if property == "hessian" or property == "hessian_pd":
            hessian = output[property]
            hessian += self.tau * torch.eye(hessian.shape[0], device=hessian.device)
            return -torch.linalg.solve(hessian, forces.flatten()).reshape(forces.shape)
        elif property == "inv_hessian":
            inv_hessian = output[property]
            return -(inv_hessian @ forces.flatten()).reshape(forces.shape)
        elif property == "diagonal":
            diagonal = output[property]
            return -(forces.flatten() / diagonal.flatten()).reshape(forces.shape)
        else:
            raise Exception("Invalid property")
        
        
def clean(a):
    return a.cpu().detach().flatten().to(dtype = torch.float32)

def mse(a, b):
    return ((a - b) ** 2).mean().item()

def cosine_similarity(a, b):
    dot_product = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    return (dot_product / (norm_a * norm_b)).item()

# mean absolute percentage error
def mape(a, b):
    return (torch.abs((a - b) / a)).mean().item()