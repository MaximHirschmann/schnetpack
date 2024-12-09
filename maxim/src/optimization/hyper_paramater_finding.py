import sys
import os 

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
import schnetpack.transform as trn

sys.path.insert(1, schnetpack_dir + "\\maxim\\src")
from strategies import *
from gradient_descent import GDParams, gradient_descent
from Utils import load_data

import torch
from ase import Atoms
import random
from time import time
from tabulate import tabulate


class StrategyEvalutionResult:
    strategy: StrategyBase
    gradient_descent_params: GDParams
    avg_score: float
    avg_steps: float
    avg_time: float
    

def strategy_evaluation(data, strategy: StrategyBase, N: int = 100, 
                        gradient_descent_params: GDParams = GDParams()):
    if type(strategy) is AutoDiffHessianStrategy:
        strategy.tau = gradient_descent_params.autodiff_muh
        
    t0 = time()
    steps = 0
    avg_score = 0
    for i in random.sample(range(len(data.test_dataset)), N):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z],
            positions=structure[spk.properties.R],
        )
        
        converter = spk.interfaces.AtomsConverter(
            neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
        )
        
        history, t = gradient_descent(atoms, strategy, gradient_descent_params)
        steps += len(history)
        avg_score += history[-1]
    
    avg_time = (time() - t0) / N
    avg_score /= N
    steps /= N
    
    result = StrategyEvalutionResult()
    result.strategy = strategy
    result.gradient_descent_params = gradient_descent_params
    result.avg_score = avg_score
    result.avg_steps = steps
    result.avg_time = avg_time
    
    print(f"Strategy {strategy.name} N: {N} finished with avg score {avg_score} in {steps} steps in {t} seconds")
    
    return result
    

def hyperparameter_search(data):
    params = [
        GDParams(autodiff_muh = 0),
        GDParams(autodiff_muh = 0.01),
        GDParams(autodiff_muh = 0.1),
        GDParams(autodiff_muh = 1),
        GDParams(autodiff_muh = 2),
    ]
    results = []
    for param in params:
        strategy = AutoDiffHessianStrategy(data.test_dataset[0])
        results.append(strategy_evaluation(data, strategy, N = 5, gradient_descent_params = param))
    
    table = [] 
    for result in results:
        table.append([result.strategy.name, result.gradient_descent_params, result.avg_score, result.avg_steps, result.avg_time])
        
    print(tabulate(table, headers=["Strategy", "Muh", "Avg Score", "Avg Steps", "Avg Time"]))


device = "cpu"

if __name__ == "__main__":
    hyperparameter_search(load_data())
