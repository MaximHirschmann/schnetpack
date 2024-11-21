"""
Gradient Descent with Gradient vs Newton's Step
"""

import sys
import os
from typing import List 
from datatypes import OptimizationEvaluationXAxis, OptimizationEvaluationAllPlotsFix, GradientDescentResult
from .OptimizationMetric import EnergyMetric, ClosestMinimumMetric
from plotting  import plot_average, plot_all_runs, plot_atom
from Utils import load_data
from .strategies import *
from .gradient_descent import GradientDescentParameters, BacktrackingLineSearchParams, gradient_descent
import torch
from ase import Atoms
import matplotlib.pyplot as plt
import random

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator

    
def compare():
    device = "cpu"
    data = load_data()
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    histories = []
    results: List[List[GradientDescentResult]] = []
    
    basic_energy_model = load_model("energy_model", device=device)
    best_energy_model = load_model("jonas_forces_500_scfpy_loose", device=device)
    best_energy_model_hessian = load_model("jonas_hessian_500_loose", device=device)
    
    hessian_model = load_model("hessian1", device=device)
    hessian_model_kronecker = load_model("hessian_kronecker", device=device)
    inv_hessian_model = load_model("inv_hessian2", device=device)
    inv_hessian_model_kronecker = load_model("inv_hessian_kronecker", device=device)
    original_hessian_model_kronecker = load_model("original_hessian_kronecker", device=device)
    original_hessian_model_uut = load_model("uut_model", device=device)
    newton_step_model = load_model("newton_step", device=device)

    base_model = best_energy_model
    strategies = [
        ForcesStrategy(base_model),
        OriginalHessianStrategy(base_model, original_hessian_model_kronecker),
        OriginalHessianStrategy(base_model, original_hessian_model_uut, name = "D+UUT"),
        AutoDiffHessianStrategy(base_model, data.test_dataset[0], name = "AD Hessian"),
        AutoDiffHessianStrategy(base_model, data.test_dataset[0], create_graph=True, name = "AD Hessian Create Graph"),
        AutoDiffHessianStrategy(base_model, data.test_dataset[0], vectorize=True, name = "AD Hessian Vectorize"),
        AutoDiffHessianStrategy(base_model, data.test_dataset[0], create_graph=True, vectorize=True, name = "AD Hessian Create Graph Vectorize"),
        # DiagonalStrategy(base_model),
    ]
    
    line_search_params = BacktrackingLineSearchParams(active = True)
    # for tau in [0.1]: #[0.02, 0.05, 0.1, 0.2, 0.5, 1]:
    #     strategies.append(OriginalHessianStrategy(original_hessian_model_kronecker, name = f"Og hessian {tau}", tau = tau))
    
    N = 5
    for idx, i in enumerate(random.sample(range(len(data.test_dataset)), N)):
        histories.append([])
        results.append([])
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], 
            positions=structure[spk.properties.R]
        )

        energy_0, _ = StrategyBase(base_model).prepare_energy_and_forces(converter(atoms))
        print("Structure", idx)
        print(f"Initial energy: {energy_0.item()}")
        for strategy in strategies:
            result, t = gradient_descent(atoms.copy(), strategy)
            print(f"Result {strategy.name}: {result.score_history[-1]} in {len(result.score_history):.4f} steps in {t} seconds")
            
            histories[-1].append(result.score_history)
            results[-1].append(result)
        print()

    energy_metric = EnergyMetric(base_model, "basic", data.test_dataset[0][spk.properties.Z])
    closest_minimum_metric = ClosestMinimumMetric()
    
    metric_to_use = closest_minimum_metric
    
    for result in results:
        for res in result:
            res.apply_metric(metric_to_use)
            res.apply_metric(energy_metric)
            
    
    labels = [strategy.name for strategy in strategies]
    plot_average(results, labels, energy_metric, OptimizationEvaluationXAxis.Iteration, title = "Energy over iteration steps")
    plot_average(results, labels, closest_minimum_metric, OptimizationEvaluationXAxis.Iteration, title = "Energy over iteration steps")
    plot_average(results, labels, energy_metric, OptimizationEvaluationXAxis.Time, title = "Energy over time")
    # atom_numbers = [atom_n.item() for atom_n in data.test_dataset[0][spk.properties.Z]]
    # plot_atom(closest_minimum_metric.conformers[0].positions, atom_numbers, title = "One true minimum")
    
    # starting_pos = results[0][0].position_history[0]
    # plot_atom(starting_pos, atom_numbers, title = "Starting structure")
    
    # for i, strategy in enumerate(strategies):
    #     plot_atom(results[0][i].position_history[-1], atom_numbers, title = f"Final structure {strategy.name}")
        
    # plot_average(results, labels, metric_to_use, OptimizationEvaluationXAxis.Time, title = "Energy over time")
    # plot_all_runs(results, labels, energy_metric, OptimizationEvaluationAllPlotsFix.Strategy, OptimizationEvaluationXAxis.Iteration, title = "Energy over iteration steps")
    # plot_all_runs(results, labels, energy_metric, OptimizationEvaluationAllPlotsFix.Run, OptimizationEvaluationXAxis.Iteration, title = "Energy over iteration steps")
    # plot_all_runs(results, labels, energy_metric, OptimizationEvaluationAllPlotsFix.Strategy, OptimizationEvaluationXAxis.Time, title = "Energy over time")
    # plot_all_runs(results, labels, energy_metric, OptimizationEvaluationAllPlotsFix.Run, OptimizationEvaluationXAxis.Time, title = "Energy over time")
    
    # plot_average(histories, labels)
    # plot_average_over_time(results, labels)
    # plot_all_histories(histories, labels)
    # plot_true_values(results, labels, atoms.copy(), energy_0)
    
    


if __name__ == "__main__":
    device = "cpu"
    compare()