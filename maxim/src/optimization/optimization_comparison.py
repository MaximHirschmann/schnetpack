"""
Gradient Descent with Gradient vs Newton's Step
"""

import sys
import os
from typing import List 
from datatypes import OptimizationEvaluationXAxis, OptimizationEvaluationAllPlotsFix, GradientDescentResult
from .OptimizationMetric import EnergyMetric, ClosestMinimumMetric, ForceNormMetric
from plotting  import plot_average, plot_all_runs, plot_atom
from Utils import load_data, get_model_path
from .strategies import *
from .gradient_descent import GDParams, BacktrackingLineSearchParams, gradient_descent
from .LBFGS import LBFGS
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
    
    # jonas_all_forces is the best model
    basic_energy_model = "energy_model"
    best_energy_model = "jonas_forces_500_scfpy_loose" # 500 training samples, scfpy: data generation method
    best_energy_model_hessian = "jonas_hessian_500_loose"
    base_model = load_model(best_energy_model, device=device)
    base_model_path = get_model_path(best_energy_model)
    
    energy_metric = EnergyMetric(base_model, "basic", data.test_dataset[0][spk.properties.Z])
    closest_minimum_metric = ClosestMinimumMetric(data.test_dataset[0][spk.properties.Z])
    metric_to_use = closest_minimum_metric
    
    hessian_kronecker = load_model("hessian_kronecker", device=device)
    hessian_duut = load_model("hessian_duut", device=device)
    hessian_pd_kronecker = load_model("hessian_pd_kronecker", device=device)
    hessian_pd_duut = load_model("hessian_pd_duut", device=device)
    inv_hessian_kronecker = load_model("inv_hessian_kronecker", device=device)
    inv_hessian_duut = load_model("inv_hessian_duut", device=device)
    newton_step_pd_l1 = load_model("newton_step_pd_l1", device=device)
    
    strategies = [
        ForcesStrategy(base_model, GDParams(max_step_size=5e-4), name="Forces", normalize=False),
        
        # HessianStrategy(base_model, hessian_kronecker, GDParams(max_step_size=5e-2), tau=30, hessian_key="hessian", normalize=False, name="Hessian Kronecker"),
        # HessianStrategy(base_model, hessian_pd_kronecker, GDParams(max_step_size=5e-2), tau=30, hessian_key="hessian_pd", normalize=False, name="Hessian PD Kronecker"),
        # HessianStrategy(base_model, hessian_duut, GDParams(max_step_size=5e-2), tau=30, hessian_key="hessian", normalize=False, name="Hessian DUUT"),
        # HessianStrategy(base_model, hessian_pd_duut, GDParams(max_step_size=5e-2), tau=30, hessian_key="hessian_pd", normalize=False, name="Hessian PD DUUT"),
    
        # # InvHessianStrategy(base_model, inv_hessian_kronecker, GDParams(max_step_size=5e-3), normalize=False, name="Inv Hessian Kronecker 5e-3"),    
        # InvHessianStrategy(base_model, inv_hessian_duut, GDParams(max_step_size=5e-3), normalize=False, name="Inv Hessian DUUT 5e-3"),
        # AutoDiffHessianStrategy(base_model, data.test_dataset[0], gd_params=GDParams(max_step_size=2e-1), tau=35, vectorize=True, name = "AD Hessian 30"),
        
        NewtonStepStrategy(base_model, newton_step_pd_l1, GDParams(max_step_size=1e-2), newton_step_key="newton_step_pd", name="Newton-Step Strategy"), 
        
        # DiagonalStrategy(base_model),
        StrategyBase(base_model, GDParams(), "LBFGS"),
    ]
    
    # hyperparams = [
    #     {"max_step_size": 5e-4},
    #     {"max_step_size": 2e-4},
    #     {"max_step_size": 1e-4},
    #     {"max_step_size": 1e-5},
    # ]
    # for hyperparam in hyperparams:
    #     max_step_size = hyperparam["max_step_size"]
    #     strategies.append(ForcesStrategy(base_model, GDParams(max_step_size=max_step_size), name=f"Forces {max_step_size}"))
    
    
    cutoff_radius = base_model.representation.cutoff_fn.cutoff.item()
    spk_calculator = spk.interfaces.SpkCalculator(
        model_file=base_model_path,
        neighbor_list=spk.transform.MatScipyNeighborList(cutoff=cutoff_radius),
        device="cuda",
        dtype=torch.float64,
        energy_unit="kcal/mol",
    )


    N = 10
    for idx, i in enumerate(random.sample(range(len(data.test_dataset)), N)):
        histories.append([])
        results.append([])
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], 
            positions=structure[spk.properties.R]
        )
        atoms.pbc = np.array([False, False, False])
        atoms.calc = spk_calculator
        
        energy_0, _ = StrategyBase(base_model).prepare_energy_and_forces(converter(atoms))
        print("Structure", idx)
        print(f"Initial energy: {energy_0.item()}")
        for strategy in strategies:
            if strategy.name == "LBFGS":
                atoms_copy = atoms.copy()
                atoms_copy.calc = spk_calculator
                dyn = LBFGS(atoms_copy, logfile=None)
                dyn.run(fmax=0.001)
                result = GradientDescentResult(dyn.score_history, dyn.position_history, dyn.time_history)
                t = dyn.time_history[-1]
            else:
                result, t = gradient_descent(atoms.copy(), 
                                             strategy, 
                                             lineSearchParams=BacktrackingLineSearchParams(active = False)
                                            )
                
            print(f"Result {strategy.name}: {result.score_history[-1]} in {len(result.score_history):.4f} steps in {t:.2f} seconds")
            
            histories[-1].append(result.score_history)
            results[-1].append(result)
        print()

    for result in results:
        for res in result:
            res.apply_metric(metric_to_use)
            res.apply_metric(energy_metric)
            
    force_norm_metric = ForceNormMetric("force_norm")
    direction_norm_metric = ForceNormMetric("direction_norm")
    labels = [strategy.name for strategy in strategies]
    # plot_average(results, labels, force_norm_metric, OptimizationEvaluationXAxis.Iteration, title = "Force Norm over iteration steps")
    # plot_average(results, labels, direction_norm_metric, OptimizationEvaluationXAxis.Iteration, title = "Direction Norm over iteration steps")
    plot_average(results, labels, energy_metric, OptimizationEvaluationXAxis.Iteration, title = "Energy over iteration steps")
    plot_average(results, labels, energy_metric, OptimizationEvaluationXAxis.Time, title = "Energy over time")
    plot_average(results, labels, closest_minimum_metric, OptimizationEvaluationXAxis.Iteration, title = "Distance to closest Minimum over iteration steps")
    
    plot_all_runs(results, labels, closest_minimum_metric, 
                  OptimizationEvaluationAllPlotsFix.Run, 
                  OptimizationEvaluationXAxis.Iteration, 
                  title = "Distance to closest Minimum over iteration steps")
    
    plot_all_runs(results, labels, energy_metric,
                  OptimizationEvaluationAllPlotsFix.Run,
                  OptimizationEvaluationXAxis.Iteration,
                  title = "Energy over iteration steps")
    
    # atom_numbers = [atom_n.item() for atom_n in data.test_dataset[0][spk.properties.Z]]
    # plot_atom(closest_minimum_metric.conformers[0].positions, atom_numbers, title = "true minimum 0")
    # plot_atom(closest_minimum_metric.conformers[1].positions, atom_numbers, title = "true minimum 1")
    # plot_atom(closest_minimum_metric.conformers[2].positions, atom_numbers, title = "true minimum 2")
    
    # starting_pos = results[0][0].position_history[0]
    # plot_atom(starting_pos, atom_numbers, title = "Starting structure")
    
    # for i, strategy in enumerate(strategies):
    #     plot_atom(results[0][i].position_history[-1], atom_numbers, title = f"Final structure {strategy.name}")
    
    # plt.show()
    
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