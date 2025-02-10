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
import itertools

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
    
    energy_metric = EnergyMetric(base_model, "energy_metric", data.test_dataset[0][spk.properties.Z])
    closest_minimum_metric = ClosestMinimumMetric(data.test_dataset[0][spk.properties.Z])
    force_norm_metric = ForceNormMetric("force_norm")
    direction_norm_metric = ForceNormMetric("direction_norm")
    
    model_dir = os.path.join(os.getenv("LOCALAPPDATA"), "schnetpack", "models")

    hessian_kronecker = load_model(os.path.join(model_dir, "hessian_kronecker_20") , device=device)
    hessian_duut = load_model(os.path.join(model_dir, "hessian_duut"), device=device)
    inv_hessian_kronecker = load_model(os.path.join(model_dir, "inv_hessian_kronecker"), device=device)
    inv_hessian_duut = load_model(os.path.join(model_dir, "inv_hessian_duut"), device=device)
    diagonal_l0 = load_model(os.path.join(model_dir, "diagonal_l0"), device=device)
    diagonal_l1 = load_model(os.path.join(model_dir, "diagonal_l1"), device=device)
    newton_step_pd_l1 = load_model(os.path.join(model_dir, "newton_step"), device=device)
    
    strategies = [
        ForcesStrategy(base_model, GDParams(step_size=0.001, momentum=0.5), name="Forces", normalize=False),
        
        DiagonalStrategy(base_model, diagonal_l0, GDParams(step_size=0.01, momentum=0.9), diagonal_key="diagonal", name="Diagonal L0"),
        DiagonalStrategy(base_model, diagonal_l1, GDParams(step_size=0.01, momentum=0.9), diagonal_key="diagonal", name="Diagonal L1"),
        
        HessianStrategy(base_model, hessian_kronecker, GDParams(step_size=0.02, momentum=0.9), tau=35, hessian_key="hessian", normalize=False, name="Hessian Kronecker"),
        HessianStrategy(base_model, hessian_duut, GDParams(step_size=0.02, momentum=0.9), tau=35, hessian_key="hessian", normalize=False, name="Hessian $D+UU^T$"),
        
        # InvHessianStrategy(base_model, inv_hessian_kronecker, GDParams(step_size=5e-3), normalize=False, name="Inv Hessian Kronecker 5e-3"),    
        InvHessianStrategy(base_model, inv_hessian_duut, GDParams(step_size=0.005, momentum = 0.5), normalize=False, name="Inv Hessian $D+UU^T$"),
        
        AutoDiffHessianStrategy(base_model, data.test_dataset[0], gd_params=GDParams(max_step_size=0.1, step_size=0.2, momentum=0.5, momentum_only_good_directions=True), tau=35, vectorize=True, name = "Autodiff Hessian"),
        
        NewtonStepStrategy(base_model, newton_step_pd_l1, GDParams(step_size=0.025, momentum=0.65, max_iterations=100), newton_step_key="newton_step_pd", name="Newton step Strategy"),
        
        StrategyBase(base_model, GDParams(), "L-BFGS"),
    ]
    
    momentum_fix_strategies = [
        # AutoDiffHessianStrategy(base_model, data.test_dataset[0], gd_params=GDParams(max_step_size=0.1, step_size=0.2, momentum=0.5, momentum_only_good_directions=True), tau=35, vectorize=True, name = "AD Hessian Momentum fix"),
    ]

    # momentum_and_step_size= list(itertools.product(
    #         [0, 0.7],
    #         [0.05],
    #     )
    # )
    # strategies = []
    # for momentum, step_size in momentum_and_step_size:
    #     strategies.append(AutoDiffHessianStrategy(
    #         base_model,
    #         data.test_dataset[0],
    #         GDParams(step_size=step_size, momentum=momentum), 
    #         name=f"AD {momentum} {step_size}",
    #         vectorize=True,
    #         tau=35,
    #         )
    #     )
    
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


    # hardest structures: [10, 13, 39, 76]
    N = 100
    for idx, i in enumerate(random.sample(range(len(data.test_dataset)), N)): # enumerate(random.sample(range(len(data.test_dataset)), N)):
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
            if "BFGS" in strategy.name:
                atoms_copy = atoms.copy()
                atoms_copy.calc = spk_calculator
                dyn = LBFGS(atoms_copy, logfile=None)
                dyn.run(fmax=0.001)
                result = GradientDescentResult(dyn.score_history, dyn.position_history, dyn.time_history)
                t = dyn.time_history[-1]
                last_score = result.score_history[-1]["LBFGS metric"]
            else:
                result, t = gradient_descent(atoms.copy(), 
                                             strategy, 
                                             lineSearchParams=BacktrackingLineSearchParams(active = False)
                                            )
                last_score = result.score_history[-1]["basic"]
            print(f"Result {strategy.name}: {last_score:.2f} in {len(result.score_history)} steps in {t:.2f} seconds")
            
            histories[-1].append(result.score_history)
            results[-1].append(result)
        print()

    for result in results:
        for res in result:
            res.apply_metric(closest_minimum_metric)
            res.apply_metric(energy_metric)

    closest_minimum_metric.calculate_energy(results[0][0].position_history[-1])
    labels = [strategy.name for strategy in strategies]
    # plot_average(results, labels, force_norm_metric, OptimizationEvaluationXAxis.Iteration, title = "Force Norm over iteration steps")
    # plot_average(results, labels, direction_norm_metric, OptimizationEvaluationXAxis.Iteration, title = "Direction Norm over iteration steps")
    plot_average(results, labels, energy_metric, OptimizationEvaluationXAxis.Iteration, title = "Convergence of Energy Minimizing Gradient Descent over Iteration")
    plot_average(results, labels, energy_metric, OptimizationEvaluationXAxis.Time, title = "Convergence of Energy Minimizing Gradient Descent over Time")
    plot_average(results, labels, closest_minimum_metric, OptimizationEvaluationXAxis.Iteration, title = "Convergence of Energy Minimizing Gradient Descent over Iteration")
    plot_average(results, labels, closest_minimum_metric, OptimizationEvaluationXAxis.Time, title = "Convergence of Energy Minimizing Gradient Descent over Time")
    
    # plot_all_runs(results, labels, closest_minimum_metric, 
    #               OptimizationEvaluationAllPlotsFix.Run, 
    #               OptimizationEvaluationXAxis.Iteration, 
    #               title = "Distance to closest Minimum over iteration steps")
    
    # plot_all_runs(results, labels, energy_metric,
    #               OptimizationEvaluationAllPlotsFix.Run,
    #               OptimizationEvaluationXAxis.Iteration,
    #               title = "Energy over iteration steps")
    
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