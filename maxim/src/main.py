from optimization.optimization_comparison import compare
from optimization.evaluator import Evaluator
from plotting import plot_hessian2
from Utils import load_data, load_model
from train import evaluate_model, plot_kronecker_products, train

from dataclasses import dataclass
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import tabulate
import sys
import torch

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator
from ase import Atoms


def plot_kronecker_products():
    data = load_data()
    
    model_dir = os.path.join(os.getenv("LOCALAPPDATA"), "schnetpack", "models")
    model = load_model(os.path.join(model_dir, "hessian_kronecker_20"), device="cuda")
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    for i in range(1):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
        )

        inputs = converter(atoms)

        # prepare representations in inputs
        inputs = model.initialize_derivatives(inputs)
        for m in model.input_modules:
            inputs = m(inputs)
        inputs = model.representation(inputs)

        model.output_modules[0].plot_kronecker_products(inputs)
        
        

def plot_random_hessian():
    data = load_data()
    
    # 1 x 3 subplots
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(18, 12)  # Wide plot, reduced height
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3)  # Remove extra whitespace

    for j in range(3):
        ax = axs[j]
        rand_idx = random.randint(0, len(data.dataset)) 
        hessian = data.dataset[rand_idx]["hessian"]
        
        plot_hessian2(hessian, ax=ax, text_fontsize=16, colorbar_fontsize=16, use_small_xyz_ticks=False)
    
    plt.savefig("random_hessians.svg")
    plt.show()
    
def train_fn(data, properties, epochs):
    train(data, 
          continue_last_training=False, 
          properties_to_train_for=properties, 
          epochs=epochs)
    
    # model = load_model()

    # loss = evaluate_model(model, data, 
    #         properties = ["diagonal"],
    #         title="Diagonal L1 prediction",
    #         showDiff=True,
    #     )
    
    # plot_kronecker_products(model, data)
    
    # print("END")
    
def check_dataset():
    data = load_data()
    
    step_norms = []
    pd_norms = []
    
    for i in data.train_dataset:
        newton_step_norm = 0
        newton_step_norm = np.linalg.norm(i["newton_step"])
        newton_step_pd_norm = np.linalg.norm(i["newton_step_pd"])
        
        step_norms.append(newton_step_norm)
        pd_norms.append(newton_step_pd_norm)
        
    step_norms.sort(reverse=True)
    pd_norms.sort(reverse=True)
    
    print(step_norms[:100])
    print(pd_norms[:100])
        
    print("END")
    
@dataclass(init = True, repr = True)
class Run:
    name: str = ""
    property: str = ""
    epochs: int = 3
    
    use_duut: bool = False
    use_kronecker: bool = False
    F_1: int = 6 # dimension of low rank approximation - only for hessian and inv_hessian
    F_2: int = 20 # number of kronecker products - only for hessian and inv_hessian
    diag_l0: bool = False # if True, diagonal is trained with l0 features else l1
    
    
    
def train_and_evaluate():
    data = load_data()

    epochs = 30
    runs = [
        # Run(name="hessian_duut", property="hessian", epochs=epochs, use_duut=True, F_1=6),
        # Run(name="hessian_kronecker", property="hessian", epochs=epochs, use_kronecker=True, F_2=20),
        # Run(name="inv_hessian_duut", property="inv_hessian", epochs=epochs, use_duut=True, F_1=6),
        # Run(name="inv_hessian_kronecker", property="inv_hessian", epochs=epochs, use_kronecker=True, F_2=20),
        # Run(name="diagonal_l0", property="diagonal", epochs=epochs, diag_l0=True),
        # Run(name="diagonal_l1", property="diagonal", epochs=epochs, diag_l0=False),
        # Run(name="newton_step", property="newton_step_pd", epochs=epochs),
        # Run(name="hessian_duut_1", property="hessian", epochs=epochs, use_duut=True, F_1=1),
        # Run(name="hessian_duut_2", property="hessian", epochs=epochs, use_duut=True, F_1=2),
        # Run(name="hessian_duut_3", property="hessian", epochs=epochs, use_duut=True, F_1=3),
        # Run(name="hessian_duut_4", property="hessian", epochs=epochs, use_duut=True, F_1=4),
        # Run(name="hessian_duut_5", property="hessian", epochs=epochs, use_duut=True, F_1=5),
        # Run(name="hessian_duut_6", property="hessian", epochs=epochs, use_duut=True, F_1=6),
        # Run(name="hessian_duut_7", property="hessian", epochs=epochs, use_duut=True, F_1=7),
        # Run(name="hessian_duut_8", property="hessian", epochs=epochs, use_duut=True, F_1=8),
        # Run(name="hessian_duut_9", property="hessian", epochs=epochs, use_duut=True, F_1=9),
        # Run(name="hessian_duut_10", property="hessian", epochs=epochs, use_duut=True, F_1=10),
        Run(name="hessian_kronecker_4", property="hessian", epochs=epochs, use_duut=False, use_kronecker=True, F_2=4),
        Run(name="hessian_kronecker_8", property="hessian", epochs=epochs, use_duut=False, use_kronecker=True, F_2=8),
        Run(name="hessian_kronecker_12", property="hessian", epochs=epochs, use_duut=False, use_kronecker=True, F_2=12),
        Run(name="hessian_kronecker_16", property="hessian", epochs=epochs, use_duut=False, use_kronecker=True, F_2=16),
        Run(name="hessian_kronecker_20", property="hessian", epochs=epochs, use_duut=False, use_kronecker=True, F_2=20),
        Run(name="hessian_kronecker_24", property="hessian", epochs=epochs, use_duut=False, use_kronecker=True, F_2=24),
        Run(name="hessian_kronecker_28", property="hessian", epochs=epochs, use_duut=False, use_kronecker=True, F_2=28),
        
    ]
    
    model_dir = os.path.join(os.getenv("LOCALAPPDATA"), "schnetpack", "models")
    
    base_model = load_model("jonas_forces_500_scfpy_loose")
    
    evaluator = Evaluator(base_model, tau = 35)
    eval_results = []
    for run in runs:
        train(data,
              continue_last_training=False,
              properties_to_train_for=[run.property],
              epochs=run.epochs,
              model_dir=model_dir,
              model_name=run.name,
              use_duut=run.use_duut,
              use_kronecker=run.use_kronecker,
              F_1 = run.F_1,
              F_2 = run.F_2,
              diag_l0 = run.diag_l0,
              )
        
        print(f"Evaluating {run.name})")
        
        evaluation = evaluator.evaluate(
            model=load_model(os.path.join(model_dir, run.name), device="cuda"),
            test_data=data.test_dataset,
            property=run.property,
        )
        print(evaluation)
        eval_results.append(evaluation)
        
        # loss = evaluate_model(
        #     load_model(os.path.join(model_dir, run.name)), data, 
        #     properties = [run.property],
        #     title=f"Model {run.name}",
        #     showDiff=True,
        # )
        
    
    for i in range(len(eval_results)):
        name = runs[i].name
        print(name, eval_results[i])
        
def evaluate_once_again():
    data = load_data()
    
    models_and_properties = [
        # ("jonas_all_forces", "forces"),
        # ("hessian_duut", "hessian"),
        # ("hessian_kronecker", "hessian"),
        # ("inv_hessian_duut", "inv_hessian"),
        # ("inv_hessian_kronecker", "inv_hessian"),
        # ("diagonal_l0", "diagonal"),
        # ("diagonal_l1", "diagonal"),
        # ("newton_step", "newton_step_pd"),
        # ("autodiff", "autodiff"),
        # ("hessian_duut_1", "hessian"),
        # ("hessian_duut_2", "hessian"),
        # ("hessian_duut_3", "hessian"),
        # ("hessian_duut_4", "hessian"),
        # ("hessian_duut_5", "hessian"),
        # ("hessian_duut_6", "hessian"),
        # ("hessian_duut_7", "hessian"),
        # ("hessian_duut_8", "hessian"),
        # ["hessian_duut_9", "hessian"],
        # ["hessian_duut_10", "hessian"],
        ["hessian_kronecker_4", "hessian"],
        ["hessian_kronecker_8", "hessian"],
        ["hessian_kronecker_12", "hessian"],
        ["hessian_kronecker_16", "hessian"],
        ["hessian_kronecker_20", "hessian"],
        ["hessian_kronecker_24", "hessian"],
        ["hessian_kronecker_28", "hessian"],
    ]
    
    # base_model = load_model("jonas_forces_500_scfpy_loose")
    base_model = load_model("jonas_all_forces")
    
    evaluator = Evaluator(base_model, tau = 35)
    model_dir = os.path.join(os.getenv("LOCALAPPDATA"), "schnetpack", "models")
    results = []
    headers = []
    for model_name, property in models_and_properties:
        if property == "autodiff" or property == "forces":
            model = base_model
        else:
            model = load_model(os.path.join(model_dir, model_name), device="cuda")
            
        eval_res = evaluator.evaluate(
            model=model,
            test_data=data.test_dataset,
            property=property,
        )
        
        headers = ["Model"] + list(eval_res.keys())
        values = [model_name] + [f"{v:.3f}".replace(".", ",") for v in eval_res.values()]
        results.append(values)
        print(model_name)
    table = tabulate.tabulate(results, headers=headers)
    print(table)
    
        
def find_best_tau():
    data = load_data()
    
    model = "hessian_kronecker"
    property = "autodiff"
    
    base_model = load_model("jonas_forces_500_scfpy_loose")
    model_dir = os.path.join(os.getenv("LOCALAPPDATA"), "schnetpack", "models")
    
    taus = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    results = []
    for tau in taus:
        evaluator = Evaluator(base_model, tau = tau)
        eval_res = evaluator.evaluate(
            model=load_model(os.path.join(model_dir, model), device="cuda"),
            test_data=data.test_dataset,
            property=property,
        )
        
        results.append((tau, eval_res["newton_step_mse"], eval_res["newton_step_cos_sim"]))
        print(tau)
        
    table = tabulate.tabulate(results, headers=["tau", "mse", "cos_sim"], tablefmt="github", floatfmt=".2f")
    print(table)

def compare_hessian_and_inverse():
    data = load_data()
    
    fontsize = 18
    N = 4  # Number of structures to compare
    size_per_subplot = 4  # Controls the size of each subplot
    fig, axs = plt.subplots(2, N, figsize=(size_per_subplot * N, size_per_subplot * 2), constrained_layout=True)

    for i in range(N):
        structure = data.dataset[i]
        
        hessian = structure["hessian"]
        inv_hessian = structure["inv_hessian"]
        
        # Hessian Plot
        im1 = axs[0, i].imshow(hessian, aspect="equal")  # Square aspect
        axs[0, i].set_title(f"Structure {i+1}: Hessian", fontsize=fontsize)
        cbar = fig.colorbar(im1, ax=axs[0, i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=fontsize)
        
        # Inverse Hessian Plot
        im2 = axs[1, i].imshow(inv_hessian, aspect="equal")  # Square aspect
        axs[1, i].set_title(f"Structure {i+1}: Inverse Hessian", fontsize=fontsize)
        cbar = fig.colorbar(im2, ax=axs[1, i], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=fontsize)
        
        # no ticks
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        
    plt.savefig("hessian_and_inverse.svg")
    plt.show()
    
def plot_conformers():
    from optimization.OptimizationMetric import ClosestMinimumMetric
    
    closest_minimum_metric = ClosestMinimumMetric([6, 6, 8, 1, 1, 1, 1, 1, 1])
    
    closest_minimum_metric.plot_conformers()
    

if __name__ == '__main__': 
    # plot_kronecker_products()
    compare()
    # plot_random_hessian()
    # train_fn()
    # check_dataset()
    # train_and_evaluate()
    # evaluate_once_again()
    # find_best_tau()
    # compare_hessian_and_inverse()
    # plot_conformers()
    
    print("END")