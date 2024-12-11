
import sys
import os 
from time import time
from plotting import plot, plot2
from Utils import load_model, load_data
import torch
import numpy as np

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator
from ase import Atoms
import torchmetrics
import pytorch_lightning as pl

def logger(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        print(f"----- running {func.__name__} -----")
        result = func(*args, **kwargs)
        print(f"----- {func.__name__} ran in {time()-t0:.2f} seconds -----")
        return result
    return wrapper

def get_training_directory():
    return os.path.join(os.getcwd(), "training\\logs")


@logger
def train(data, continue_last_training = False):
    epochs = 30
    properties_to_train_for = [
        "diagonal"
    ]
    
    cutoff = 5.
    n_atom_basis = 128
    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    paiNN = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis, 
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    
    module_for_property = {
        "energy": spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy"),
        "forces": spk.atomistic.Forces(energy_key="energy", force_key="forces"),
        "polarizability": spk.atomistic.Polarizability(n_in = n_atom_basis, polarizability_key = "polarizability"),
        "newton_step": spk.atomistic.NewtonStep(n_in = n_atom_basis, newton_step_key = "newton_step"),
        "newton_step_pd": spk.atomistic.NewtonStep(n_in = n_atom_basis, newton_step_key = "newton_step_pd"),
        "hessian": spk.atomistic.HessianDUUT(n_in = n_atom_basis, hessian_key = "hessian"),
        "hessian_pd": spk.atomistic.HessianDUUT(n_in = n_atom_basis, hessian_key = "hessian_pd"),
        "inv_hessian": spk.atomistic.HessianDUUT(n_in = n_atom_basis, hessian_key = "inv_hessian"),
        "diagonal": spk.atomistic.HessianDiagonalL1(n_in = n_atom_basis, diagonal_key = "diagonal"),
    }
    
    if continue_last_training:
        model = load_model()
    else:
        model = spk.model.NeuralNetworkPotential(
            representation=paiNN,
            input_modules=[pairwise_distance],
            output_modules=[module_for_property[property] for property in properties_to_train_for],
            postprocessors=[
                trn.CastTo64(),
            ]
        )

    def get_model_output(name):
        return spk.task.ModelOutput(
            name=name,
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError()
            }
        )

    task = spk.task.AtomisticTask(
        model=model,
        outputs=[get_model_output(property) for property in properties_to_train_for],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 2e-3}
    )

    filepath_model = os.path.join(get_training_directory(), "best_inference_model")

    logger = pl.loggers.TensorBoardLogger(save_dir=get_training_directory())
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=filepath_model,
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=get_training_directory(),
        max_epochs=epochs,
    )

    trainer.fit(task, datamodule=data)


@logger
def evaluate_model(model, data, 
        properties,
        showDiff=True,
        title = ""
        ):
    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # random pick
    for i in range(10):
        structure = data.test_dataset[np.random.randint(len(data.test_dataset))]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
        )

        inputs = converter(atoms)
        results = model(inputs)
        
        plot2(structure, results, properties, title = title, showDiff = showDiff)

@logger
def plot_kronecker_products(model, data):
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    for i in range(5):
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


@logger
def calculate_average_hessian(data):
    hessian = torch.zeros(27, 27)
    
    for i in range(len(data.dataset)):
        hessian += data.dataset[i]["hessian"]
    
    hessian = hessian/len(data.dataset)
    
    # plot_hessian(hessian)
    

# @logger
# def calculate_average_newton_step(data):
#     hessian = torch.zeros(9, 3)
    
#     for i in range(len(data.dataset)):
#         datapoint = data.dataset[i]
#         hessian += datapoint["newton_step"]
#         plot_structure(datapoint)
    
#     hessian = hessian/len(data.dataset)
    
#     # plot_hessian(hessian)
    
@logger
def compare_directions(model, data, compare = ["forces", "best_direction", "newton_step"]):
    """
    Compares the directions of the best direction and the forces
    by which minimizes the energy the most.
    """
    
    
    def get_energy(atom, force):
        atom.positions = atom.positions + force
        inputs = converter(atom)
        energy = model(inputs)["energy"].detach().cpu().numpy()[0]
        atom.positions = atom.positions - force
        return energy
    
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    
    scales = [0.001, 0.01, 0.5, 1, 2]
    header = ["scale"] + [f"{key}_model" for key in compare] + [f"{key}_real" for key in compare]
    
    avg_table = np.zeros((len(scales), len(header)))
    for i in range(len(data.test_dataset)):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
        )

        inputs = converter(atoms)
        model_results = model(inputs)
        
        energy = model_results["energy"].detach().cpu().numpy()[0]
        model_values = [model_results[key].detach().cpu().numpy() for key in compare]
        real_values = [structure[key].detach().cpu().numpy() for key in compare]
        
        # invert newton step
        newton_idx = compare.index("newton_step")
        if newton_idx != -1:
            model_values[newton_idx] = -model_values[newton_idx]
            real_values[newton_idx] = -real_values[newton_idx]
        
        table = []
        for scale in scales:
            table.append([scale] 
                         + [get_energy(atoms, scale * value) for value in model_values]
                         + [get_energy(atoms, scale * value) for value in real_values]
                         )
            
        # do simple line search
        # find the best scale for each direction
        # eps = 1e-3
        # best_row = [-1.0]
        # for j in range(len(compare)):
        #     for value in [model_values[j]] + [real_values[j]]:
        #         best_energy = 1e10
        #         for n in range(1, 1000):
        #             energy = get_energy(atoms, n * eps * value)
        #             if energy < best_energy:
        #                 best_energy = energy
        #             else:
        #                 break
                
        #         best_row.append(best_energy)
        # table.append(best_row)
        
        avg_table += np.array(table)
        
        if i < 5:
            print(f"CALCULATING FOR SAMPLE {i}")
            print(f"ENERGY: {energy}")
            print(tabulate(table, headers=header))
            
    avg_table /= len(data.test_dataset)
    print()
    print("AVERAGE TABLE")
    print(tabulate(avg_table, headers=header))
    
    
def main():
    data = load_data()
    
    # calculate_average_newton_step(data)
    
    # train(data)
    
    model = load_model("uut_model")

    # loss = evaluate_model(model, data, 
    #         properties = ["original_hessian"],
    #         title="Kronecker Model - Comparison of predicted and real hessian",
    #         showDiff=True,
    #     )
    
    plot_kronecker_products(model, data)

    # compare_directions(model, data, compare = ["forces", "newton_step"])
    
    # calculate_average_hessian(data)
    
    # calculate_average_newton_step(data)
    
    print("END")
    

if __name__ == "__main__":
    schnetpack_dir = os.getcwd()
    sys.path.insert(1, schnetpack_dir + "\\src")

    import schnetpack as spk
    from schnetpack.data import ASEAtomsData
    import schnetpack.transform as trn

    import torch
    import torchmetrics
    import pytorch_lightning as pl
    from ase import Atoms
    from tabulate import tabulate
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    main()
    