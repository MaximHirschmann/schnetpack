
import sys
import os 
from time import time
from plotting import plot, plot_structure


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
def load_data():
    # filepath_hessian_db = os.path.join(os.getcwd(), 'maxim\\data\\ene_grad_hess_1000eth\\data.db')
    filepath_hessian_db = os.path.join(os.getcwd(), "maxim\\data\\custom_database.db")

    hessianData = spk.data.AtomsDataModule(
        filepath_hessian_db, 
        distance_unit="Ang",
        property_units={"energy": "Hartree",
                        "forces": "Hartree/Bohr",
                        "hessian": "Hartree/Bohr/Bohr",
                        "newton_step": "Bohr",
                        # "best_direction": "Hartree/Bohr"
                        },
        batch_size=10,
        
        transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ],
        
        num_train=800,
        num_val=100,
        num_test=100,
        
        pin_memory=True, # set to false, when not using a GPU
        
    )
    hessianData.prepare_data()
    hessianData.setup()
    
    return hessianData


@logger
def train(data):
    epochs = 40
    cutoff = 5.
    n_atom_basis = 30

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    paiNN = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis, 
        n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy")
    pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")
    pred_polarizability = spk.atomistic.Polarizability(n_in = n_atom_basis, polarizability_key = "polarizability")
    pred_hessian = spk.atomistic.Hessian(n_in = n_atom_basis, hessian_key = "hessian", final_linear=True)
    pred_hessian2 = spk.atomistic.Hessian2(n_in = n_atom_basis, hessian_key = "hessian")
    pred_hessian3 = spk.atomistic.Hessian3(n_in = n_atom_basis, hessian_key = "hessian")
    pred_hessian4 = spk.atomistic.Hessian4(n_in = n_atom_basis, hessian_key = "hessian")
    pred_hessian5 = spk.atomistic.Hessian5(n_in = n_atom_basis, hessian_key = "hessian")
    pred_hessian6 = spk.atomistic.Hessian6(n_in = n_atom_basis, hessian_key = "hessian")
    pred_newton_step = spk.atomistic.NewtonStep(n_in = n_atom_basis, newton_step_key = "newton_step")
    pred_best_direction = spk.atomistic.BestDirection(n_in = n_atom_basis, best_direction_key = "best_direction")
    pred_forces_copy = spk.atomistic.Forces2(n_in = n_atom_basis, forces_copy_key = "forces_copy")


    nnpot = spk.model.NeuralNetworkPotential(
        representation=paiNN,
        input_modules=[pairwise_distance],
        output_modules=[
            pred_energy,
            pred_forces, 
            # pred_hessian3, 
            pred_newton_step,
            # pred_best_direction,
            # pred_forces_copy
            ],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
        ]
    )

    output_energy = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.2,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.2,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_polarizability = spk.task.ModelOutput(
        name="polarizability",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    
    output_hessian = spk.task.ModelOutput(
        name="hessian",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_newton_step = spk.task.ModelOutput(
        name="newton_step",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.2,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_best_direction = spk.task.ModelOutput(
        name="best_direction",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.2,
        metrics = {
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    
    output_forces_copy = spk.task.ModelOutput(
        name="forces_copy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.2,
        metrics = {
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[
            output_energy, 
            output_forces, 
            # output_hessian, 
            output_newton_step,
            # output_best_direction,
            # output_forces_copy
            ],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
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
def load_model(
    model_path: str = None
):
    if model_path is None:
        model_path = os.path.join(get_training_directory(), "best_inference_model")

    # load model
    best_model = torch.load(model_path, map_location=device)
    
    return best_model


@logger
def evaluate_model(model, data, 
        showDiff=True,
        plotForces=True,
        plotNewtonStep=True,
        plotHessians=True,
        plotInverseHessians=True,
        plotBestDirection=True,
        plotForcesCopy=True):
    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )

    
    for i in range(len(data.test_dataset)):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
        )

        inputs = converter(atoms)
        results = model(inputs)
        
        if i < 10:
            plot(structure, results, showDiff, plotForces, plotNewtonStep, plotHessians, plotInverseHessians, plotBestDirection, plotForcesCopy)


@logger
def calculate_average_hessian(data):
    hessian = torch.zeros(27, 27)
    
    for i in range(len(data.dataset)):
        hessian += data.dataset[i]["hessian"]
    
    hessian = hessian/len(data.dataset)
    
    # plot_hessian(hessian)
    

@logger
def calculate_average_newton_step(data):
    hessian = torch.zeros(9, 3)
    
    for i in range(len(data.dataset)):
        datapoint = data.dataset[i]
        hessian += datapoint["newton_step"]
        plot_structure(datapoint)
    
    hessian = hessian/len(data.dataset)
    
    # plot_hessian(hessian)
    
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
    
    # model_name = "newton_step_model"
    # model_path = os.getcwd() + "\\maxim\\models\\" + model_name
    # model = load_model(
    #     model_path
    # )
    
    model = load_model()
    
    loss = evaluate_model(model, data, 
            showDiff=True,
            plotForces=True,
            plotNewtonStep=True,
            plotHessians=False,
            plotInverseHessians=False,
            plotBestDirection=False,
            plotForcesCopy=False
        )
    
    compare_directions(model, data, compare = ["forces", "newton_step"])
    
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
    