
import sys
import os 
from time import time
from plotting import plot_forces, plot_hessian, plot_hessians, plot_forces_and_hessians, plot
    

def logger(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        print(f"----- running {func.__name__} -----")
        result = func(*args, **kwargs)
        print(f"----- {func.__name__} ran in {time()-t0:.2f} seconds -----")
        return result
    return wrapper


@logger
def load_data():
    # filepath_hessian_db = os.path.join(os.getcwd(), 'maxim\\data\\ene_grad_hess_1000eth\\data.db')
    filepath_hessian_db = os.path.join(os.getcwd(), "maxim\\data\\custom_database.db")

    hessianData = spk.data.AtomsDataModule(
        filepath_hessian_db, 
        distance_unit="Ang",
        property_units={"energy": "Hartree",
                        "forces": "Hartree/Bohr",
                        "hessian": "Hartree/Bohr/Bohr"
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
    epochs = 5
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


    nnpot = spk.model.NeuralNetworkPotential(
        representation=paiNN,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces, pred_hessian3, pred_newton_step],
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
        loss_weight=0.4,
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
        loss_weight=0.4,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )


    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces, output_hessian, output_newton_step],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    directory_training = os.path.join(os.getcwd(), "maxim\\data\\ene_grad_hess_1000eth")
    filepath_model = os.path.join(directory_training, "best_inference_model")

    logger = pl.loggers.TensorBoardLogger(save_dir=directory_training)
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
        default_root_dir=directory_training,
        max_epochs=epochs,
    )

    trainer.fit(task, datamodule=data)

@logger
def load_model():
    directory_training = os.path.join(os.getcwd(), "maxim\\data\\ene_grad_hess_1000eth")
    filepath_model = os.path.join(directory_training, "best_inference_model")

    # load model
    best_model = torch.load(filepath_model, map_location=device)
    
    return best_model


@logger
def evaluate_model(model, data, 
        showDiff=True,
        plotForces=True,
        plotNewtonStep=True,
        plotHessians=True,
        plotInverseHessians=True):
    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )

    # create atoms object from dataset
    output_hessian = spk.task.ModelOutput(
        name="hessian",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )
    
    loss = 0
    for i in range(len(data.test_dataset)):
        structure = data.test_dataset[i]
        atoms = Atoms(
            numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
        )

        inputs = converter(atoms)
        results = model(inputs)
        
        results["hessian"] = results["hessian"].to("cpu")
        results["forces"] = results["forces"].to("cpu")
        
        if i < 5:
            plot(structure, results, showDiff, plotForces, plotNewtonStep, plotHessians, plotInverseHessians)

        hessian_loss = output_hessian.calculate_loss(structure, results)
        
        loss += hessian_loss.item()
    loss = loss/len(data.test_dataset)
    
    print(f"Test loss: {loss}")
    
    return loss



@logger
def calculate_average_hessian(data):
    hessian = torch.zeros(27, 27)
    
    for i in range(len(data.dataset)):
        hessian += data.dataset[i]["hessian"]
    
    hessian = hessian/len(data.dataset)
    
    plot_hessian(hessian)
    

    

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
    
    import numpy as np
    

    device = "cpu"

    data = load_data()
    
    train(data)
    
    model = load_model()
    
    loss = evaluate_model(model, data, 
            showDiff=True,
            plotForces=True,
            plotNewtonStep=True,
            plotHessians=False,
            plotInverseHessians=False
        )
    
    # calculate_average_hessian(data)