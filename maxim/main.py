
import sys
import os 
from time import time

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
    filepath_hessian_db = os.path.join(os.getcwd(), 'maxim\\data\\ene_grad_hess_1000eth\\data.db')
    filepath_no_hessian_db = os.path.join(os.getcwd(), 'maxim\\data\\ene_grad_hess_1000eth\\data-no-hessian.db')

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
    pred_hessian = spk.atomistic.Hessian(n_in = n_atom_basis, hessian_key = "hessian")

    nnpot = spk.model.NeuralNetworkPotential(
        representation=paiNN,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces, pred_hessian],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets("energy", add_mean=True, add_atomrefs=False)
        ]
    )

    output_energy = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.09,
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
        loss_weight=0.9,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )


    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces, output_hessian],
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
        max_epochs=50, # for testing, we restrict the number of epochs
    )

    trainer.fit(task, datamodule=data)


def load_model():
    directory_training = os.path.join(os.getcwd(), "maxim\\data\\ene_grad_hess_1000eth")
    filepath_model = os.path.join(directory_training, "best_inference_model")

    # load model
    best_model = torch.load(filepath_model, map_location=device)
    
    return best_model



def evaluate_model(model, data):
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
        
        hessian_loss = output_hessian.calculate_loss(structure, results)
        
        loss += hessian_loss.item()
    loss = loss/len(data.test_dataset)
    
    print(f"Test loss: {loss}")
    
    return loss


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


    device = "cuda"

    data = load_data()
    
    train(data)
    
    model = load_model()
    
    loss = evaluate_model(model, data)