
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
    pred_hessian = spk.atomistic.Hessian(n_in = n_atom_basis, hessian_key = "hessian", final_linear=True)
    pred_hessian2 = spk.atomistic.Hessian2(n_in = n_atom_basis, hessian_key = "hessian")

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
        loss_weight=0.001,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.009,
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
        loss_weight=0.99,
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
        max_epochs=30, # for testing, we restrict the number of epochs
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
def evaluate_model(model, data, to_plot_forces = False, to_plot_hessian = False, compare_newton_steps = False):
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
            if to_plot_forces and to_plot_hessian:
                plot_forces_and_hessians(structure, results)
            elif to_plot_forces:
                plot_forces(structure["forces"], results["forces"])
            elif to_plot_hessian:
                plot_hessians(structure["hessian"], results["hessian"])
            
            if compare_newton_steps:
                compare_newton_step(structure, results)
        
        hessian_loss = output_hessian.calculate_loss(structure, results)
        
        loss += hessian_loss.item()
    loss = loss/len(data.test_dataset)
    
    print(f"Test loss: {loss}")
    
    return loss

def compare_newton_step(structure, results):
    hessian_true = structure["hessian"].numpy()
    hessian_pred = results["hessian"].detach().numpy()
    forces_true = structure["forces"].numpy()
    forces_pred = results["forces"].detach().numpy()
    forces_true = forces_true.reshape(27)
    forces_pred = forces_pred.reshape(27)
    
    newton_step_true = -np.linalg.solve(hessian_true, forces_true)
    newton_step_pred = -np.linalg.solve(hessian_pred, forces_pred)
    
    # normalize
    # newton_step_true = newton_step_true / np.linalg.norm(newton_step_true)
    # newton_step_pred = newton_step_pred / np.linalg.norm(newton_step_pred)
    
    print(f"True newton step: {newton_step_true}")
    print(f"Predicted newton step: {newton_step_pred}")
    print(f"Norm of difference: {np.linalg.norm(newton_step_true - newton_step_pred)}")
    print(f"Scalar product: {np.dot(newton_step_true, newton_step_pred) / (np.linalg.norm(newton_step_true) * np.linalg.norm(newton_step_pred))}")
    print("")
    



@logger
def calculate_average_hessian(data):
    hessian = torch.zeros(27, 27)
    
    for i in range(len(data.dataset)):
        hessian += data.dataset[i]["hessian"]
    
    hessian = hessian/len(data.dataset)
    
    plot_hessian(hessian)
    

def plot_hessian(hessian):
    hessian = hessian.numpy()
    
    plt.imshow(hessian)
    plt.title("Average hessian")
    plt.show()
        

def plot_forces(forces_true, forces_pred):
    forces_true, forces_pred = forces_true.numpy(), forces_pred.detach().numpy()
    forces_true = forces_true.reshape(3, 9)
    forces_pred = forces_pred.reshape(3, 9)
    vmin = min(forces_true.min(), forces_pred.min())
    vmax = max(forces_true.max(), forces_pred.max())
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axs[0].imshow(forces_true, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title("True forces")
    
    im2 = axs[1].imshow(forces_pred, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title("Predicted forces")
    
    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])
    
    plt.show()
    
    

def plot_hessians(hessian_true, hessian_pred):
    true, pred = hessian_true.numpy(), hessian_pred.detach().numpy()
    true_inv, pred_inv = np.linalg.inv(true), np.linalg.inv(pred)
    
    vmin = min(true.min(), pred.min())
    vmax = max(true.max(), pred.max())
    
    cmap='viridis'
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    im1 = axs[0, 0].imshow(true, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("True hessian")
    
    im2 = axs[0, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("Predicted hessian")

    im3 = axs[1, 0].imshow(true_inv, cmap=cmap)
    axs[1, 0].set_title("True inverse hessian")

    im4 = axs[1, 1].imshow(pred_inv, cmap=cmap)
    axs[1, 1].set_title("Predicted inverse hessian")

    # Add colorbars
    fig.colorbar(im1, ax=axs[0, 0])
    fig.colorbar(im2, ax=axs[0, 1])
    fig.colorbar(im3, ax=axs[1, 0])
    fig.colorbar(im4, ax=axs[1, 1])

    # Hide ticks
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 3x3 lines
        for i in range(1, 3):
            axs[0, 0].axhline(i*9-0.5, color='black')
            axs[0, 0].axvline(i*9-0.5, color='black')
            axs[0, 1].axhline(i*9-0.5, color='black')
            axs[0, 1].axvline(i*9-0.5, color='black')
            axs[1, 0].axhline(i*9-0.5, color='black')
            axs[1, 0].axvline(i*9-0.5, color='black')
            axs[1, 1].axhline(i*9-0.5, color='black')
            axs[1, 1].axvline(i*9-0.5, color='black')
    
    plt.tight_layout()
    plt.show()


def plot_forces_and_hessians(structure, results):
    forces_true = structure["forces"]
    forces_pred = results["forces"]
    hessian_true = structure["hessian"]
    hessian_pred = results["hessian"]
    
    # Plot forces
    forces_true, forces_pred = forces_true.numpy(), forces_pred.detach().numpy()
    forces_true = forces_true.reshape(3, 9)
    forces_pred = forces_pred.reshape(3, 9)
    vmin_force = min(forces_true.min(), forces_pred.min())
    vmax_force = max(forces_true.max(), forces_pred.max())
    
    # Plot Hessians
    true_hessian, pred_hessian = hessian_true.numpy(), hessian_pred.detach().numpy()
    true_inv, pred_inv = np.linalg.inv(true_hessian), np.linalg.inv(pred_hessian)
    
    vmin_hessian = min(true_hessian.min(), pred_hessian.min())
    vmax_hessian = max(true_hessian.max(), pred_hessian.max())
    
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    
    cmap = "inferno"
    
    # Plot true forces
    im1 = axs[0, 0].imshow(forces_true, cmap=cmap, vmin=vmin_force, vmax=vmax_force)
    axs[0, 0].set_title("True forces")
    fig.colorbar(im1, ax=axs[0, 0])
    
    # Plot predicted forces
    im2 = axs[0, 1].imshow(forces_pred, cmap=cmap, vmin=vmin_force, vmax=vmax_force)
    axs[0, 1].set_title("Predicted forces")
    fig.colorbar(im2, ax=axs[0, 1])
    
    # Plot true hessian
    im3 = axs[1, 0].imshow(true_hessian, cmap=cmap, vmin=vmin_hessian, vmax=vmax_hessian)
    axs[1, 0].set_title("True hessian")
    fig.colorbar(im3, ax=axs[1, 0])
    
    # Plot predicted hessian
    im4 = axs[1, 1].imshow(pred_hessian, cmap=cmap, vmin=vmin_hessian, vmax=vmax_hessian)
    axs[1, 1].set_title("Predicted hessian")
    fig.colorbar(im4, ax=axs[1, 1])
    
    # Plot true inverse hessian
    im5 = axs[2, 0].imshow(true_inv, cmap=cmap)
    axs[2, 0].set_title("True inverse hessian")
    fig.colorbar(im5, ax=axs[2, 0])
    
    # Plot predicted inverse hessian
    im6 = axs[2, 1].imshow(pred_inv, cmap=cmap)
    axs[2, 1].set_title("Predicted inverse hessian")
    fig.colorbar(im6, ax=axs[2, 1])

    # Hide ticks
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 3x3 lines
        for i in range(1, 3):
            axs[1, 0].axhline(i*9-0.5, color='black')
            axs[1, 0].axvline(i*9-0.5, color='black')
            axs[1, 1].axhline(i*9-0.5, color='black')
            axs[1, 1].axvline(i*9-0.5, color='black')
            axs[2, 0].axhline(i*9-0.5, color='black')
            axs[2, 0].axvline(i*9-0.5, color='black')
            axs[2, 1].axhline(i*9-0.5, color='black')
            axs[2, 1].axvline(i*9-0.5, color='black')
    
    # plt.tight_layout()
    plt.show()
    

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
    
    import matplotlib.pyplot as plt
    import numpy as np


    device = "cuda"

    data = load_data()
    
    #train(data)
    
    model = load_model()
    
    loss = evaluate_model(model, data, 
                          to_plot_forces=True,
                          to_plot_hessian=True, 
                          compare_newton_steps=False
                          )
    
    # calculate_average_hessian(data)