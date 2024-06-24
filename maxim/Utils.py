
import sys
import os 
from time import time
from plotting import plot, plot_structure

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
import schnetpack.transform as trn

import torch
import time
from functools import wraps

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the wrapped function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        return result, elapsed_time  # Return both the result and the elapsed time
    
    return wrapper

def get_training_directory():
    return os.path.join(os.getcwd(), "training\\logs")


def load_model(
    model_path: str = None,
    device: torch.device = torch.device("cuda")
):
    if model_path is None:
        model_path = os.path.join(get_training_directory(), "best_inference_model")

    if not "\\" in model_path and not "/" in model_path:
        model_path = os.getcwd() + "\\maxim\\models\\" + model_path
        
    # load model
    best_model = torch.load(model_path, map_location=device)
    
    return best_model


def load_data():
    # filepath_hessian_db = os.path.join(os.getcwd(), 'maxim\\data\\ene_grad_hess_1000eth\\data.db')
    filepath_hessian_db = os.path.join(os.getcwd(), "maxim\\data\\custom_database.db")
    print(filepath_hessian_db)

    hessianData = spk.data.AtomsDataModule(
        filepath_hessian_db, 
        distance_unit="Ang",
        property_units={
                "energy": "Hartree",
                "forces": "Hartree/Bohr",
                "hessian": "Hartree/Bohr/Bohr",
                "inv_hessian": "Bohr/Hartree/Bohr",
                "newton_step": "Hartree/Bohr",
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