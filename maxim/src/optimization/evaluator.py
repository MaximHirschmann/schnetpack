import sys
import os
from typing import List 
from datatypes import OptimizationEvaluationXAxis, OptimizationEvaluationAllPlotsFix, GradientDescentResult
from plotting  import plot_average, plot_all_runs, plot_atom
from Utils import load_data, get_model_path
from ase import Atoms
import matplotlib.pyplot as plt

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")
import schnetpack as spk
from schnetpack.data import ASEAtomsData
import schnetpack.transform as trn
from schnetpack.interfaces import SpkCalculator
