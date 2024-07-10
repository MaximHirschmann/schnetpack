import os
import sys
from Utils import load_data

schnetpack_dir = os.getcwd()
sys.path.insert(1, schnetpack_dir + "\\src")

import schnetpack as spk
from schnetpack.data import ASEAtomsData

import matplotlib.pyplot as plt
import numpy as np

def do_pca(data):
    hessians = []
    length = data.num_train + data.num_val + data.num_test
    for i in range(length):
        print(i)
        hessian = data.dataset[i]['original_hessian'].flatten()
        hessians.append(hessian)
    hessians = np.array(hessians)
    
    means = np.mean(hessians, axis=0)
    stds = np.std(hessians, axis=0)
    
    normalized_hessians = (hessians - means) / stds
    
    cov = np.cov(normalized_hessians.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    take = 10
    
    eigenvectors = eigenvectors[:, :take]
    eigenvalues = eigenvalues[:take]
    
    plt.bar(eigenvalues, height=1)
    plt.show()
    
    return means, stds, eigenvectors, eigenvalues
    
if __name__ == '__main__':
    data = load_data()
    do_pca(data)
    
