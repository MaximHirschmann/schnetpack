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
        hessian = np.array(hessian)
        hessians.append(hessian)
    hessians = np.array(hessians) # shape: (num_samples, 729)
    
    means = np.mean(hessians, axis=0)
    stds = np.std(hessians, axis=0)
    
    normalized_hessians = (hessians - means) / stds
    
    cov = np.cov(normalized_hessians.T) # shape: (729, 729)
    plt.imshow(cov)
    plt.show()
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov) 
    
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    plt.plot(eigenvalues)
    plt.show()

    take = 20
    
    eigenvectors = eigenvectors[:take, :]
    eigenvalues = eigenvalues[:take]
    
    for i in range(10):
        plt.imshow(eigenvectors[i].reshape(27, 27))
        plt.show()
    
    return means, stds, eigenvectors, eigenvalues
    
if __name__ == '__main__':
    data = load_data()
    do_pca(data)
    
