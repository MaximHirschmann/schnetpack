from optimization.optimization_comparison import compare
from plotting import plot_hessian2
from Utils import load_data, load_model
from train import evaluate_model, plot_kronecker_products, train

import random
import numpy as np
import matplotlib.pyplot as plt

def plot_random_hessian():
    data = load_data()
    
    # 4 x 3 subplots
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(18, 12)
    fig.tight_layout(pad=3.0)
    
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[i, j]
            rand_idx = random.randint(0, len(data.dataset)) 
            hessian = data.dataset[rand_idx]["original_hessian"]
            
            plot_hessian2(hessian, ax = ax)
    
    plt.savefig("random_hessians.svg")
    plt.show()
    
def train_fn():
    data = load_data()
    
    # calculate_average_newton_step(data)
    
    train(data, continue_last_training=False)
    
    model = load_model()

    loss = evaluate_model(model, data, 
            properties = ["diagonal"],
            title="Diagonal L1 prediction",
            showDiff=True,
        )
    
    # plot_kronecker_products(model, data)
    
    print("END")
    
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
    
if __name__ == '__main__': 
    compare()
    # plot_random_hessian()
    # train_fn()
    # check_dataset()