import matplotlib.pyplot as plt
import numpy as np
    
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


def plot_forces_and_hessians(structure, results, showDiff=True):
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
    
    columns = 2
    if showDiff:
        columns = 3
        
    fig, axs = plt.subplots(3, columns, figsize=(10, 15))
    
    cmap = "inferno"
    
    # Plot true forces
    im1 = axs[0, 0].imshow(forces_true, cmap=cmap, vmin=vmin_force, vmax=vmax_force)
    axs[0, 0].set_title("True forces")
    fig.colorbar(im1, ax=axs[0, 0])
    
    # Plot predicted forces
    im2 = axs[0, 1].imshow(forces_pred, cmap=cmap, vmin=vmin_force, vmax=vmax_force)
    axs[0, 1].set_title("Predicted forces")
    fig.colorbar(im2, ax=axs[0, 1])
    
    # Plot difference
    if showDiff:
        im3 = axs[0, 2].imshow(forces_true - forces_pred, cmap=cmap, vmin=vmin_force, vmax=vmax_force)
        axs[0, 2].set_title("Difference")
        fig.colorbar(im3, ax=axs[0, 2])
    
    # Plot true hessian
    im3 = axs[1, 0].imshow(true_hessian, cmap=cmap, vmin=vmin_hessian, vmax=vmax_hessian)
    axs[1, 0].set_title("True hessian")
    fig.colorbar(im3, ax=axs[1, 0])
    
    # Plot predicted hessian
    im4 = axs[1, 1].imshow(pred_hessian, cmap=cmap, vmin=vmin_hessian, vmax=vmax_hessian)
    axs[1, 1].set_title("Predicted hessian")
    fig.colorbar(im4, ax=axs[1, 1])
    
    # Plot difference
    if showDiff:
        im5 = axs[1, 2].imshow(true_hessian - pred_hessian, cmap=cmap, vmin=vmin_hessian, vmax=vmax_hessian)
        axs[1, 2].set_title("Difference")
        fig.colorbar(im5, ax=axs[1, 2])
    
    # Plot true inverse hessian
    im5 = axs[2, 0].imshow(true_inv, cmap=cmap)
    axs[2, 0].set_title("True inverse hessian")
    fig.colorbar(im5, ax=axs[2, 0])
    
    # Plot predicted inverse hessian
    im6 = axs[2, 1].imshow(pred_inv, cmap=cmap)
    axs[2, 1].set_title("Predicted inverse hessian")
    fig.colorbar(im6, ax=axs[2, 1])
    
    # Plot difference
    if showDiff:
        im7 = axs[2, 2].imshow(true_inv - pred_inv, cmap=cmap)
        axs[2, 2].set_title("Difference")
        fig.colorbar(im7, ax=axs[2, 2])



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
            axs[1, 2].axhline(i*9-0.5, color='black')
            axs[1, 2].axvline(i*9-0.5, color='black')
            axs[2, 0].axhline(i*9-0.5, color='black')
            axs[2, 0].axvline(i*9-0.5, color='black')
            axs[2, 1].axhline(i*9-0.5, color='black')
            axs[2, 1].axvline(i*9-0.5, color='black')
            axs[2, 2].axhline(i*9-0.5, color='black')
            axs[2, 2].axvline(i*9-0.5, color='black')
    
    # plt.tight_layout()
    plt.show()

def plot(structure, 
         results, 
         showDiff=True,
         plotForces=True,
         plotNewtonStep=True,
         plotHessians=True,
         plotInverseHessians=True):
    
    def add_subplot(axs, row, col, data, title, vmin=None, vmax=None):
        im = axs[row, col].imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[row, col].set_title(title)
        plt.colorbar(im, ax=axs[row, col])

    def prepare_data(true_data, pred_data):
        true_data, pred_data = true_data.numpy(), pred_data.detach().numpy()
        return true_data, pred_data, min(true_data.min(), pred_data.min()), max(true_data.max(), pred_data.max())

    rows = sum([plotForces, plotNewtonStep, plotHessians, plotInverseHessians])
    columns = 2 + showDiff

    fig, axs = plt.subplots(rows, columns, figsize=(10, 5*rows))
    
    row = 0
    if plotForces:
        forces_true, forces_pred, vmin_force, vmax_force = prepare_data(structure["forces"], results["forces"])
        forces_true, forces_pred = forces_true.reshape(3, 9), forces_pred.reshape(3, 9)
        
        add_subplot(axs, row, 0, forces_true, "Predicted forces", vmin_force, vmax_force)
        add_subplot(axs, row, 1, forces_pred, "True forces", vmin_force, vmax_force)
        if showDiff:
            add_subplot(axs, row, 2, forces_true - forces_pred, "Difference", vmin_force, vmax_force)
        row += 1

    if plotNewtonStep:
        true_newton, pred_newton, vmin_newton, vmax_newton = prepare_data(structure["newton_step"], results["newton_step"])
        true_newton, pred_newton = true_newton.reshape(3, 9), pred_newton.reshape(3, 9)
        
        add_subplot(axs, row, 0, pred_newton, "Predicted Newton step", vmin_newton, vmax_newton)
        add_subplot(axs, row, 1, true_newton, "True Newton step", vmin_newton, vmax_newton)
        if showDiff:
            add_subplot(axs, row, 2, true_newton - pred_newton, "Difference", vmin_newton, vmax_newton)
        row += 1

    if plotHessians:
        true_hessian, pred_hessian, vmin_hessian, vmax_hessian = prepare_data(structure["hessian"], results["hessian"])

        add_subplot(axs, row, 0, pred_hessian, "Predicted hessian", vmin_hessian, vmax_hessian)
        add_subplot(axs, row, 1, true_hessian, "True hessian", vmin_hessian, vmax_hessian)
        if showDiff:
            add_subplot(axs, row, 2, true_hessian - pred_hessian, "Difference", vmin_hessian, vmax_hessian)
        row += 1

    if plotInverseHessians:
        true_inv, pred_inv = np.linalg.inv(true_hessian), np.linalg.inv(pred_hessian)

        add_subplot(axs, row, 0, pred_inv, "Predicted inverse hessian")
        add_subplot(axs, row, 1, true_inv, "True inverse hessian")
        if showDiff:
            add_subplot(axs, row, 2, true_inv - pred_inv, "Difference")
        row += 1

    plt.tight_layout()
    plt.show()
