import matplotlib.pyplot as plt
import numpy as np
    


def plot(structure, 
         results, 
         showDiff=True,
         plotForces=True,
         plotNewtonStep=True,
         plotHessians=True,
         plotInverseHessians=True,
         plotBestDirection=True,
         plotForcesCopy=True):
    
    l = locals()
    rows = sum([int(v) for k, v in l.items() if type(v) is bool and k != "showDiff"])
    columns = 2 + showDiff

    fig, axs = plt.subplots(rows, columns, figsize=(10, 5*rows))
    
    def add_subplot(axs, row, col, data, title, vmin=None, vmax=None):
        if rows == 1:
            im = axs[col].imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
            axs[col].set_title(title)
            plt.colorbar(im, ax=axs[col])
            return
        else:
            im = axs[row, col].imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
            axs[row, col].set_title(title)
            plt.colorbar(im, ax=axs[row, col])

    def prepare_data(true_data, pred_data):
        true_data, pred_data = true_data.cpu().numpy(), pred_data.cpu().detach().numpy()
        return true_data, pred_data, min(true_data.min(), pred_data.min()), max(true_data.max(), pred_data.max())
    
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
        true_inv_hessian, pred_inv_hessian, vmin_inv_hessian, vmax_inv_hessian = prepare_data(structure["inv_hessian"], results["inv_hessian"])

        add_subplot(axs, row, 0, pred_inv_hessian, "Predicted inverse hessian", vmin_inv_hessian, vmax_inv_hessian)
        add_subplot(axs, row, 1, true_inv_hessian, "True inverse hessian", vmin_inv_hessian, vmax_inv_hessian)
        if showDiff:
            add_subplot(axs, row, 2, true_inv_hessian - pred_inv_hessian, "Difference", vmin_inv_hessian, vmax_inv_hessian)
        row += 1
        
    if plotBestDirection:
        true_best_direction, pred_best_direction, vmin_direction, vmax_direction = prepare_data(structure["best_direction"], results["best_direction"])
        true_best_direction, pred_best_direction = true_best_direction.reshape(3, 9), pred_best_direction.reshape(3, 9)
        
        add_subplot(axs, row, 0, pred_best_direction, "Predicted best direction", vmin_direction, vmax_direction)
        add_subplot(axs, row, 1, true_best_direction, "True best direction", vmin_direction, vmax_direction)
        if showDiff:
            add_subplot(axs, row, 2, true_best_direction - pred_best_direction, "Difference", vmin_direction, vmax_direction)
        row += 1

    if plotForcesCopy:
        _, _, vmin_force, vmax_force = prepare_data(structure["forces"], results["forces"])
        forces_copy_true, forces_copy_pred, _, _ = prepare_data(structure["forces_copy"], results["forces_copy"])
        forces_copy_true, forces_copy_pred = forces_copy_true.reshape(3, 9), forces_pred.reshape(3, 9)
        
        add_subplot(axs, row, 0, forces_copy_true, "Predicted forces copy", vmin_force, vmax_force)
        add_subplot(axs, row, 1, forces_copy_pred, "True forces copy", vmin_force, vmax_force)
        if showDiff:
            add_subplot(axs, row, 2, forces_copy_true - forces_copy_pred, "Difference", vmin_force, vmax_force)
        row += 1
        
    plt.tight_layout()
    plt.show()


def plot_structure(structure):
    positions = structure["_positions"].cpu().numpy()
    # forces = structure["forces"].cpu().numpy()
    forces = structure["forces"].cpu().numpy()
    
    scale_factor = 10.0
    forces_scaled = forces * scale_factor
        
    # Define colors for each atom (you can customize this as needed)
    # colors = plt.cm.jet(np.linspace(0, 1, positions_np.shape[0]))
    colors = np.array([
        "k",
        "k",
        "r",
        "b",
        "b",
        "b",
        "b",
        "b",
        "b"
    ])

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=100, label='Atom Positions')
    
    # Quiver plot for force vectors
    for i in range(positions.shape[0]):
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                forces_scaled[i, 0], forces_scaled[i, 1], forces_scaled[i, 2],
                color=colors[i], arrow_length_ratio=0.1, label='Force Vector' if i == 0 else "")

    # Adding labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Positions of Atoms')

    # Show the plot
    plt.legend()
    plt.show()
    
# just like plot_structure but for multiple structures
def plot_structures(structures):
    fig, axs = plt.subplots(len(structures), 1, figsize=(10, 5*len(structures)), subplot_kw={'projection': '3d'})
    
    for i, structure in enumerate(structures):
        positions = structure["_positions"].cpu().numpy()
        forces = structure["forces"].cpu().numpy()
        
        scale_factor = 10.0
        forces_scaled = forces * scale_factor

        colors = np.array([
            "k",
            "k",
            "r",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b"
        ])

        sc = axs[i].scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=100, label='Atom Positions')
        
        for j in range(positions.shape[0]):
            axs[i].quiver(positions[j, 0], positions[j, 1], positions[j, 2],
                    forces_scaled[j, 0], forces_scaled[j, 1], forces_scaled[j, 2],
                    color=colors[j], arrow_length_ratio=0.1, label='Force Vector' if j == 0 else "")

        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')
        axs[i].set_zlabel('Z')
        axs[i].set_title('3D Positions of Atoms')
        
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_atoms(list_of_atoms):
    # plot the atoms into the same graph
    
    fig, axs = plt.subplots(1, 1, figsize=(20, 20), subplot_kw={'projection': '3d'})
    
    markers = ['o', '^', 's', 'P', '*', 'X', 'D']
    
    for i, atoms in enumerate(list_of_atoms):
        positions = atoms.get_positions()
        colors = np.array([
            "k",
            "k",
            "r",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b"
        ])
        
        sc = axs.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=100, label='Atom Positions', marker = markers[i])
        
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_zlabel('Z')
    axs.set_title('3D Positions of Atoms')
    plt.legend()
    plt.show()
    
    
        