import sys
import os 

schnetpack_dir = os.getcwd()

sys.path.insert(1, schnetpack_dir + "\\maxim\\src")
from optimization.best_evaluate import best_evaluate


from typing import List
import matplotlib.pyplot as plt
import numpy as np

    
def plot_hessian(hessian):
    # Assumes the hessian is of shape 27 x 27
    if type(hessian) is not np.ndarray:
        hessian = hessian.cpu().detach().numpy()

    fig, ax = plt.subplots()
    
    # Plot the hessian
    cax = ax.imshow(hessian, cmap="viridis")
    plt.colorbar(cax)

    # Adding fine grid lines every 3 cells
    ax.set_xticks(np.arange(-0.5, 27, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 27, 3), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    # Sublabels (x, y, z rotating)
    sublabels = ['x', 'y', 'z']
    tick_labels = [sublabels[i % 3] for i in range(27)]
    
    main_labels = [r"$C_1$", r"$C_2$", r"$O_{C1}$", r"$H_{C1}$", r"$H_{C1}$", r"$H_{C2}$", r"$H_{C2}$", r"$H_{C2}$", r"$H_{O}$"]
    
    # Set sublabels for x-axis and y-axis
    ax.set_xticks(np.arange(27))
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(np.arange(27))
    ax.set_yticklabels(tick_labels)

    # Set main labels every 3rd column and row
    for i, label in enumerate(main_labels):
        ax.text(i * 3 + 1, -1.5, label, ha='center', va='center', fontsize=12, color='black', fontweight='bold', transform=ax.transData)
        ax.text(27 + .5, i * 3 + 1, label, ha='center', va='center', fontsize=12, color='black', fontweight='bold', transform=ax.transData)
    
    plt.show()
    

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


def plot2(structure, results, properties, showDiff = True):
    ncols = 2 + showDiff
    nrows = len(properties)

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5*nrows))

    def add_subplot(axs, row, col, data, title, vmin=None, vmax=None):
        if nrows == 1:
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
        if true_data.ndim == 1:
            true_data = true_data[..., np.newaxis]
        if pred_data.ndim == 1:
            pred_data = pred_data[..., np.newaxis]
        
        vmin = min(true_data.min(), pred_data.min())
        vmax = max(true_data.max(), pred_data.max())
        
        if -1 < vmin < 0 and 0 < vmax < 1:
            vmin = -0.75
            vmax = 0.75

        return true_data, pred_data, vmin, vmax
    
    row = 0
    for property in properties:
        true_data, pred_data, vmin, vmax = prepare_data(structure[property], results[property])
        
        add_subplot(axs, row, 0, pred_data, f"Predicted {property}", vmin, vmax)
        add_subplot(axs, row, 1, true_data, f"True {property}", vmin, vmax)
        if showDiff:
            add_subplot(axs, row, 2, true_data - pred_data, f"Difference", vmin, vmax)
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
    colors_by_number = {
        1: "b",
        6: "k",
        8: "r"
    }
    
    for i, atoms in enumerate(list_of_atoms):
        colors = [colors_by_number[num] for num in atoms.numbers]
        positions = atoms.get_positions()
        sc = axs.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=100, label='Atom Positions', marker = markers[i])
        
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_zlabel('Z')
    axs.set_title('3D Positions of Atoms')
    plt.legend()
    plt.show()
    
    

def plot_average(histories: List[List[str]], labels, title = "Average Energy History"):
    # convert each history to numpy array and make them the same length
    max_length = max([len(history) for history in histories[0]])
    np_histories = np.zeros((len(histories[0]), max_length))
    for i in range(len(histories[0])):
        for j in range(len(histories)):
            fitted = histories[j][i][:max_length]
            if len(fitted) < max_length:
                fitted += [fitted[-1]] * (max_length - len(fitted))
            np_histories[i] += fitted
    np_histories /= len(histories)    

    # plot average
    plt.figure()
    
    for i, label in enumerate(labels):
        plt.plot(np_histories[i], label=label)
    
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_average_over_time(
    results, #: List[List[GradientDescentResult]] 
    labels, 
    title = "Average Energy History Over Time",
    timeframe = 3):
    
    timesteps = np.linspace(0, timeframe, 100) # from t = 0s to t = 10s with 100 timesteps
    
    # Transpose
    results_by_strategy = [[] for _ in labels]
    for i in results:
        for j, result in enumerate(i):
            results_by_strategy[j].append(result)
    
    averages = [np.zeros(len(timesteps)) for _ in labels]
    for i, strategy_result in enumerate(results_by_strategy):
        for result in strategy_result:
            scores = [0]
            i_timesteps = 0
            i_history = 0
            while i_timesteps < len(timesteps) and i_history < len(result.score_history):
                if result.time_history[i_history] <= timesteps[i_timesteps]:
                    scores[-1] = result.score_history[i_history]
                    i_history += 1
                else:
                    scores.append(scores[-1])
                    i_timesteps += 1
            scores = scores[:len(timesteps)]
            scores = scores + [scores[-1]] * (len(timesteps) - len(scores))
            averages[i] += scores
        averages[i] /= len(strategy_result)
        
    plt.figure()
    for i, label in enumerate(labels):
        plt.plot(timesteps, averages[i], label=label)
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        
    plt.title(title)
    plt.legend()
    plt.show()
    
    
def plot_all_histories(histories: List[List[str]], labels, title = "All Energy Histories"):
    cols = 6
    fig, axs = plt.subplots(int(np.ceil(len(histories) / cols)), cols, figsize=(15, 15))
    
    for i, history in enumerate(histories):
        if len(histories) <= cols:
            ax = axs[i]
        else:
            ax = axs[i // cols, i % cols]
        for j, strategy_history in enumerate(histories[i]):
            ax.plot(strategy_history, label=labels[j])
            
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.suptitle(title)
    plt.show()
        
def plot_true_values(results, labels, base_atom, base_internal_energy):
    title = "Final positions evaluated by best model"
    labels = labels + ["Base"]
    
    length = len(results[0]) + 1
    final_internal_values = np.zeros(length)
    final_global_values = np.zeros(length)
    for i in range(len(results)):
        for j in range(len(results[i])):
            final_internal_values[j] += results[i][j].score_history[-1]
            final_global_values[j] += best_evaluate(results[i][j].final_atom)
    final_internal_values /= len(results)
    final_global_values /= len(results)
    
    final_internal_values[-1] = base_internal_energy
    final_global_values[-1] = best_evaluate(base_atom)
    
    # display two bar plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    rect1 = axs[0].bar(np.arange(length), final_internal_values, 0.35, label="Internal")
    rect2 = axs[1].bar(np.arange(length), final_global_values, 0.35, label="Global")
    
    axs[0].set_xticks(np.arange(length))
    axs[0].set_xticklabels(labels)
    axs[0].set_title(title)
    # set y limit
    axs[0].set_ylim([final_internal_values.min(), final_internal_values.max()])
    
    axs[1].set_xticks(np.arange(length))
    axs[1].set_xticklabels(labels)
    axs[1].set_ylim([final_global_values.min(), final_global_values.max()])
    
    # different colors
    for i in range(len(rect1)):
        rect1[i].set_color('C0')
        rect2[i].set_color('C1')
        
    axs[0].legend()
    axs[1].legend()
    plt.show()
        
        
    # bar plot
    # fig, axs = plt.subplots()
    
    # rect1 = axs.bar(np.arange(len(labels)), final_internal_values, 0.35, label="Internal")
    # rect2 = axs.bar(np.arange(len(labels)) + 0.35, final_global_values, 0.35, label="Global")
    
    # axs.set_xticks(np.arange(len(labels)))
    # axs.set_xticklabels(labels)
    # axs.set_title(title)
    # axs.legend()
    # plt.show()
    
if __name__ == "__main__":
    hessian = np.random.rand(27, 27)
    hessian = (hessian + hessian.T) / 2
    hessian += np.eye(27) 
    hessian -= 1
    
    plot_hessian(hessian)
     
    
    