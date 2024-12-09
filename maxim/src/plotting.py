from enum import Enum, StrEnum
import sys
import os

from optimization.OptimizationMetric import EnergyMetric, OptimizationMetricInterface 

schnetpack_dir = os.getcwd()

sys.path.insert(1, schnetpack_dir + "\\maxim\\src")
from datatypes import GradientDescentResult
from datatypes import OptimizationEvaluationXAxis, OptimizationEvaluationYAxis
from datatypes import OptimizationEvaluationAllPlotsFix
from optimization.best_evaluate import best_evaluate


from typing import List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from datatypes import OptimizationEvaluationXAxis

    
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
    
def plot_hessian2(hessian, ax=None, colorbar=True, vmin=None, vmax = None):
    # Assumes the hessian is of shape 27 x 27
    if type(hessian) is not np.ndarray:
        hessian = hessian.cpu().detach().numpy()

    if ax is None:
        fig, ax = plt.subplots()

    # Plot the hessian
    if vmin is None or vmax is None:
        cax = ax.imshow(hessian, cmap="viridis")
    else:
        cax = ax.imshow(hessian, cmap="viridis", vmin=vmin, vmax=vmax)
    
    # Optionally add a colorbar (assumes plt.colorbar if outside of subplot control)
    if colorbar:
        plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.1)

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
        ax.text(27, i * 3 + 1, label, ha='left', va='center', fontsize=12, color='black', fontweight='bold', transform=ax.transData)

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

def plot2(structure, results, properties, title = "", showDiff = True):
    ncols = 2 + showDiff
    nrows = len(properties)

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5*nrows))

    def add_subplot(axs, row, col, data, title, vmin=None, vmax=None):
        ax = axs[row, col] if nrows > 1 else axs[col]
        
        if data.shape == (27, 27):
            plot_hessian2(data, ax, colorbar=True, vmin=vmin, vmax=vmax)
            ax.set_title(title, pad=20)
            return
        else:
            im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            

    def prepare_data(true_data, pred_data):
        true_data, pred_data = true_data.cpu().numpy(), pred_data.cpu().detach().numpy()
        if true_data.ndim == 1:
            true_data = true_data[..., np.newaxis]
        if pred_data.ndim == 1:
            pred_data = pred_data[..., np.newaxis]
        
        vmin = min(true_data.min(), pred_data.min())
        vmax = max(true_data.max(), pred_data.max())
        
        # if -1 < vmin < 0 and 0 < vmax < 1:
        #     vmin = -0.75
        #     vmax = 0.75

        return true_data, pred_data, vmin, vmax
    
    row = 0
    for property in properties:
        true_data, pred_data, vmin, vmax = prepare_data(structure[property], results[property])
        
        add_subplot(axs, row, 0, pred_data, f"Predicted {property}", vmin, vmax)
        add_subplot(axs, row, 1, true_data, f"True {property}", vmin, vmax)
        if showDiff:
            add_subplot(axs, row, 2, true_data - pred_data, f"Difference", vmin, vmax)
        row += 1
        
    if title != "":
        plt.suptitle(title)
    else:
        plt.suptitle("Comparison of predicted and true properties")
    plt.tight_layout()
    plt.show()
    
def plot_average(histories: List[List[str]], labels, title = "Average Energy History"):
    # Convert each history to numpy array and make them the same length
    max_length = max(len(history) for history in histories[0])
    np_histories = np.zeros((len(histories[0]), max_length))
    
    for i in range(len(histories[0])):
        for j in range(len(histories)):
            fitted = histories[j][i][:max_length]
            if len(fitted) < max_length:
                fitted += [fitted[-1]] * (max_length - len(fitted))
            np_histories[i] += fitted
    
    np_histories /= len(histories)

    # Plot average
    plt.figure()
    lines = []
    
    for i, label in enumerate(labels):
        line, = plt.plot(np_histories[i], label=label)  # Capture line object
        lines.append(Line2D([0], [0], color=line.get_color(), linewidth=3))  # Custom handle with thicker line
    
    # x axis
    plt.xlabel("Iteration Steps")
    plt.xlim([0, 200])
    
    # y axis
    plt.ylabel("Energy in Hartree")
    plt.ylim([np_histories.min(), -0.012])
    
    plt.title(title)
    plt.legend(handles=lines, labels=labels)  # Use custom legend handles
    plt.show()
    

    
    
def get_x_axis_values(
    results: List[List[GradientDescentResult]],
    x_axis: OptimizationEvaluationXAxis
) -> List[List[List[float]]]:
    """
    Extract the x-axis values (either iteration steps or real time) 
    based on the x_axis parameter.
    """
    if x_axis == OptimizationEvaluationXAxis.Iteration:
        # Use iteration steps as x-axis
        return [
            [list(range(len(result.position_history))) for result in strategy_results]
            for strategy_results in results
        ]
    elif x_axis == OptimizationEvaluationXAxis.Time:
        # Use time history as x-axis
        return [
            [result.time_history for result in strategy_results]
            for strategy_results in results
        ]
    else:
        raise ValueError(f"Unsupported x_axis value: {x_axis}")


def plot_average(
    results: List[List[GradientDescentResult]],
    labels: List[str],
    metric: OptimizationMetricInterface,
    x_axis: OptimizationEvaluationXAxis = OptimizationEvaluationXAxis.Iteration,
    title="Average Energy History Over Time",
):
    # Get x-axis values dynamically
    x_values = get_x_axis_values(results, x_axis)
    
    # Extract scores: runs x strategies x iteration_steps
    scores = [[[scores[metric.name] for scores in result.score_history] 
               for result in strategy_results] for strategy_results in results]
    
    # Transpose to strategies x runs x iteration_steps
    scores = [list(strategy_scores) for strategy_scores in zip(*scores)]
    x_values = [list(strategy_x_values) for strategy_x_values in zip(*x_values)]
    
    average_scores = []
    average_x_values = []
    if x_axis == OptimizationEvaluationXAxis.Iteration:
        for strategy_scores in scores:
            max_length = max(len(run_scores) for run_scores in strategy_scores)
            sums = np.zeros(max_length)
            counts = np.zeros(max_length)
            for run_scores in strategy_scores:
                for i in range(max_length):
                    if i < len(run_scores):
                        sums[i] += run_scores[i]
                        counts[i] += 1
                    else:
                        sums[i] += run_scores[-1]
                        counts[i] += 1
            averages = sums / counts
            average_scores.append(averages)
            average_x_values.append(list(range(len(averages))))
    elif x_axis == OptimizationEvaluationXAxis.Time:
        even_time = np.linspace(0, 3, 100)
        for strategy_scores, strategy_x_values in zip(scores, x_values):
            averages = np.zeros(len(even_time))
            counts = np.zeros(len(even_time))
            for run_scores, run_x_values in zip(strategy_scores, strategy_x_values):
                # choose the value of the closest time point
                for i, time in enumerate(even_time):
                    closest_idx = np.argmin(np.abs(np.array(run_x_values) - time))
                    averages[i] += run_scores[closest_idx]
                    counts[i] += 1
                    
            averages /= counts
            average_scores.append(averages)
            average_x_values.append(even_time)
        
    # Initialize plot
    plt.figure(figsize=(10, 6))
    
    # Plot averages and prepare legend handles
    lines = []
    for i, (label, averages, x_vals) in enumerate(zip(labels, average_scores, average_x_values)):
        line, = plt.plot(x_vals, averages, label=label, linewidth=2)
        lines.append(Line2D([0], [0], color=line.get_color(), linewidth=3))
    
    # Plot aesthetics
    plt.xlabel("Time (s)" if x_axis == OptimizationEvaluationXAxis.Time else "Iteration Steps")
    max_x = None
    if x_axis == OptimizationEvaluationXAxis.Iteration:
        max_x = 300
    elif x_axis == OptimizationEvaluationXAxis.Time:
        max_x = 3
    plt.xlim([0, min(max(len(x) for x in average_x_values), max_x)])  # Custom range for x-axis
    plt.ylabel(metric.y_axis)
    # average_after_10_steps = np.mean([avg[10] for avg in average_scores], axis=0)
    if metric.name == "basic":
        plt.ylim([-97095, -97070])
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(handles=lines, labels=labels, loc='best')
    plt.tight_layout()
    plt.show()
    
def plot_all_runs(
    results: List[List[GradientDescentResult]],
    labels: List[str],
    metric: OptimizationMetricInterface,
    fix: OptimizationEvaluationAllPlotsFix,
    x_axis: OptimizationEvaluationXAxis = OptimizationEvaluationXAxis.Iteration,
    title="Optimization Results Across Runs",
):
    """
    Plot all individual runs for each strategy using subplots.
    Each subplot corresponds to one strategy or run, depending on the fix.
    """
    # Get x-axis values dynamically
    x_values = get_x_axis_values(results, x_axis)

    # Extract scores: runs x strategies x iteration_steps
    scores = [[[scores[metric.name] for scores in result.score_history] 
               for result in strategy_results] for strategy_results in results]

    # Transpose to strategies x runs x iteration_steps
    scores_T = [list(strategy_scores) for strategy_scores in zip(*scores)]
    x_values = [list(strategy_x_values) for strategy_x_values in zip(*x_values)]

    # Determine the number of rows and columns for subplots
    if fix == OptimizationEvaluationAllPlotsFix.Strategy:
        n_rows = len(labels)
        n_cols = 1
    elif fix == OptimizationEvaluationAllPlotsFix.Run:
        n_rows = 3
        n_cols = 1 + (len(scores) - 1) // n_rows
    else:
        raise ValueError(f"Unsupported fix type: {fix}")

    # Initialize subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows), sharex=True)

    # Ensure axes are iterable (handle case with single subplot)
    if n_rows == 1:
        axes = [axes]

    max_x = None
    if x_axis == OptimizationEvaluationXAxis.Iteration:
        max_x = 200
    elif x_axis == OptimizationEvaluationXAxis.Time:
        max_x = 2 # seconds
        
    # Plot based on fix type
    if fix == OptimizationEvaluationAllPlotsFix.Strategy:
        for i, (strategy_scores, strategy_x_values, label) in enumerate(zip(scores_T, x_values, labels)):
            ax = axes[i // n_cols, i % n_cols]
            for run_scores, run_x_values in zip(strategy_scores, strategy_x_values):
                ax.plot(run_x_values, run_scores, alpha=0.7)
            ax.set_title(label)
            # ax.set_xlabel("Time (s)" if x_axis == OptimizationEvaluationXAxis.Time else "Iteration Steps")
            ax.set_ylabel(metric.y_axis)
            ax.grid(True, linestyle='--', alpha=0.5)
            # ax.set_ylim([min(min(score) for score in strategy_scores), -0.012])
            ax.set_xlim([0, min(max(len(x) for x in strategy_x_values), max_x)])

    elif fix == OptimizationEvaluationAllPlotsFix.Run:
        for i in range(len(scores)):
            ax = axes[i // n_cols, i % n_cols]
            for j in range(len(scores[i])):
                ax.plot(x_values[j][i], scores[i][j], alpha=0.7, label=labels[j])
                ax.set_title(f"Run {i + 1}")
                # ax.set_xlabel("Time (s)" if x_axis == OptimizationEvaluationXAxis.Time else "Iteration Steps")
                ax.set_ylabel(metric.y_axis)
                ax.grid(True, linestyle='--', alpha=0.5)
                min_value = min(score for score in scores[i][j])
                max_value = max(score for score in scores[i][j])
                ax.set_ylim([
                    min_value - ((max_value - min_value) * 0.1), 
                    max_value + ((max_value - min_value) * 0.1)
                    ])
                ax.set_yticks([min_value, max_value])
                ax.set_xlim([0, max_x])

    # Overall title and layout adjustments
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.text(0.5, 0.04, "Time (s)" if x_axis == OptimizationEvaluationXAxis.Time else "Iteration Steps", ha='center')
    # fig.text(0.04, 0.5, "Energy in Hartree", va='center', rotation='vertical')
    
    # legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    
    # plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.show()
    

def plot_atom(positions, numbers, title = ""):
    colors = {6: "black", 8: "red", 1: "blue"}
    colors = [colors[n] for n in numbers]
    charges = {6: "C", 8: "O", 1: "H"}
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # draw the bond between atom (0, 2) and (2, 8)
    for i, j in [(0, 2), (2, 8), (0, 1), (0, 3), (0, 4), (1, 7), (1, 5), (1, 6)]:
        x = [positions[i][0], positions[j][0]]
        y = [positions[i][1], positions[j][1]]
        z = [positions[i][2], positions[j][2]]
        ax.plot(x, y, z, c="gray")
        
    for i, (x, y, z) in enumerate(positions):
        ax.scatter(x, y, z, c=colors[i], label=i, s=100)
        ax.text(x, y, z + 0.15, f"{charges[numbers[i]]}", color='black')  # Shift the label slightly above the point
    
    # remove the axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    
    if title != "":
        plt.title(title)
        
    # rotate
    ax.view_init(elev=30, azim=20)

    # plt.legend()
    # plt.show()
    
    
if __name__ == "__main__":
    hessian = np.random.rand(27, 27)
    hessian = (hessian + hessian.T) / 2
    hessian += np.eye(27) 
    hessian -= 1
    
    plot_hessian(hessian)
     
    
    