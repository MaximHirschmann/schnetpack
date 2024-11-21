
from typing import List, Dict
from matplotlib import pyplot as plt
import numpy as np

from optimization.OptimizationMetric import OptimizationMetricInterface


class GradientDescentResult:
    def __init__(self, score_history: List[Dict[str, float]], position_history: List[np.array], time_history: list) -> None:
        self.score_history = score_history # Each step gets a dictionary with the score (value) by the metric (key)
        self.time_history = time_history
        self.position_history = position_history
        
        self.total_time = time_history[-1]
        self.total_steps = len(score_history)
        

    def plot_score(self):
        plt.plot(self.time_history, self.score_history, "ro-")
        plt.show()


    def apply_metric(self, metric: OptimizationMetricInterface) -> None:
        for i in range(len(self.score_history)):
            if metric.name in self.score_history[i]:
                continue
            self.score_history[i][metric.name] = metric.calculate_energy(self.position_history[i])