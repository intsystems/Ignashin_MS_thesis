import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from ..graph import SkyGraph
from .agent import SkyAgent

class LinearSkyAgent(SkyAgent):
    def __init__(self, num_nodes, beta_cost=1.0, beta_capacity=1.0, stay_prob=0.2):
        """
        num_nodes: количество вершин
        beta_cost: веса стоимости (уникальные для каждой вершины или одно значение)
        beta_capacity: веса пропускной способности (уникальные для каждой вершины или одно значение)
        stay_prob: вероятность остаться в вершине (уникальная или одно значение)
        """
        super().__init__(num_nodes)
        
        # Валидация параметров
        self.beta_cost = self._validate_and_expand(beta_cost, num_nodes, min_value=0, name="β_cost")
        self.beta_capacity = self._validate_and_expand(beta_capacity, num_nodes, min_value=0, name="β_capacity")
        self.stay_prob = self._validate_and_expand(stay_prob, num_nodes, min_value=0, max_value=1, name="stay_prob")

    def _validate_and_expand(self, param, num_nodes, min_value=None, max_value=None, name="param"):
        """Валидация параметра и приведение к массиву (если передано одно число — копируем на все вершины)"""
        if isinstance(param, (int, float)):  
            param = np.full(num_nodes, param)  # Если одно число → копируем для всех вершин
        param = np.array(param)
        
        if len(param) != num_nodes:
            raise ValueError(f"{name} должен быть длины {num_nodes}, но получено {len(param)}")

        if min_value is not None and np.any(param < min_value):
            raise ValueError(f"{name} не может быть меньше {min_value}")

        if max_value is not None and np.any(param > max_value):
            raise ValueError(f"{name} не может быть больше {max_value}")

        return param

    def name(self):
        """Возвращает название с округлёнными средними параметрами"""
        return (f"LinearAgent(β_cost={self.beta_cost.mean():.2f}, "
                f"β_cap={self.beta_capacity.mean():.2f}, "
                f"stay={self.stay_prob.mean():.2f})")

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Численно устойчивый softmax
        return exp_x / exp_x.sum()

    def get_policy(self, graph: SkyGraph):
        transition_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            neighbors = list(graph.graph.successors(i))

            if neighbors:
                # Учитываем уникальные параметры для вершины i
                costs = np.array([graph.cost_matrix[i][j] for j in neighbors])
                capacities = np.array([graph.capacity_matrix[i][j] for j in neighbors])

                value = -self.beta_cost[i] * costs + self.beta_capacity[i] * capacities  # Чем больше value, тем лучше
                probabilities = self.softmax(value)  # Применяем softmax

                # Распределяем вероятности: часть уходит на stay_prob[i], остальное на соседей
                transition_matrix[i, neighbors] = (1 - self.stay_prob[i]) * probabilities
                transition_matrix[i, i] = self.stay_prob[i]  # Петля

        return transition_matrix