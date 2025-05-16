import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from ..graph import SkyGraph


class SkyAgent:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        
    def name(self,):
        return "Uniform Agent"

    def get_policy(self, graph: SkyGraph):
        transition_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            neighbors = list(graph.graph.successors(i))
            neighbors.append(i)  # Добавляем саму вершину в список соседей
            if neighbors:
                prob = 1 / len(neighbors)
                for neighbor in neighbors:
                    transition_matrix[i][neighbor] = prob
        
        return transition_matrix