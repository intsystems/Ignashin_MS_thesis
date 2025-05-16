import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from .graph import SkyGraph
from .agents.agent import SkyAgent

class Logger:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.dict = dict()

    def update(self, name: str, value):
        if name not in self.dict:
            self.dict[name] = [value]
        else:
            self.dict[name].append(value)

LOGGER = Logger()

class SkyModel:
    def __init__(self, graph: SkyGraph, agent: SkyAgent):
        self.graph = graph
        self.agent = agent
    
    def run_simulation(self, T):
        """
        1) Матрица политики (вероятностная) * вектор популяции = матрица политики численная
        """
        total_congested_people = 0
        
        LOGGER.reset()
        for t in range(T):
            policy = self.agent.get_policy(self.graph)

            desired_flow = np.multiply(policy, self.graph.population[:, np.newaxis])
            
            excess_flow = np.maximum(desired_flow - self.graph.capacity_matrix, 0)
            
            actual_flow = np.maximum(desired_flow - excess_flow, 0) + np.diag(excess_flow.sum(axis=-1))

            self.graph.population -= actual_flow.sum(axis=1).astype(int)
            self.graph.population += actual_flow.sum(axis=0).astype(int)
            
            congested_people = np.sum(excess_flow)
            total_congested_people += congested_people

            LOGGER.update(f'{self.agent.name()} policy', deepcopy(policy))
            LOGGER.update(f'{self.agent.name()} desired_flow', deepcopy(desired_flow))
            LOGGER.update(f'{self.agent.name()}excess_flow', deepcopy(excess_flow))
            LOGGER.update(f'{self.agent.name()}actual_flow', deepcopy(actual_flow))
            LOGGER.update(f'{self.agent.name()}people', deepcopy(self.graph.population.copy()))
        
        return total_congested_people / T