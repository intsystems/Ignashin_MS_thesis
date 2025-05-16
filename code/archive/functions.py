import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def dummy_compute_traffic_jams(P, pi, dt=1):
    """Тупо Вычисляет суммарные пробки за момент dt."""
    num_states = P.shape[0]
    incoming_flow = np.zeros(num_states)
    outgoing_flow = np.zeros(num_states)
    
    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                flow = pi[i] * P[i, j] * dt
                incoming_flow[j] += flow
                outgoing_flow[i] += flow
    
    traffic_jams = incoming_flow - outgoing_flow
    return traffic_jams

def compute_traffic_jams(P, pi, dt=1):
    """Вычисляет суммарные пробки за момент dt. матрчное умножение"""
    return pi@P - pi

def find_stationary_distribution(P):
    """Находит стационарное распределение для матрицы переходов P."""
    num_states = P.shape[0]
    A = np.vstack([P.T - np.eye(num_states), np.ones(num_states)])
    b = np.append(np.zeros(num_states), 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]  # Решение системы уравнений
    return pi

