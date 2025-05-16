import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_markov_chain():
    """Создаёт вероятностную матрицу переходов с петлями."""
    P = np.array([[0.4, 0.3, 0.3],
                  [0.2, 0.5, 0.3],
                  [0.3, 0.3, 0.4]])  # Каждая строка суммируется в 1
    return P

def visualize_markov_chain(P):
    """Визуализирует граф Марковской цепи с раздельными рёбрами для двунаправленных переходов."""
    G = nx.DiGraph()
    num_states = P.shape[0]
    
    for i in range(num_states):
        for j in range(num_states):
            if P[i, j] > 0:  # Добавляем только ненулевые переходы
                G.add_edge(i, j, weight=P[i, j])
    
    pos = nx.circular_layout(G)  # Расположение узлов в круге
    labels = {i: f"S{i}" for i in range(num_states)}
    
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=2000, edge_color='gray')
    
    edge_labels = {}
    for i, j in G.edges():
        edge_labels[(i, j)] = f"{P[i, j]:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.8, font_size=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
    
    plt.title("Марковская цепь")
    plt.show()



def arendt_sampling(P, initial_dist, steps=10):
    """Выполняет сэмплирование состояний по апостериорному распределению."""
    num_states = P.shape[0]
    states = np.arange(num_states)
    current_state = np.random.choice(states, p=initial_dist)
    
    history = [current_state]
    
    for _ in range(steps - 1):
        current_state = np.random.choice(states, p=P[current_state])
        history.append(current_state)
    
    return history
