import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class SkyGraph:
    def __init__(self, num_nodes):
        self.graph = nx.DiGraph()
        self.num_nodes = num_nodes
        self.capacity_matrix = np.full((num_nodes, num_nodes), np.inf, dtype=float)
        self.cost_matrix = np.full((num_nodes, num_nodes), -1, dtype=float)
        self.distance_matrix = np.full((num_nodes, num_nodes), -1, dtype=float)
        self.population = np.zeros(num_nodes, dtype=int)
        
        for i in range(num_nodes):
            self.graph.add_node(i)
    
    def add_road(self, u, v, capacity, cost, distance):
        self.graph.add_edge(u, v, capacity=capacity, cost=cost, distance=distance)
        self.capacity_matrix[u][v] = capacity
        self.cost_matrix[u][v] = cost
        self.distance_matrix[u][v] = distance
    
    def remove_road(self, u, v):
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            self.capacity_matrix[u][v] = -1
            self.cost_matrix[u][v] = -1
            self.distance_matrix[u][v] = -1
    
    def remove_node(self, node):
        if self.graph.has_node(node):
            self.graph.remove_node(node)
            self.population = np.delete(self.population, node)
            self.capacity_matrix = np.delete(np.delete(self.capacity_matrix, node, axis=0), node, axis=1)
            self.cost_matrix = np.delete(np.delete(self.cost_matrix, node, axis=0), node, axis=1)
            self.distance_matrix = np.delete(np.delete(self.distance_matrix, node, axis=0), node, axis=1)
            self.num_nodes -= 1
    
    def update_road(self, u, v, capacity=None, cost=None, distance=None):
        if self.graph.has_edge(u, v):
            if capacity is not None:
                self.graph[u][v]['capacity'] = capacity
                self.capacity_matrix[u][v] = capacity
            if cost is not None:
                self.graph[u][v]['cost'] = cost
                self.cost_matrix[u][v] = cost
            if distance is not None:
                self.graph[u][v]['distance'] = distance
                self.distance_matrix[u][v] = distance
    
    def set_population(self, node, count):
        self.population[node] = count
    
    def visualize(self, use_real_distances=False):
        if use_real_distances:
            pos = {i: (i, sum(self.distance_matrix[i]) if sum(self.distance_matrix[i]) >= 0 else i) for i in range(self.num_nodes)}
        else:
            pos = nx.spring_layout(self.graph)
        
        plt.figure(figsize=(10, 6))
        
        # Рисуем вершины
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=150, edge_color='gray')

        # Обрабатываем возможное наложение меток ребер
        edge_labels = {}
        edge_offsets = {}
        for u, v, d in self.graph.edges(data=True):
            label = f"C:{d['capacity']}, Cost:{d['cost']}, D:{d['distance']}"
            if (v, u) in edge_labels:  # Если есть обратное ребро, сдвигаем метки
                edge_offsets[(u, v)] = 0.15
                edge_offsets[(v, u)] = -0.15
            else:
                edge_offsets[(u, v)] = 0.0
            edge_labels[(u, v)] = label
        
        for (u, v), label in edge_labels.items():
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels={(u, v): label}, label_pos=0.3 + edge_offsets[(u, v)], font_size=10, font_color='red')
        
        # Отображаем вершины в виде табличек
        for node, (x, y) in pos.items():
            plt.text(x, y, f'V:{node}\nP:{self.population[node]}', fontsize=12, ha='center', bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
        
        plt.show()

    def _default(self, mul = 1):
        
        self.add_road(0, 1, 20 * mul , 4, 10)
        self.add_road(1, 0, 20 * mul, 2, 10)

        self.add_road(1, 2, 30 * mul, 10, 7 )
        self.add_road(2, 1, 30 * mul, 5, 7)
        
        self.add_road(2, 3, 80 * mul, 15, 8)
        self.add_road(3, 2, 80 * mul, 7, 8 )
        
        self.add_road(3, 4, 50 * mul, 23, 5)
        self.add_road(4, 3, 50 * mul, 10, 5)
        
        self.add_road(3, 1, 10 * mul, 7, 10)
        
        self.add_road(4, 2, 10 * mul, 10, 10)

        self.set_population(0, 100)
        self.set_population(1, 200)
        self.set_population(2, 150)
    
