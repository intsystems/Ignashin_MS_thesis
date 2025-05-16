import numpy as np
from scipy.optimize import minimize
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from numpy.linalg import norm

def project_to_simplex(v):
    """Проекция вектора v на единичный симплекс."""
    v = np.asarray(v)
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / np.arange(1, n+1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


class CyclesGraph:
    """
    A class to represent a graph of cycles with lifts.

    Attributes
    ----------
    cycles : np.ndarray
        An array of dictionaries containing information about cycles.
    lifts : np.ndarray
        An array of dictionaries containing information about lifts.
    lift_cycle_matrix : np.ndarray
        A binary matrix indicating which lifts belong to which cycles.
    people_counts : np.ndarray
        An array storing the number of people in each cycle.
    free_travel_times : np.ndarray
        An array storing the free travel times for each cycle.
    max_lift_flow : np.ndarray
        An array storing the maximum flow capacity of each lift.
    attractiveness : np.ndarray
        An array storing the attractiveness value for each cycle.
    """
    
    def __init__(self):
        self.cycles = np.array([], dtype=object)
        self.lifts = np.array([], dtype=object)
        self.lift_cycle_matrix = np.zeros((0, 0), dtype=int)
        self.people_counts = np.array([], dtype=int)
        self.free_travel_times = np.array([], dtype=float)
        self.max_lift_flow = np.array([], dtype=int)
        self.attractiveness = np.array([], dtype=float)
    
    def add_cycle(self, cycle_info, people_count=0, free_travel_time=0, attractiveness=0.0):
        """
        Add a cycle to the graph.

        Parameters
        ----------
        cycle_info : dict
            Information about the cycle.
        people_count : int
            Number of people in the cycle.
        free_travel_time : float
            Free travel time for the cycle.
        attractiveness : float
            Attractiveness score for the cycle.
        """
        self.cycles = np.append(self.cycles, cycle_info)
        self.people_counts = np.append(self.people_counts, people_count)
        self.free_travel_times = np.append(self.free_travel_times, free_travel_time)
        self.attractiveness = np.append(self.attractiveness, attractiveness)
        
        if self.lifts.size > 0:
            new_row = np.zeros((1, self.lifts.size), dtype=int)
            self.lift_cycle_matrix = np.vstack([self.lift_cycle_matrix, new_row])
        else:
            self.lift_cycle_matrix = np.zeros((self.cycles.size, 0), dtype=int)
    
    def add_lift(self, lift_info, max_flow=0):
        """
        Add a lift to the graph.
        """
        self.lifts = np.append(self.lifts, lift_info)
        self.max_lift_flow = np.append(self.max_lift_flow, max_flow)
        
        if self.cycles.size > 0:
            new_col = np.zeros((self.cycles.size, 1), dtype=int)
            self.lift_cycle_matrix = np.hstack([self.lift_cycle_matrix, new_col])
        else:
            self.lift_cycle_matrix = np.zeros((0, self.lifts.size), dtype=int)
    
    def set_lift_cycle_connection(self, cycle_idx, lift_idx, connection=1):
        """
        Set the connection between a cycle and a lift.
        """
        if 0 <= cycle_idx < self.cycles.size and 0 <= lift_idx < self.lifts.size:
            self.lift_cycle_matrix[cycle_idx, lift_idx] = connection
        else:
            raise IndexError("Invalid cycle or lift index.")


    def solve_lift_system(self, Theta, T, b, n, objective_return=False, method="scipy"):
        """
        Solve the lift system using specified method.
        
        Parameters:
        Theta: Cycle-lift matrix: Theta[i, j] equals to 1 if and only if the lift j belongs to the cycle i
        T: Free travel times per cycle vector
        b: Maximum lift flows vector
        n: People counts per cycle vector
        objective_return: Whether to return objective value
        method: Solver method ("scipy", "gradient", or "cvxpy")
        
        Returns:
        Tuple of (t_opt, f_opt) or (t_opt, f_opt, objective) if objective_return=True
        """
        if method == "scipy":
            return self.solve_lift_system_scipy(Theta, T, b, n, objective_return)
        elif method == "gradient":
            return self.solve_lift_system_gradient(Theta, T, b, n, objective_return)
        elif method == "cvxpy":
            return self.solve_lift_system_cvxpy(Theta, T, b, n, objective_return)
        else:
            raise ValueError(f"Unknown solver method: {method}")
        
    def visualize_graph(self, ax=None, figsize=(12, 10), font_size=9):
        """
        Visualize the graph of cycles and lifts as a bipartite graph with attributes.

        Cycles are blue, lifts are red. Attributes are shown near each node.
        """
        G = nx.Graph()

        cycle_nodes = [f"C{i}" for i in range(len(self.cycles))]
        lift_nodes = [f"L{j}" for j in range(len(self.lifts))]

        # Add nodes
        G.add_nodes_from(cycle_nodes, bipartite=0)
        G.add_nodes_from(lift_nodes, bipartite=1)

        # Add edges
        for i, c_node in enumerate(cycle_nodes):
            for j, l_node in enumerate(lift_nodes):
                if self.lift_cycle_matrix[i, j]:
                    G.add_edge(c_node, l_node)

        # Layout: multipartite for bipartite style
        pos = nx.multipartite_layout(G, subset_key="bipartite")

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()  # Если ax не передан, используем текущий ось

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               nodelist=cycle_nodes,
                               node_color="skyblue",
                               node_size=1200,
                               label="Cycles", ax=ax)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=lift_nodes,
                               node_color="lightcoral",
                               node_size=1000,
                               label="Lifts", ax=ax)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", ax=ax)

        # Draw labels
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

        # Draw node attributes as annotations
        for i, c_node in enumerate(cycle_nodes):
            label = (
                f"n={self.people_counts[i]}\n"
                f"T={self.free_travel_times[i]:.2f}\n"
                f"a={self.attractiveness[i]:.2f}"
            )
            x, y = pos[c_node]
            ax.text(x, y - 0.08, label, fontsize=font_size,
                    ha='center', va='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", lw=1))

        for j, l_node in enumerate(lift_nodes):
            label = f"b={self.max_lift_flow[j]}"
            x, y = pos[l_node]
            ax.text(x, y - 0.08, label, fontsize=font_size,
                    ha='center', va='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=1))

        ax.set_title("Cycle-Lift Graph with Parameters", fontsize=14)
        ax.axis("off")
        
        return ax



    def solve_lift_system_cvxpy(self, Theta, T, b, n, objective_return=False):

        # print(Theta.shape ,T.shape ,b.shape , n.shape)

        import cvxpy as cp
        U = Theta.shape[0]
        f = cp.Variable(U)
        
        objective = cp.Minimize(
                T @ f - n @ cp.log(f)
        )
        constraints = [Theta.T @ f - b <= 0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        t_opt = constraints[0].dual_value
        f_opt = f.value
        
        t_opt = t_opt * (t_opt > 1e-7)
        
        if objective_return:
            return t_opt, f_opt, problem.value
        return t_opt, f_opt
    
    def solve_lift_system_scipy(self,Theta, T, b, n, objective_return = False):

        C, U = Theta.shape
        t_init = np.ones(U) / U
        def objective(t):
            denom = T + Theta @ t
            denom = denom + 1e-7 # Защита от деления на ноль
            f = n / denom  # Вычисление f
            return np.maximum(0, b - Theta.T @ f ) @ t + (np.maximum(Theta.T@f - b , 0)**2).sum()

    
        bounds = [(0, None)] * U  # t >= 0
        res = minimize(objective, t_init, bounds=bounds, method='SLSQP', tol = 1e-15 )
        
        t_opt = res.x
        denom = T + Theta @ t_opt  # Повторная защита от деления на ноль
        f_opt = n / denom
        if objective_return:
            return t_opt, f_opt, objective(t_opt)
        else:
            return t_opt, f_opt


        
    def solve_lift_system_gradient(self, Theta, T, b, n, objective_return=False, learning_rate=0.01, max_iter=1000, tol=1e-6):
        import torch
        from torch.optim import AdamW

        C, U = Theta.shape
        t_init = torch.ones(U, dtype=torch.float32) / U  # Инициализация t как тензор PyTorch

        # Отключаем градиенты для Theta и T, так как они не изменяются
        Theta = torch.tensor(Theta, dtype=torch.float32, requires_grad=False)
        T = torch.tensor(T, dtype=torch.float32, requires_grad=False)
        b = torch.tensor(b, dtype=torch.float32, requires_grad=False)
        n = torch.tensor(n, dtype=torch.float32, requires_grad=False)

        # Определяем целевую функцию
        def objective(t):
            denom = T + torch.matmul(Theta, t)  # Защита от деления на ноль
            denom = denom + 1e-8  # Маленькая константа для защиты от деления на ноль
            f = n / denom  # Вычисление f
            return torch.matmul(torch.maximum(torch.zeros_like(b), b - torch.matmul(Theta.T, f)), t) + (torch.sum(torch.maximum(torch.matmul(Theta.T, f) - b, torch.zeros_like(b))**2))

        t = t_init.clone().detach().requires_grad_(True)  # Инициализация t с требованием градиента

        # Инициализация оптимизатора AdamW
        optimizer = AdamW([t], lr=learning_rate)

        for i in range(max_iter):
            optimizer.zero_grad()  # Обнуляем градиенты перед каждым шагом
            loss = objective(t)  # Вычисление потерь
            # print(loss)

            loss.backward()  # Обратное распространение градиентов

            optimizer.step()  # Обновление t с помощью AdamW

            # Применяем ReLU для ограничения t >= 0
            with torch.no_grad():
                t.data = torch.relu(t.data)  # Обновляем t, гарантируя, что все значения >= 0

            # Проверка на сходимость
            if torch.norm(t.grad) < tol:
                break

        denom = T + torch.matmul(Theta, t)  # Повторная защита от деления на ноль
        f_opt = n / denom  # Вычисление f для оптимального t

        if objective_return:
            return t.detach().numpy(), f_opt.detach().numpy(), loss.item()  # Возвращаем значение потерь
        else:
            return t.detach().numpy(), f_opt.detach().numpy()

    def frank_wolfe(self, max_iters=100, tol=1e-6, method = 'scipy',tqdm_use = False):
        
        Theta, T, b = self.lift_cycle_matrix, self.free_travel_times, self.max_lift_flow
        
        C = Theta.shape[0]
        n = np.abs(np.random.normal(size = C))
        n = n / n.sum()
        FW_gaps = []
        n_values = []
        qualities = []
        t_array = []
        f_array = []

        iterator =  tqdm(range(1, max_iters)) if tqdm_use else range(1, max_iters)
        for k in iterator:
            t, f = self.solve_lift_system(Theta, T, b, n, method = method)

            # print(t, f)
            n_values.append(n)

            grad = Theta @ t - self.attractiveness

            idx_min = np.argmin(grad)
            idxes = np.where(grad == grad[idx_min])[0]
            idx_new_min = np.random.choice(idxes)
            n_new = np.zeros_like(n)
            n_new[idx_new_min] = 1

            FW_gap = np.dot(grad, n - n_new)
            FW_gaps.append(FW_gap)
            t_array.append(t)
            f_array.append(f)        
            qualities.append(grad)

            step_size = 2 / (k + 2)  # Стандартный шаг Frank-Wolfe
            n = (1 - step_size) * n + step_size * n_new
            
        return n , FW_gaps, qualities, n_values, t_array, f_array
    
    def tau_operator(self, Theta, a, t, T):
        return -a / (Theta @ t + T)

    def gap_function(self, n, tau_n):
        return np.dot(tau_n, n - project_to_simplex(n - tau_n))


    def extragradient(self, max_iters=100, gamma = 0.01, tol=1e-6, method = 'scipy',tqdm_use = False):
        Theta, T, b = self.lift_cycle_matrix, self.free_travel_times, self.max_lift_flow
        
        C = Theta.shape[0]
        n = np.abs(np.random.normal(size = C))
        n = n / n.sum()
        FW_gaps = []
        n_values = []
        qualities = []
        t_array = []
        f_array = []

        iterator =  tqdm(range(1, max_iters)) if tqdm_use else range(1, max_iters)
        for k in iterator:
            t, f = self.solve_lift_system(Theta, T, b, n, method = method)
            t_array.append(t)
            f_array.append(f)        

            a = self.attractiveness
            tau_n = self.tau_operator(Theta, a, t, T)
            
            n_values.append(n)
            FW_gaps.append(self.gap_function(n, tau_n))
            qualities.append(tau_n)
                           
            y = project_to_simplex(n - gamma * tau_n)
            tau_y = self.tau_operator(Theta, a, t, T)
            n_new = project_to_simplex(n - gamma * tau_y)

            n = n_new
            
        return n , FW_gaps, qualities, n_values, t_array, f_array