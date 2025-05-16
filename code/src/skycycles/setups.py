import numpy as np
from .model import CyclesGraph  # Замени на путь до твоего класса


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_graphs_by_attractiveness_grid(start=0, stop=20, step=2):
    graphs = []
    attractiveness_grid = np.arange(start, stop + step, step)

    for a_val in attractiveness_grid:
        graph = CyclesGraph()
        for i in range(2):
            graph.add_cycle(
                cycle_info=i,
                people_count=0,  # Не используется
                free_travel_time=5.0,
                attractiveness= a_val * ( 1 + 9*i )  # Десятикратная разница в привлекательности с шкалированием a_val
            )
            graph.add_lift(
                lift_info=i,
                max_flow=0.5 - i * 0.45
            )
            graph.set_lift_cycle_connection(
                cycle_idx=i,
                lift_idx=i
            )
        graphs.append(graph)

    return graphs


def generate_gigagrafs_by_attractiveness_grid(start=0, stop=20, step=2):
    graphs = []
    attractiveness_grid = np.arange(start, stop + step, step)

    for a_val in attractiveness_grid:
        graph = CyclesGraph()
        for i in range(5):
            graph.add_cycle(
                cycle_info=i,
                people_count=0,  # Не используется
                free_travel_time=1.2 - 1.0 * np.exp(-i),
                attractiveness= a_val * ( i + 1 )  
            )
            graph.add_lift(
                lift_info=i,
                max_flow=0.06 * (5 - i)
            )
            for j in range(i + 1):
                graph.set_lift_cycle_connection(
                    cycle_idx=i,
                    lift_idx=j
                )
        graph.add_lift(
            lift_info=10,
            max_flow=228
        )
        graphs.append(graph)

    return graphs


def visualize_graphs(graphs, figsize=(2, 2), cols=5):
    """
    Функция для визуализации списка графов в несколько колонок.
    
    Parameters:
    - graphs: список графов, которые нужно визуализировать
    - figsize: размер каждого графика (ширина, высота)
    - cols: количество колонок в каждой строке
    """
    # Количество графов
    n_graphs = len(graphs)
    # Количество строк для размещения графов
    rows = (n_graphs + cols - 1) // cols  # округление вверх

    # Создание фигуры с подграфиками
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize[0], rows * figsize[1]))
    axes = axes.flatten()  # Преобразуем массив подграфиков в одномерный список для удобства

    # Отображение графов
    for i, g in enumerate(graphs):
        ax = axes[i]
        ax.set_title(f"Graph {i}", fontsize=6)
        g.visualize_graph(ax=ax, figsize=figsize, font_size=6)

    # Убираем лишние подграфики (если графов меньше, чем мест)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Подгоняем расположение подграфиков
    plt.tight_layout()
    plt.show()

def run_experiments(graphs, algorithm_fn, N_samples=100, max_iters=100, algorithm_kwargs=None):
    if algorithm_kwargs is None:
        algorithm_kwargs = {}

    all_results = {
        'n values': [],
        'FW gaps': [],
        'jams': [],
        'labels': [],
    }

    for idx, graph in enumerate(graphs):
        n_samples, n_gaps, n_qualities = [], [], []

        for _ in tqdm(range(N_samples), desc=f'Graph {idx}'):
            result = algorithm_fn(graph, max_iters=max_iters, **algorithm_kwargs)
            n, FW_gaps, qualities, n_values, t_array, f_array = result

            n_samples.append(n_values)
            n_qualities.append(qualities)
            n_gaps.append(FW_gaps)

        # Aggregate per graph
        arrays = {
            'n values': np.array(n_samples),
            'FW gaps': np.array(n_gaps),
            'jams': np.array(n_qualities),
        }

        label = f'Graph {idx}'  # Simplified label
        for key in ['n values', 'FW gaps', 'jams']:
            mean = arrays[key].mean(axis=0)  # shape: (iters, dim) or (iters,)
            std = arrays[key].std(axis=0)

            all_results[key].append((mean, std))
        all_results['labels'].append(label)

    # Visualizing 'n values' for each graph in 5 columns (horizontally)
    key = 'n values'
    n_plots = len(all_results[key])
    cols = 5  # Number of columns for the subplots
    rows = (n_plots + cols - 1) // cols  # Calculate number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes = axes.flatten()

    for i, (mean, std) in enumerate(all_results[key]):
        ax = axes[i]
        ax.set_title(all_results['labels'][i], fontsize=8)
        iters = np.arange(mean.shape[0])

        if mean.ndim == 1:
            ax.plot(iters, mean, label='n')
            ax.fill_between(iters, mean - std, mean + std, alpha=0.3)
        else:
            for dim in range(mean.shape[1]):
                ax.plot(iters, mean[:, dim], label=f'n[{dim}]')
                ax.fill_between(iters, mean[:, dim] - std[:, dim], mean[:, dim] + std[:, dim], alpha=0.2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('n value')
        ax.grid(True)
        ax.legend(fontsize=6)

    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Plotting FW gaps in log-log scale
    plt.figure(figsize=(6, 4))
    for (mean, std), label in zip(all_results['FW gaps'], all_results['labels']):
        iters = np.arange(mean.shape[0])

        mean = mean
        std = std

        plt.loglog(iters, mean, label=label)
        plt.fill_between(iters, mean, mean + std, alpha=0.3)

    plt.title('Gaps by iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Gap')
    plt.grid(True)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

        # Visualizing 'jams' for each graph in 5 columns (horizontally)
    key = 'jams'
    n_plots = len(all_results[key])
    cols = 5
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    axes = axes.flatten()

    for i, (mean, std) in enumerate(all_results[key]):
        ax = axes[i]
        ax.set_title(all_results['labels'][i], fontsize=8)
        iters = np.arange(mean.shape[0])

        if mean.ndim == 1:
            ax.plot(iters, mean, label='jams')
            ax.fill_between(iters, mean - std, mean + std, alpha=0.3)
        else:
            for dim in range(mean.shape[1]):
                ax.plot(iters, mean[:, dim], label=f'jams[{dim}]')
                ax.fill_between(iters, mean[:, dim] - std[:, dim], mean[:, dim] + std[:, dim], alpha=0.2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel("tau(n)")
        ax.grid(True)
        ax.legend(fontsize=6)

    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
