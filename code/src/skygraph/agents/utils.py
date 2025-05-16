import os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from src.skygraph import SkyGraph
from src.skygraph import SkyAgent, LinearSkyAgent
from src.skygraph import SkyModel
from src.skygraph.model import LOGGER


def test_agents(agents):
    
    muls = np.linspace(1, 5,100)
    
    for agent in agents:
        avg_congesteds = []
        for mul in muls:
            graph = SkyGraph(5)
            graph._default(mul = mul)
            model = SkyModel(graph, agent)
            avg_congested = model.run_simulation(10)
            avg_congesteds.append(avg_congested)
        plt.plot(muls, avg_congesteds, label = agent.name())
    
    plt.legend()
    plt.title('Средняя пробка от пропускной способности')
    plt.ylabel('средняя пробка за итерацию')
    plt.xlabel('Множитель пропускной способности')
    plt.show()