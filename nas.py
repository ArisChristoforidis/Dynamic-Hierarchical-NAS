# Author: Aris Christoforidis.

from neural_module import NeuralModule
from evaluation import Evaluator, NasBenchEvaluator
from communication import Communicator
from mpi4py import MPI

import networkx as nx
import matplotlib.pyplot as plt

def main():
    # Establish communication.
    communicator = Communicator()
    
    # Switch to this to use nasbench.
    evaluator = NasBenchEvaluator()
    #evaluator = Evaluator()
    neural_module = NeuralModule(1,evaluator)
    # Do some example mutations.
    for _ in range(5):
        acc, time = evaluator.evaluate(neural_module)
        print(f"Accuracy: {acc:.2f} | Time: {time:.2f} sec")
        full_graph, layer_names, _, _ = neural_module.get_graph()
        # Draw graph.
        nx.draw_spring(full_graph,with_labels=True,labels=layer_names)
        plt.show()
        # Mutate.
        neural_module.mutate()
    print(f"Size: {communicator._get_size()}")
    return

if __name__ == "__main__":
    main()