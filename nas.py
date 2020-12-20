# Author: Aris Christoforidis.
from module_manager import ModuleManager
from config import GENERATIONS, POPULATION_SIZE, UNEVALUATED_FITNESS
from neural_module import NeuralModule
from evaluation import Evaluator, NasBenchEvaluator
from communication import Communicator
from mpi4py import MPI

import networkx as nx
import matplotlib.pyplot as plt

def main():
    # Establish communication.
    communicator = Communicator()
    
    # Initialize evaluator.
    # Switch to this to use nasbench.
    evaluator = NasBenchEvaluator()
    #evaluator = Evaluator()

    # Initialize module manager.
    manager = ModuleManager(evaluator)

    # Make initial population.
    population = [NeuralModule(None, manager) for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        print(f"Generation {generation+1}")

        # Inform module manager for generation change.
        manager.on_generation_increase()

        # Try mutate neural modules.
        for module in population:
            module.mutate()

        # Evaluate those that where mutated.
        for module in population:
            if module.fitness == UNEVALUATED_FITNESS:
                accuracy, time = evaluator.evaluate(module)
                module.set_fitness(accuracy)


        # TODO: Eliminate weaker modules. TBD

        # TODO: Create new random modules. TBD

        continue

    """
    # Do some example mutations.
    for _ in range(5):
        # Evaluate.
        accuracy, time = evaluator.evaluate(neural_module)
        neural_module.set_fitness(accuracy)
        print(f"Accuracy: {accuracy:.2f} | Time: {time:.2f} sec")
        
        # Draw graph.
        full_graph, layer_names, _, _ = neural_module.get_graph()
        nx.draw_spring(full_graph,with_labels=True,labels=layer_names)
        plt.show()
        
        # Mutate.
        neural_module.mutate()
    """
    print(f"Size: {communicator._get_size()}")
    return

if __name__ == "__main__":
    main()