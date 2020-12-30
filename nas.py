# Author: Aris Christoforidis.
from module_manager import ModuleManager
from config import DELETE_NETWORKS_EVERY, GENERATIONS, NETWORK_REMAIN_PERCENTAGE, POPULATION_SIZE, UNEVALUATED_FITNESS
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
    #evaluator = NasBenchEvaluator()
    evaluator = Evaluator()

    # Initialize module manager.
    manager = ModuleManager(evaluator)

    # Make initial population.
    population = [NeuralModule(None, manager) for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        print(f"Generation {generation+1}")
        generation_training_time = 0

        # Inform module manager for generation change.
        manager.on_generation_increase()

        # Try mutate neural modules.
        for module in population:
            module.mutate()

        # Evaluate those that where mutated.
        for idx, module in enumerate(population):
            if module.fitness == UNEVALUATED_FITNESS:
                accuracy, time = evaluator.evaluate(module)
                generation_training_time += time
                # module.show_abstract_graph()
                # module.show_full_graph()
                module.set_fitness(accuracy)
                print(f"Neural module {idx+1}: {accuracy:.3f}")
        print(f"Time elapsed: {generation_training_time:.2f} seconds")
        print("="*50)


        # Eliminate weaker modules.
        if generation % DELETE_NETWORKS_EVERY == 0:
            # Sort based on fitness.
            population.sort(key=lambda x: -x.fitness)
            # Calculate how many networks to keep and throw away the rest.
            networks_to_keep = int(NETWORK_REMAIN_PERCENTAGE * POPULATION_SIZE)
            population = population[:networks_to_keep]

            # Create new random modules.
            networks_to_create = POPULATION_SIZE - networks_to_keep
            population.extend([NeuralModule(None, manager) for _ in range(networks_to_create)])

            # Logging.
            fitness_threshold = population[networks_to_keep-1].fitness
            print(f"Replacing {networks_to_create} networks. Fitness threshold: {fitness_threshold}")

        continue

    print(f"Size: {communicator._get_size()}")
    return

if __name__ == "__main__":
    main()