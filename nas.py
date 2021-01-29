# Author: Aris Christoforidis.
from enums import SaveMode
from nas_state import NasState
from module_manager import ModuleManager
from config import BEST_NETWORK_SCORE_LABEL, DATASET, DELETE_NETWORKS_EVERY, GENERATIONS, INVALID_NETWORK_FITNESS, INVALID_NETWORK_TIME, NETWORK_REMAIN_PERCENTAGE, POPULATION_SIZE, TRAINING_EPOCHS, UNEVALUATED_FITNESS
from neural_module import NeuralModule
from evaluation import Evaluator, NasBenchEvaluator
from communication import Communicator

import matplotlib.pyplot as plt

LOAD_FROM_CHECKPOINT = True

def main():
    # Establish communication.
    communicator = Communicator()
    
    # Create new nas state(saving).
    if LOAD_FROM_CHECKPOINT == False:
        # Initialize evaluator.
        # Switch to this to use nasbench.
        #evaluator = NasBenchEvaluator()
        evaluator = Evaluator()
        # Initialize module manager.
        manager = ModuleManager(evaluator)
        # Make initial population.
        population = [NeuralModule(None, manager) for _ in range(POPULATION_SIZE)]
        starting_generation = 0
        state = NasState(name=DATASET, evaluator=evaluator)
    else:
        # Load the latest checkpoint.
        state = NasState.load(name=DATASET, mode=SaveMode.PICKLE)
        evaluator = state.evaluator
        manager = state.module_manager
        population = state.population
        starting_generation = state.generation

    for generation in range(starting_generation, GENERATIONS):
        print(f"Generation {generation}")
        print("="*128)
        generation_training_time = 0

        # Inform module manager for generation change.
        manager.on_generation_increase()

        # NOTE: Some modules may not be able to be evaluated, due to a pytorch issue.
        # We replace these with new modules.
        invalid_modules = []
        # Evaluate those that where mutated.
        for idx, module in enumerate(population):
            module.mutate()
            if module.fitness == UNEVALUATED_FITNESS:
                # Slowly increase the training time.
                accuracy, time = evaluator.evaluate(module,evaluation_epochs=TRAINING_EPOCHS + generation)
                # If the network can't be evaluated, add a new one, and mark it
                # for deletion.
                if accuracy == INVALID_NETWORK_FITNESS and time == INVALID_NETWORK_TIME:
                    print(f"Network {idx+1} could not be evaluated, replacing.")
                    # Add another neural module to the population.(Evaluated in 
                    # this generation)
                    population.append(NeuralModule(None, manager))
                    invalid_modules.append(module)
                    continue
                # If the network is ok, proceed with the algorithm.
                generation_training_time += time
                # module.show_abstract_graph()
                # module.show_full_graph()
                module.set_fitness(accuracy)
                print(f"Neural module {idx+1}: {accuracy:.3f}")

        print(f"Time elapsed: {generation_training_time /60:.2f} minutes")
        
        # Print best module fitness.
        best_module = max(population, key=lambda x: x.fitness)
        print(f"Generation best module fitness: {best_module.fitness:.3f}")
        print(f"Global best module fitness: {manager.best_network_data[BEST_NETWORK_SCORE_LABEL]:.3f}")

        # Remove invalid modules.
        for module in invalid_modules:
            population.remove(module)

        print(f"Candidate module count: {len(manager._candidate_modules)}")
        print(f"Notable module count: {len(manager._notable_modules)}")

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
    
        # Save best network if it changed.
        if manager.best_module_updated == True:
            manager.on_best_module_updated()
        
        # Save checkpoint.
        state.save(generation=generation, population=population, module_manager=manager, mode=SaveMode.PICKLE)
    
    print(f"Size: {communicator._get_size()}")
    return

if __name__ == "__main__":
    main()