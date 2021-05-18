# Author: Aris Christoforidis.
from enums import SaveMode
from nas_state import NasState
from module_manager import ModuleManager
from config import BEST_NETWORK_SCORE_LABEL, DATASET, DELETE_NETWORKS_EVERY, GENERATIONS, INVALID_NETWORK_FITNESS, INVALID_NETWORK_TIME, NETWORK_REMAIN_PERCENTAGE, POPULATION_SIZE, TRAINING_EPOCHS, UNEVALUATED_FITNESS
from neural_module import NeuralModule
from evaluation import Evaluator, FashionMnistEvaluator, NasBenchEvaluator
from communication import Communicator
from performance_supervisor import PerformanceSupervisor
from math import log10
import matplotlib.pyplot as plt

LOAD_FROM_CHECKPOINT = False

def main():
    # Establish communication.
    communicator = Communicator()
    problem_name =   "fashion-mnist" # "activity_recognition" 
    # Create new nas state(saving).
    if LOAD_FROM_CHECKPOINT == False:
        # Initialize evaluator.
        # Switch to this to use nasbench.
        #evaluator = NasBenchEvaluator()
        # Switch to this to use activity recognition.
        #evaluator = Evaluator()
        # Switch to this to use fashion-mnist.
        evaluator = FashionMnistEvaluator()
        # Initialize module manager.
        manager = ModuleManager(evaluator)
        # Make initial population.
        population = [NeuralModule(None, manager) for _ in range(POPULATION_SIZE)]
        starting_generation = 0
        performance_supervisor = PerformanceSupervisor(problem_name, starting_generation)
        state = NasState(name=problem_name, evaluator=evaluator, performance_supervisor=performance_supervisor)
    else:
        # Load the latest checkpoint.
        state = NasState.load(name=problem_name, mode=SaveMode.PICKLE)
        evaluator = state.evaluator
        manager = state.module_manager
        population = state.population
        starting_generation = state.generation
        performance_supervisor = state.performance_supervisor

    networks_to_create = POPULATION_SIZE - len(population)
    
    for generation in range(starting_generation, GENERATIONS):
        print("="*128)
        print(f"Generation {generation}")
        generation_training_time = 0

        # Inform module manager for generation change.
        manager.on_generation_increase()
        
        # Create new random modules.
        print(f"Population size:{len(population)}. Creating {networks_to_create} additional networks.")
        population.extend([NeuralModule(None, manager) for _ in range(networks_to_create)])

        # NOTE: Some modules may not be able to be evaluated, due to a pytorch issue.
        # We replace these with new modules.
        invalid_modules = []
        # Evaluate those that where mutated.
        for idx, module in enumerate(population):
            module.mutate()
            if module.fitness == UNEVALUATED_FITNESS:
                # Increase the training time as complexity goes up. NOTE: Not used.
                module_properties = module.get_module_properties()
                training_time = max(1, min(int(module_properties.complexity / max(log10(generation+1), 1)), TRAINING_EPOCHS))
                print("-" * 128)
                print(f"Training for {training_time} epochs")
                accuracy, time = evaluator.evaluate(module,evaluation_epochs=training_time)
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

        # Remove invalid modules.
        for module in invalid_modules:
            population.remove(module)

        # Eliminate weaker modules.
        networks_to_create = 0
        if generation % DELETE_NETWORKS_EVERY == 0:
            # Sort based on fitness.
            population.sort(key=lambda x: -x.fitness)
            # Calculate how many networks to keep and throw away the rest.
            networks_to_keep = int(NETWORK_REMAIN_PERCENTAGE * POPULATION_SIZE)
            networks_to_create = POPULATION_SIZE - networks_to_keep
            population = population[:networks_to_keep]

            # Logging.
            fitness_threshold = population[networks_to_keep-1].fitness
            print(f"Replacing {networks_to_create} networks. Fitness threshold: {fitness_threshold}")

        
        # Save best network if it changed.
        if manager.best_module_updated == True:
            manager.on_best_module_updated(evaluate=True)
            performance_supervisor.on_best_global_fitness_found(generation, manager.best_network_data)
        
        # Save checkpoint.
        state.save(generation=generation, population=population, module_manager=manager, mode=SaveMode.PICKLE)

        # Save metrics and plot.
        performance_supervisor.on_generation_increase(population)



    print(f"Size: {communicator._get_size()}")
    return

if __name__ == "__main__":
    main()