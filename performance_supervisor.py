import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from config import BEST_NETWORK_SCORE_LABEL, EVALUATED_SCORE_LABEL, PLOT_SAVE_BASE_PATH

class PerformanceSupervisor:

    def __init__(self, problem_name, starting_generation = 0):
        self.name = problem_name
        self.generation = starting_generation
        self.average_fitness = []
        self.best_fitness = []
        self.best_networks = {}
        self.global_best_fitness = []

    def on_generation_increase(self, population):

        sum_fitness = sum([p.fitness for p in population])
        generation_average_fitness = sum_fitness / len(population)
        generation_best_fitness = max([p.fitness for p in population])

        self.average_fitness.append(generation_average_fitness)
        self.best_fitness.append(generation_best_fitness)

        # Increase generation(NOTE: Before plotting)
        self.generation += 1
        
        self.plot()
    
    def on_best_global_fitness_found(self, generation, best_network_data):
        self.best_networks[generation] = best_network_data

    def plot(self):
        x = list(range(len(self.average_fitness)))
        try:
            # Plot average fitness.
            plt.plot(x,self.average_fitness,color='#FF4500')        
            # Plot generation best.
            plt.plot(x,self.best_fitness,color='#18A558')
            
            # Plot evaluated best.
            x = sorted(list(self.best_networks.keys()))
            best_evaluated_fitness_scores = [self.best_networks[g][EVALUATED_SCORE_LABEL] for g in x]
            plt.plot(x, best_evaluated_fitness_scores, color="#0074D9")

            print(f"Best unevaluated net value: {max(self.best_fitness):.3f}%")
            print(f"Best evaluated net value: {max([net_data[EVALUATED_SCORE_LABEL] for net_data in self.best_networks.values()]):.3f}%")

            save_path = f"{PLOT_SAVE_BASE_PATH}/{self.name}.png"
            plt.savefig(save_path)
        except Exception:
            print("Performance could not be plotted at this point.")
            return