# Author: Aris Christoforidis.

from config import BEST_NETWORK_DATA_SAVE_BASE_PATH, BEST_NETWORK_SCORE_LABEL, BEST_NETWORK_PROPERTIES_LABEL , MAX_NOTABLE_MODULES, MIN_PROPERTIES_OBS_COUNT
from enums import ModuleType
from module_properties import ModuleProperties, PropertiesInfo, TempPropertiesInfo
from evaluation import Evaluator
import random as rnd
import pickle as pkl

class ModuleManager:
    """
    The module manager keeps track of notable modules and scores them. It provides
    methods for picking a notable module based on their weighted score. 
    """

    def __init__(self, evaluator : Evaluator):
        # Create module properties objects for starting layers.
        self._notable_modules = {ModuleProperties(ModuleType.NEURAL_LAYER, layer, None, [], None, 1, 0) : PropertiesInfo() for layer in evaluator.get_available_layers()}
        self._candidate_modules = {}
        self.best_network_data = {BEST_NETWORK_SCORE_LABEL : -1, BEST_NETWORK_PROPERTIES_LABEL : None }

    def get_random_notable_modules(self,count=1, restrict_to=None):
        """
        Get a random notable module, weighted using the fitness scores.

        Parameters
        ----------
        count: int
            The number of modules to get.

        restrict_to: ModuleType
            Leave as is if you want to use all layers. Set it to a module type
            to get a layer of that type.

        Returns
        -------
        notable_modules: list(ModuleProperties)
            The notable modules list.
        """
        if restrict_to == None:
            # All modules considered.
            candidates = list(self._notable_modules.keys())
            w = [info.get_total_fitness() for info in [self._notable_modules[module] for module in candidates]]
        else:
            # Restricted to a type. May return an empty list.
            candidates = [module for module in self._notable_modules.keys() if module.module_type == restrict_to]
            w = [self._notable_modules[module].get_total_fitness() for module in candidates]
        
        # If there are no numbers in here, divide probability evenly.
        if sum(w) <= 0: w = [1/len(w) for _ in w]
        
        notable_modules = rnd.choices(population=candidates, weights=w,k=count)
        return notable_modules
    
    def record_module_properties(self, neural_module):
        """
        Receive a (top level) neural module and evaluate whether it (or a child
        module) should be added to the notable_modules list.

        Parameters
        ----------
        neural_module: NeuralModule
            A NeuralModule object.
        """
        # Initialize the list by adding the properties of the root module.
        properties_list = [neural_module.get_module_properties()]

        while len(properties_list) > 0:
            # Pop the first element of the list for evaluation.
            properties = properties_list.pop(0)
            
            if properties.module_type == ModuleType.ABSTRACT_MODULE:
                # Get this module's children properties and add them to the properties_list.
                # This will ensure that all child modules will be examined.
                properties_list.extend(properties.child_module_properties)

            # NOTE: Maybe do a check for UNEVALUATED_FITNESS here?
            #  Probably not necessary.
            fitness = properties.module_fitness

            # If this properties object describes a "notable" module 
            if properties in self._notable_modules:
                self._notable_modules[properties].record(fitness)
            else:
                # If this is a new properties module, record it.
                if properties not in self._candidate_modules:
                    # More complex graphs modules will have a longer TTL.
                    complexity = properties.total_nodes + properties.total_edges
                    self._candidate_modules[properties] = TempPropertiesInfo(complexity)
                # Record fitness.
                self._candidate_modules[properties].record(fitness)

        # Compare it with the best module.
        self.compare_with_best_module(neural_module)

    def on_generation_increase(self):
        """ Call this when a generation changes. """

        # Update TTL for candidate modules and delete expired ones.
        for module_properties in self._candidate_modules:
            should_delete = self._candidate_modules[module_properties].on_generation_increase()
            if should_delete == True: self._candidate_modules.pop(module_properties)

        # Get the info list and sort the list based on the average fitness 
        # of the notable modules.
        info_list = list(self._notable_modules.values())
        info_list.sort(key=lambda x: -x.average_fitness)
        # Get the minimum fitness value.
        min_fitness_threshold = info_list[-1].average_fitness
        
        # Candidate module list.
        for module_properties, temp_info in self._candidate_modules.items():
            # If this module has not been observed enough times, don't consider it.
            if temp_info.occurence_count < MIN_PROPERTIES_OBS_COUNT: continue
            # If the candidate module has a fitness higher that the minimum on
            # the notable modules list, add it to the notable modules.
            if temp_info.average_fitness > min_fitness_threshold:
                info = PropertiesInfo(temp_info)
                self._notable_modules[module_properties] = info

        # Notable module list.
        if len(self._notable_modules) > MAX_NOTABLE_MODULES:
            # Find the cutoff element and its value.
            sorted_notable_modules = sorted(self._notable_modules.values(),key=lambda x: -x.average_fitness)
            min_fitness_value = sorted_notable_modules[MAX_NOTABLE_MODULES].average_fitness
            
            # Iterate through the dictionary and remove any entries below the
            # cutoff fitness value.
            marked_for_deletion = []
            for module_properties, info in self._notable_modules.items():
                # Careful with the core layers.
                # NOTE: I don't think that this is necessary, but its better to be safe.
                # In fact this may be removed in the future with the logic being: if
                # at some point the core layers are removed, then the mutation will 
                # happen with just the abstracted nodes, which would contain core layers.
                if module_properties.module_type == ModuleType.NEURAL_LAYER: continue
                # Compare for potential removal.
                if info.average_fitness < min_fitness_value:
                    marked_for_deletion.append(module_properties)
            
            # Delete marked properties.
            for properties in marked_for_deletion:
                self._notable_modules.pop(properties)


    def compare_with_best_module(self, neural_module, verbose=True):
        """
        Compares a (top level) module with the global best found thus far. If the
        new module is better, the best one is replaced.

        Parameters
        ----------
        neural_module: NeuralModule
            A NeuralModule object.

        verbose: bool
            Whether or not to print diagnostics.
        """

        if neural_module.fitness > self.best_network_data[BEST_NETWORK_SCORE_LABEL]:
            self.best_network_data[BEST_NETWORK_PROPERTIES_LABEL] = neural_module.get_module_properties()
            self.best_network_data[BEST_NETWORK_SCORE_LABEL] = neural_module.fitness
            if verbose == True:
                print(f"A new best network was found with a fitness value of {neural_module.fitness:.3f}")


    def save_best_module(self):
        """ Saves the best module data in a pickle file. """
        save_path = f"{BEST_NETWORK_DATA_SAVE_BASE_PATH}/activity_recognition_best.pkl" 
        with open(save_path,'wb') as save_file:
            pkl.dump(self.best_network_data, save_file)

            