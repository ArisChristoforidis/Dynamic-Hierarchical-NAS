# Author: Aris Christoforidis.

from config import BEST_MODULE_EVALUATION_EPOCHS, BEST_NETWORK_DATA_SAVE_BASE_PATH, BEST_NETWORK_SCORE_LABEL, BEST_NETWORK_LABEL, EVALUATED_SCORE_LABEL, MAX_CANDIDATE_MODULES , MAX_NOTABLE_MODULES, MIN_PROPERTIES_OBS_COUNT, UNEVALUATED_FITNESS
from enums import ModuleType
from module_properties import ModuleProperties
from properties_info import PropertiesInfo, TempPropertiesInfo
from evaluation import Evaluator
import random as rnd
import pickle as pkl
from copy import deepcopy
from bisect import bisect_left, insort
import pandas as pd
from tabulate import tabulate

class ModuleManager:
    """
    The module manager keeps track of notable modules and scores them. It provides
    methods for picking a notable module based on their weighted score. 
    """

    def __init__(self, evaluator : Evaluator):
        # Create module properties objects for starting layers.
        self.evaluator = evaluator
        self._notable_modules = {ModuleProperties(ModuleType.NEURAL_LAYER, layer, None, [], 1, 0) : PropertiesInfo() for layer in evaluator.get_available_layers()}
        self._candidate_modules = {}
        self.expired_module_hashes = []
        self.best_module_updated = False
        self.best_network_data = {BEST_NETWORK_SCORE_LABEL : -1, EVALUATED_SCORE_LABEL: -1, BEST_NETWORK_LABEL : None }

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
            w = [self._notable_modules[module_property].average_fitness for module_property in candidates]
        else:
            # NOTE: Not used.
            # Restricted to a type. May return an empty list.
            candidates = [module_property for module_property in self._notable_modules.keys() if module_property.module_type == restrict_to]
            w = [self._notable_modules[module_property].get_total_fitness() for module_property in candidates]
        
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

        properties = neural_module.get_module_properties()
        fitness = neural_module.fitness
        # Random selection.
        #fitness = 1

        # TODO: Remove in final version, this is a debug check.
        assert fitness != UNEVALUATED_FITNESS, "Tried to record module with unevaluated fitness."
        
        # Check if this a module we have already seen and rejected.
        module_hash = hash(neural_module)
        expired_idx = bisect_left(self.expired_module_hashes, module_hash)
        if expired_idx != len(self.expired_module_hashes) and self.expired_module_hashes[expired_idx] == module_hash:
            # Don't bother with already examined & expired modules.
            return
        elif properties in self._notable_modules:
                self._notable_modules[properties].record(fitness)
        else:
            # If this is a new properties module, record it.
            if properties not in self._candidate_modules:
                print(f"Adding a new module to the candidate modules! ({properties.layer},{properties.module_type},{hash(properties)})")
                # More complex graphs modules will have a longer TTL.
                self._candidate_modules[properties] = TempPropertiesInfo(properties.complexity)
            # Record fitness.
            self._candidate_modules[properties].record(fitness)

        # If this is a root module, compare it with the best module.(We know for certain
        # that a non-root module will have less fitness)
        if neural_module.depth == 1:
            self.compare_with_best_module(neural_module)


    def on_generation_increase(self):
        """ Call this when a generation changes. """
        # Update TTL for candidate modules and delete expired ones.
        marked_for_deletion = []
        for module_properties in self._candidate_modules:
            should_delete = self._candidate_modules[module_properties].on_generation_increase()
            if should_delete == True: marked_for_deletion.append(module_properties)
        
        for module_properties in marked_for_deletion:
            # Add the hash to the expired module hash list.
            insort(self.expired_module_hashes, hash(module_properties))
            self._candidate_modules.pop(module_properties)

        # Get the info list and sort the list based on the average fitness 
        # of the notable modules.
        info_list = list(self._notable_modules.values())
        info_list.sort(key=lambda x: -x.average_fitness)
        # Get the minimum fitness value.
        min_fitness_threshold = info_list[-1].average_fitness
        print(f"Fitness threshold on notables:{min_fitness_threshold}")
        # Candidate module list.
        marked_for_deletion = []
        for module_properties, temp_info in self._candidate_modules.items():
            # If this module has not been observed enough times, don't consider it.
            if temp_info.occurence_count < MIN_PROPERTIES_OBS_COUNT: continue
            # If the candidate module has a fitness higher that the minimum on
            # the notable modules list, add it to the notable modules.
            if temp_info.average_fitness > min_fitness_threshold:
                print(f"Adding a new module to the notable modules! ({module_properties.layer},{module_properties.module_type},{hash(module_properties)})")
                info = PropertiesInfo(temp_info)
                self._notable_modules[module_properties] = info
                # Mark the module to be removed from the candidates.
                marked_for_deletion.append(module_properties)
        
        # Delete the properties from the notable modules.
        for module_properties in marked_for_deletion:
            insort(self.expired_module_hashes, hash(module_properties))
            self._candidate_modules.pop(module_properties)

        # Check if the number of candidate modules is over the limit, and if so,
        # keep those with the lowest complexity. 
        if len(self._candidate_modules) > MAX_CANDIDATE_MODULES:
            sorted_temp_info_list = sorted(list(self._candidate_modules.values()), key=lambda x:x.get_total_fitness())
            complexity_threshold = sorted_temp_info_list[MAX_CANDIDATE_MODULES]

            # Find those that need deletion
            marked_for_deletion = []
            for module_properties, info in self._candidate_modules.items():
                if info.get_total_fitness() > complexity_threshold:
                    marked_for_deletion.append(module_properties)
            # Delete.
            print(f"Deleted {len(marked_for_deletion)} ")
            for module_properties in marked_for_deletion:
                # NOTE: Don't add those modules to the expired list, they were not
                # fully examined.
                self._candidate_modules.pop(module_properties)

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
                # if module_properties.module_type == ModuleType.NEURAL_LAYER: continue
                # Compare for potential removal.
                if info.average_fitness < min_fitness_value:
                    marked_for_deletion.append(module_properties)
            
            # Delete marked properties.
            for properties in marked_for_deletion:
                self._notable_modules.pop(properties)
            
        self.print_notable_modules()

    def print_notable_modules(self):
        """ Prints a table with the notable module data. """

        layer_list = []
        module_type_list = []
        average_fitness_list = []
        total_fitness_list = []
        complexity_list = []
        # Add desired properties to their respective lists.
        for module,info in self._notable_modules.items():
            if module.module_type == ModuleType.NEURAL_LAYER:
                layer_list.append(module.layer)
            else:
                layer_list.append('N/A')
            module_type_list.append(module.module_type)
            average_fitness_list.append(info.average_fitness)
            total_fitness_list.append(info.get_total_fitness())
            complexity_list.append(module.complexity)

        # Create dictionary.
        df_dict = {
            'LAYER' : layer_list,
            'MODULE TYPE' : module_type_list,
            'AVERAGE FITNESS': average_fitness_list,
            'TOTAL FITNESS': total_fitness_list,
            'COMPLEXITY': complexity_list
        }

        df = pd.DataFrame.from_dict(df_dict)
        print(tabulate(df,tablefmt='fancy_grid', headers='keys', colalign='center'))

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
            self.best_network_data[BEST_NETWORK_LABEL] = deepcopy(neural_module)
            self.best_network_data[BEST_NETWORK_SCORE_LABEL] = neural_module.fitness
            self.best_module_updated = True

    def on_best_module_updated(self, evaluate=False, verbose=True):
        # Evaluate the best module fitness on more training epochs.
        self.save_best_module()
        if evaluate == True:
            print("Evaluating new best network...")
            best_module = self.best_network_data[BEST_NETWORK_LABEL]
            accuracy, _ = self.evaluator.evaluate(best_module, evaluation_epochs=BEST_MODULE_EVALUATION_EPOCHS)
            self.best_network_data[EVALUATED_SCORE_LABEL] = accuracy
            if verbose == True:
                print(f"A new best network was found with an accuracy value of {accuracy:.3f}")

    def save_best_module(self):
        """ Saves the best module data in a pickle file. """
        save_path = f"{BEST_NETWORK_DATA_SAVE_BASE_PATH}/activity_recognition_best.pkl"
        with open(save_path,'wb') as save_file:
            #print(self.best_network_data)
            pkl.dump(self.best_network_data, save_file)
        
        # Reset the save flag.
        self.best_module_updated = False

            