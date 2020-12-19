# Author: Aris Christoforidis.

from config import MAX_NOTABLE_MODULES
from enums import ModuleType
from module_properties import ModuleProperties, PropertiesInfo, TempPropertiesInfo
from neural_module import NeuralModule
from evaluation import Evaluator
import random as rnd

class ModuleManager:
    """
    The module manager keeps track of notable modules and scores them. It provides
    methods for picking a notable module based on their weighted score. 
    """

    def __init__(self, evaluator : Evaluator):
        # Create module properties objects for starting layers.
        self._notable_modules = {ModuleProperties(ModuleType.NEURAL_LAYER, layer, None, []) : PropertiesInfo() for layer in evaluator.get_available_layers()}
        self._candidate_modules = {}

    # NOTE: Not used.
    def get_notable_modules(self):
        """
        Notable modules getter.

        Returns
        -------
        notable_modules: dict(ModuleProperties->float)
            The notable modules dict.
        """
        return self._notable_modules

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
            candidates = self._notable_modules.keys()
            w = [info.average_fitness for info in self._notable_modules.values()]
        else:
            # Restricted to a type. May return an empty list.
            candidates = [module for module in self._notable_modules.keys() if module.module_type == restrict_to]
            w = [self._notable_modules[module] for module in candidates]
        
        # If there are no numbers in here, divide probability evenly.
        if sum(w) <= 0: w = [1/len(w) for _ in w]
        
        notable_modules = rnd.choices(population=candidates, weights=w,k=count)
        return notable_modules
    
    def record_module_properties(self, neural_module):
        """
        Receive a neural module and evaluate whether it should be added to the
        notable_modules list.

        Parameters
        ----------
        neural_module: NeuralModule
            A NeuralModule object.
        """
        properties = neural_module.get_module_properties()

        # If this properties object describes a "notable" module 
        if properties in self._notable_modules:
            self._notable_modules[properties].record(neural_module.fitness)
        else:
            # If this is a new properties module, record it.
            if properties not in self._candidate_modules:
                self._candidate_modules[properties] = TempPropertiesInfo()
            # Record fitness.
            self._candidate_modules[properties].record(neural_module.fitness)

    def on_generation_increase(self):
        """
        Call this when a generation changes.
        """

        # Get the info list and sort the list based on the average fitness 
        # of the notable modules.
        info_list = list(self._notable_modules.values())
        info_list.sort(key=lambda x: -x.average_fitness)
        # Get the minimum fitness value.
        min_fitness_threshold = info_list[-1].average_fitness
        
        # Temp module list.
        for module_properties, temp_info in self._candidate_modules:
            # If the candidate module has a fitness higher that the minimum on
            # the notable modules list, add it to the notable modules.
            if temp_info.average_fitness > min_fitness_threshold:
                info = PropertiesInfo(temp_info)
                self._notable_modules[module_properties] = info

        # Notable module list.
        if len(self._notable_modules) > MAX_NOTABLE_MODULES:
            # Find the cutoff element and its value.
            threshold_info = info_list[MAX_NOTABLE_MODULES]
            min_fitness_value = threshold_info.average_fitness

            # Iterate through the dictionary and remove any entries below the
            # cutoff fitness value.
            for module_properties, info in self._notable_modules:
                # Careful with the core layers.
                # NOTE: I don't think that this is necessary, but its better to be safe.
                # In fact this may be removed in the future with the logic being: if
                # at some point the core layers are removed, then the mutation will 
                # happen with just the abstracted nodes, which would contain core layers.
                if module_properties.module_type == ModuleType.NEURAL_LAYER: continue
                # Compare for potential removal.
                if info.average_fitness < min_fitness_value:
                    self._notable_modules.pop(module_properties)

        # NOTE: Is this all?

            