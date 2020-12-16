# Author: Aris Christoforidis.

from enums import ModuleType
from module_properties import ModuleProperties
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
        self._notable_modules = {ModuleProperties(ModuleType.NEURAL_LAYER, layer, None, []) : 0 for layer in evaluator.get_available_layers()}

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
            w = self._notable_modules.values()
        else:
            # Restricted to a type. May return an empty list.
            candidates = [module for module in self._notable_modules.keys() if module.module_type == restrict_to]
            w = [self._notable_modules[module] for module in candidates]
        
        # If there are no numbers in here, divide probability evenly.
        if sum(w) <= 0: w = [1/len(w) for _ in w]
        
        notable_modules = rnd.choices(population=candidates, weights=w,k=count)
        return notable_modules
    
    def record_module_properties(self, neural_module_properties):
        """
        Receive a neural module and evaluate whether it should be added to the
        notable_modules list.

        Parameters
        ----------

        """
        