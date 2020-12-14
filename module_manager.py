# Author: Aris Christoforidis.

from neural_module import NeuralModule
from evaluation import Evaluator


class ModuleManager:
    """
    The module manager keeps track of notable modules and scores them. It provides
    methods for picking a notable module based on their weighted score. 
    """
    def __init__(self, evaluator : Evaluator):
        # TODO: 
        self.notable_modules = {NeuralModule(module, evaluator) : 0 for module in evaluator.get_available_layers()}
        