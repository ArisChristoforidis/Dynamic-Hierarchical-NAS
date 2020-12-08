from nord.neural_nets import LocalEvaluator,BenchmarkEvaluator
from nord.configs import INPUT_SHAPE, METRIC
from config import DATASET, EPOCHS, LAYERS_LIST
from torch.optim import Adam
import traceback


class Evaluator:

    def __init__(self):
        self.evaluator = LocalEvaluator(optimizer_class=Adam, optimizer_params= {}, verbose= False)

    def evaluate(self, genome):
        """
        Evaluates a network represented by a neural module.

        Parameters
        ---------
        neural_module: NeuralModule
            A neural module.

        Returns
        -------
        acc: float
            The network accuracy.
        
        time: float
            The evaluation time.
        """
        dim = INPUT_SHAPE[DATASET]
        descriptor = genome.to_descriptor(dimensions=dim)

        fitness = {METRIC: 0}
        total_time = 0
        try:
            loss, fitness, total_time = self.evaluator.descriptor_evaluate(descriptor=descriptor, epochs=EPOCHS, data_percentage=1, dataset=DATASET)
        except Exception:
            print('Invalid Descriptor')
            print(descriptor)
            trace = traceback.format_exc()
            # TODO: Initialize error log.
            traceback.print_exc()
        
        # TODO: History maybe.

        return fitness[METRIC], total_time

    def get_available_layers(self):
        return LAYERS_LIST

class NasBenchEvaluator(Evaluator):

    def __init__(self):
        self.evaluator = BenchmarkEvaluator(False)

    def evaluate(self, neural_module):
        """
        Evaluates a network represented by a neural module on the nasbench
        dataset.

        Parameters
        ---------
        neural_module: NeuralModule
            A neural module.

        Returns
        -------
        acc: float
            The netowrk accuracy.
        
        time: float
            The evaluation time.
        """
        descriptor = neural_module.to_descriptor()
        acc, time = self.evaluator.descriptor_evaluate(descriptor)
        return acc, time

    def get_available_layers(self):
        return self.evaluator.get_available_ops()