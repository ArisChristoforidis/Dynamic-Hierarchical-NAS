import re
from nord.neural_nets import LocalEvaluator, BenchmarkEvaluator, NeuralDescriptor
from nord.configs import INPUT_SHAPE
from config import CHANNEL_COUNT, LAYER_INPUT_PREFIX, LAYER_OUTPUT_SUFFIX, METRIC, NODE_INPUT_TAG, STRIDE_COUNT
from config import DATASET, EPOCHS, LAYERS_LIST
from torch.optim import Adam
import traceback


class Evaluator:

    def __init__(self):
        self.evaluator = LocalEvaluator(optimizer_class=Adam, optimizer_params= {}, verbose= False)
        self.channels = CHANNEL_COUNT
        self.strides = STRIDE_COUNT

    def evaluate(self, neural_module):
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
        descriptor = self._module_to_descriptor(neural_module)

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

    def _module_to_descriptor(self, neural_module):
        """
        Converts a neural module to a descriptor object.

        Parameters
        ----------
        neural_module: A neural module.

        Returns
        -------
        descriptor: NeuralDescriptor
            The neural descriptor object representing the net of the neural module.
        """

        full_graph, layer_types, input_idx, output_idx = neural_module.get_graph()
        descriptor = NeuralDescriptor()        

        # Add nodes.
        nodes = full_graph.nodes()
        for node in nodes:
            # Skip input/output node, we already added them.
            if node == input_idx or node == output_idx: continue
            # Create correct layer and parameters according to layer type.
            layer, kernel, params = re.split(pattern=r'_\.',string=layer_types[node])
            kernel = int(kernel)
            is_convolutional = layer == 'CONV'
            if layer == 'CONV':
                pass

        # Connect layers by iterating through the graph edges.
        edges = full_graph.edges()
        for source,dest in edges:
            # Get source, dest layer names.
            input_name =  str(source)
            output_name = str(dest)
            # Append input/output tag if they are not the INPUT/OUTPUT layers.
            if source != input_idx: input_name += "_" + LAYER_OUTPUT_SUFFIX
            if dest != output_idx: input_name += "_" + LAYER_INPUT_PREFIX
            # Connect.
            descriptor.connect_layers(input_name, output_name)

        return descriptor


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
        descriptor = self._module_to_descriptor(neural_module)
        acc, time = self.evaluator.descriptor_evaluate(descriptor)
        return acc, time

    def get_available_layers(self):
        return self.evaluator.get_available_ops()

    def _module_to_descriptor(self, neural_module):
        """
        Converts a neural module to a descriptor object.

        Parameters
        ----------
        neural_module: A neural module.

        Returns
        -------
        descriptor: NeuralDescriptor
            The neural descriptor object representing the net of the neural module.
        """
        # Create descriptor.
        descriptor = NeuralDescriptor()
        # Get full graph data.
        full_graph, layer_types, input_idx, output_idx = neural_module.get_graph()
        # Add input/output layers.
        descriptor.add_layer('input', {}, str(input_idx))
        descriptor.add_layer('output', {}, str(output_idx))

        # Create layers by iterating through the nodes.
        nodes = full_graph.nodes()
        for node in nodes:
            # Skip input/output node, we already added them.
            if node == input_idx or node == output_idx: continue
            # Get layer type and add it to the descriptor.
            layer_type = layer_types[node]
            descriptor.add_layer(layer_type, {}, str(node))

        # Connect layers by iterating through the graph edges.
        edges = full_graph.edges()
        for source,dest in edges:
            descriptor.connect_layers(str(source), str(dest))

        return descriptor
