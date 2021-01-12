import re

from torch.nn.modules.linear import Identity
from nord.neural_nets import LocalEvaluator, BenchmarkEvaluator, NeuralDescriptor
from nord.configs import INPUT_SHAPE
from config import CHANNEL_COUNT, DROPOUT_PROBABILITY, INVALID_NETWORK_FITNESS, INVALID_NETWORK_TIME, LAYER_INPUT_PREFIX, LAYER_OUTPUT_SUFFIX, METRIC, NODE_INPUT_TAG, NODE_OUTPUT_TAG, STRIDE_COUNT
from config import DATASET, EPOCHS, LAYERS_LIST
from torch.optim import Adam
import torch.nn as nn
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
            return INVALID_NETWORK_FITNESS, INVALID_NETWORK_TIME
            """
            trace = traceback.format_exc()
            # TODO: Initialize error log.
            traceback.print_exc()
            """
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

        # Add the rest of the nodes.
        nodes = full_graph.nodes()
        for node in nodes:
            # Skip input/output node, we will add them later.
            if node == input_idx or node == output_idx: continue
            # Create correct layer and parameters according to layer type.
            layer_label = layer_types[node]
            layer_name, kernel, params = re.split(pattern=r'[_\.]',string=layer_label)
            kernel = int(kernel)
            is_convolutional = layer_name == 'CONV'
            # Convolutional layer.
            if layer_name == 'CONV':
                layer = nn.Conv1d
                # The 'H' parameter means half channels.
                channels = self.channels if params == 'H' else int(self.channels / 2)
                parameters = {'in_channels': 1000,
                              'out_channels': channels,
                              'kernel_size': kernel,
                              'stride': self.strides}
            # Pooling layer.
            elif layer_name == 'POOL':
                layer = nn.MaxPool1d if params == 'M' else nn.AvgPool1d
                parameters = {'kernel_size': kernel,
                              'stride': kernel}
            elif layer_name == 'DROPOUT':
                # For dropout layers, the kernel will be the kernel as a pct
                layer = nn.Dropout
                dropout_pct = kernel / 100
                parameters = {'p': dropout_pct}
            elif layer_name == 'RELU':
                layer = nn.ReLU6
                parameters = {}
            else:
                # Not a known layer.
                raise Exception(f'[Evaluator] Undefined layer "{layer_name}"')

            # Add layer to descriptor.
            layer_name_in = f"{node}{LAYER_INPUT_PREFIX}"
            descriptor.add_layer(layer, parameters, name=layer_name_in)

            # If dealing with a convolutional layer, add intermediate layers.
            if is_convolutional == True:
                # NOTE: descriptor.add_layer_sequential() causes problems(naming inconsistencies).
                relu_layer_name = f"{node}RELU"
                batchnorm_layer_name = f"{node}BATCHNORM"
                dropout_layer_name = f"{node}DROPOUT"
                descriptor.add_layer(nn.ReLU6, {}, name=relu_layer_name )
                descriptor.add_layer(nn.BatchNorm1d, {'num_features': channels},  name=batchnorm_layer_name) 
                descriptor.add_layer(nn.Dropout, {'p': DROPOUT_PROBABILITY},  name=dropout_layer_name)
                # Connect layers.
                descriptor.connect_layers(layer_name_in,relu_layer_name)
                descriptor.connect_layers(relu_layer_name,batchnorm_layer_name)
                descriptor.connect_layers(batchnorm_layer_name,dropout_layer_name)
                
            # Add layer output. This is done in this way to facilitate the extra
            # layers that need to be added in the case of a convolutional base layer.
            layer_name_out = f"{node}{LAYER_OUTPUT_SUFFIX}"
            descriptor.add_layer_sequential(nn.Identity, {}, name=layer_name_out)

        # Add input/output layers.
        input_layer_name = f"{input_idx}_{NODE_INPUT_TAG}"
        output_layer_name = f"{output_idx}_{NODE_OUTPUT_TAG}"
        descriptor.add_layer(nn.Identity, {}, name=input_layer_name)
        descriptor.add_layer(nn.Identity, {}, name=output_layer_name)
        descriptor.first_layer = input_layer_name
        descriptor.last_layer = output_layer_name

        # Connect layers by iterating through the graph edges.
        edges = full_graph.edges()
        for source,dest in edges:
            # Get source, dest layer names.
            if source == input_idx:
                source_name = f"{input_idx}_{NODE_INPUT_TAG}"
            else:
                layer_label = layer_types[source]
                source_name = f"{source}{LAYER_OUTPUT_SUFFIX}"

            if dest == output_idx:
                dest_name = F"{output_idx}_{NODE_OUTPUT_TAG}"
            else:
                layer_label = layer_types[dest]
                dest_name = f"{dest}{LAYER_INPUT_PREFIX}"

            # Connect.
            descriptor.connect_layers(source_name, dest_name)

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
