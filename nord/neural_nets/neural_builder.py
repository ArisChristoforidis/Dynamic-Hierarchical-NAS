
"""
Created on Sat Aug  4 18:24:35 2018

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import copy
from typing import Dict, List, Tuple, Type

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import AvgPooling
from torch import Tensor
from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
from torch.nn.modules.pooling import _AvgPoolNd, _MaxPoolNd

from nord.utils import (find_optimum_pool_size, get_layer_out_size,
                        get_transpose_out_size)

from .layers import Conv2d151, Flatten, Identity, SizeScaleLayer, types_dict
from .neural_descriptor import NeuralDescriptor

MIN_MAX_DIMENSIONS = 'MIN'


class NeuralNet(nn.Module):
    """Basic class, implementing a deep neural net.
    """

    def __init__(self, net_descriptor: NeuralDescriptor, num_classes: int,
                 input_shape: Tuple,
                 input_channels: int, sort: bool = False,
                 untrained: bool = False,
                 keep_dimensions: int = None,
                 problem_type: str = 'classification'):
        """Generates the pytorch graph of the network.

        Parameters
        ----------
        net_descriptor : NeuralDescriptor
            A descriptor of the netowrk's structure.
            (see :class:`~.neural_descriptor.NeuralDescriptor`)

        num_classes : int
            The number of classes (output layer neurons) of the network.

        input_shape : tuple
            Shape of inputs.

        input_channels : int
            Number of input channels.

        sort : bool
            If set to True, the depth of each layer will be
            added before its name.

        keep_dimensions: int
            This parameter dicatates the rescaling of layer outputs.
            The outputs are rescaled to 'keep_dimensions' size.
            If None, the network will not rescale the outputs at all.


        problem_type : str
            Either regression or classification.


        """
        super().__init__()

        self.descriptor = copy.deepcopy(net_descriptor)
        self.connections = self.descriptor.connections
        self.in_connections = self.descriptor.incoming_connections
        self.input_shapes = dict()
        self.input_channels = dict()

        self.first_layer = self.descriptor.first_layer
        self.last_layer = self.descriptor.last_layer
        self.keep_dimensions = keep_dimensions

        self.functional = True
        if len(self.in_connections[self.last_layer]) == 0:
            self.functional = False
            return

        if len(self.connections[self.first_layer]) == 0:
            self.functional = False
            return

        self.layers = dict()
        self.num_classes = num_classes
        # Get all paths

        layer_dict = self.__name_all_paths(self.descriptor.layers)

        self.input_shapes[self.first_layer] = input_shape[0]
        self.input_channels[self.first_layer] = input_channels
        self.data_shape = len(input_shape)

        self.hidden_outs = dict()
        self.recursions = [[x[-1], x[0]] for x in self.get_recursions()]
        self.recursion_paths = [x[1:-1] for x in self.get_recursions()]
        self.recursion_origins = [x[0] for x in self.recursions]

        # Delete recursive layers
        # to_remove = set()
        # keys = layer_dict.keys()
        # for key in layer_dict.keys():
        #     # If it is part of a recursion delete it
        #     for path in self.recursion_paths:
        #         if key in path:
        #             to_remove.add(key)

        # for key in to_remove:
        #     del layer_dict[key]

        self.scale_all_layers()
        # Instantiate all layers
        for key in layer_dict.keys():
            # Get layer type and parameters
            layer, params = layer_dict[key]
            # Get instance
            # print(layer, params)
            layer_instance = layer(**params)
            if untrained:
                for param in layer_instance.parameters():
                    param.requires_grad = False
            # Set as attribute
            setattr(self, key, layer_instance)
            self.layers[key] = getattr(self, key)

        self.flatten = Flatten()

        in_sz = self.__get_flat_out_size_calc()
        # self.get_output_size(self.last_layer)
        self.fc1 = nn.Linear(in_sz, int(in_sz/4))
        # self.bn1 = nn.BatchNorm1d(int(in_sz/4))
        self.fc2 = nn.Linear(int(in_sz/4), int(in_sz/8))
        self.final_layer = nn.Linear(int(in_sz/8),
                                     self.num_classes)
        self.final_activation = Identity()
        if problem_type == 'classification':
            self.final_activation = nn.LogSoftmax(dim=1)

    def __get_flat_out_size_dynamic(self):
        self.eval()  # Set to eval so batchnorm won't complain
        input_channels = self.input_channels[self.first_layer]
        input_sz = self.input_shapes[self.first_layer]
        input_shape = [input_sz for _ in range(self.data_shape)]
        dummy = torch.randn([1, input_channels, *input_shape])
        out = self.flatten(self.__internal_fwd(dummy))
        final_sz = list(out.shape)[-1]

        self.train()  # ReSet to train
        return final_sz

    def __get_flat_out_size_calc(self):
        # in_sz = 0
        # in_channels = 0

        in_sz, in_dim = self.get_output_size(self.last_layer)
        in_channels = self.get_output_channels(self.last_layer)
        # print(in_channels, in_sz, self.data_shape)
        return (in_sz ** in_dim) * in_channels

    def __internal_fwd(self, x: Tensor):
        """ Implement the forward pass
        """
        # print('FORWARD', '*'*30)
        x = self.layers[self.first_layer](x)
        outs = {self.first_layer: x}

        keys = self.connections.keys()
        # Example: key='layer_1_1'
        for key in sorted(keys):
            # print('*')
            if key is self.first_layer:
                continue
            # from_layer = [layer_0_1, layer_0_2]
            other_in = None
            inputs = self.in_connections[key]
            for from_layer in set(inputs):

                if int(from_layer.split('_')[0]) > int(key.split('_')[0]):
                    # tmp_in = self.hidden_outs[from_layer]
                    continue

                else:
                    # tmp_in = layer_0_1(x)
                    tmp_in = outs[from_layer]

                # TODO: THIS "if" is for DEBUG ONLY!!!!
                # if not tmp_in.shape[-1] == self.get_output_size(from_layer):
                #     print(from_layer, '->', key)
                #     print('E:', self.get_output_size(from_layer),
                #           'R:', tmp_in.shape[-1])

                other_in = 0 if other_in is None else other_in
                if other_in is not None:
                    # print('-OTHER INS-', key, inputs)
                    other_in = tmp_in + other_in

                else:
                    other_in = tmp_in

            outs[key] = self.layers[key](other_in)

        # for key in self.recursion_origins:
        #     self.hidden_outs[key] = outs[key]

        # for k in sorted(list(outs.keys())):
        #     print('*'*30)
        #     print(k)
        #     print(outs[k])

        return outs[self.last_layer]

    def forward(self, x: Tensor):

        flats = self.flatten(self.__internal_fwd(x))
        x = self.fc1(flats)
        # x = self.bn1(x)
        x = self.fc2(x)
        x = self.final_layer(x)
        x = self.final_activation(x)
        return x

    def scale_all_layers(self):
        """ Scale all inputs for multi-input layers.
        """
        scaled = set()
        scaled.add(self.first_layer)
        not_scaled = list(self.descriptor.layers.keys())
        not_scaled.remove(self.first_layer)
        self.pure_recursions = set()
        while len(not_scaled) > 0:

            for target in not_scaled:

                origins = []
                tmp = self.in_connections[target]
                # TODO: Assure that this is not needed V
                # # Remove recursions
                # for i in range(len(tmp)):
                #     if int(tmp[i].split('_')[0]) < int(target.split('_')[0]):
                #         origins.append(tmp[i])

                origin_set = set(tmp)

                # If the origins are already scaled, proceed

                # if origin_set <= scaled:
                #     if len(origin_set) > 0:
                #         self.scale_target_inputs(target, origin_set)
                #         scaled.add(target)
                #     else:
                #         self.pure_recursions.add(target)
                #     not_scaled.remove(target)

                # elif origin_set <= self.pure_recursions:
                #     self.pure_recursions.add(target)
                #     not_scaled.remove(target)

                if origin_set <= scaled:
                    self.scale_target_inputs(target, origin_set)
                    scaled.add(target)
                    not_scaled.remove(target)

        # # Deal with recursions
        # for origin, target in self.recursions:
        #     self.scale_recursion(target, origin)

    # def scale_recursion(self, target, origin):
    #     out_sz, out_ch = self.get_min_input_size(target, [origin])
    #     self.hidden_outs[origin] = Variable(torch.zeros(1, out_ch, out_sz,
    # out_sz))
    #     in_sz = self.input_shapes[target]
    #     in_ch = self.input_channels[target]
    #     print(in_sz, in_ch)
    #     self.scale_layer_outputs(origin, target, in_sz, in_ch)

    def scale_target_inputs(self, target: str, origins: List[str]):
        """ Scale the selected origins layers, in order to concatenate
            inputs to target.

            Parameters
            ----------
            target : string
                Name of the input layer

            origins : list[string]
                Names of output layers
        """

        min_sz, min_channels = self.get_min_input_size(target, origins)
        if self.keep_dimensions is not None:
            # self.input_shapes[self.first_layer]
            min_sz = self.keep_dimensions
        # print(target, origins)

        if 'in_channels' in self.descriptor.layers[target][1]:
            self.descriptor.layers[target][1]['in_channels'] = min_channels
        elif 'num_features' in self.descriptor.layers[target][1]:
            self.descriptor.layers[target][1]['num_features'] = min_channels

        self.input_shapes[target] = min_sz
        self.input_channels[target] = min_channels

        for origin in origins:
            self.scale_layer_outputs(origin, target, min_sz, min_channels)

    def scale_layer_outputs(self, origin: str, target: str, min_sz: int, min_channels: int):
        """ Scale the selected origin layer, in order to concatenate
            inputs to target.

            Parameters
            ----------
            target : string
                Name of the input layer

            origin : string
                Names of output layer

            min_sz : int
                The desired output size
                (minimum size that enables concatenation)

            min_channels : int
                The desired output channels
                (minimum channels that enable concatenation)
        """

        origin_out = self.get_output_size(origin)[0]
        out_ch = self.get_output_channels(origin)
        layer_name = origin+'_0sizescale_'+target
        ch_name = origin+'_1chanscale_'+target

        layer = None
        params = None
        # print(origin, target, origin_out, min_sz)
        # If the target is a flatten layer do nothing.
        # or (self.descriptor.layers[target][0] is nn.MaxPool1d) or (self.descriptor.layers[target][0] is nn.MaxPool2d):
        if (self.descriptor.layers[target][0] is Flatten):
            return
        if (self.descriptor.layers[origin][0] is nn.Linear) and (self.descriptor.layers[target][0] is nn.Linear):
            return
        # Else if the output size differs from the input size
        # scale the output size
        elif not origin_out == min_sz:
            layer = SizeScaleLayer
            params = {'final_size': min_sz}

        if layer is not None:
            self.descriptor.add_layer(layer, params, name=layer_name)
            self.descriptor.connect_layers(origin, layer_name)
            self.descriptor.connect_layers(layer_name, target)
            self.descriptor.disconnect_layers(origin, target)

            self.input_shapes[layer_name] = origin_out
            self.input_channels[layer_name] = out_ch
            self.input_shapes[target] = min_sz

        # Scale channels as well
        if out_ch is not min_channels:
            ch_layer = nn.Conv2d
            if self.data_shape == 1:
                ch_layer = nn.Conv1d

            ch_params = {'in_channels': out_ch, 'out_channels': min_channels,
                         'kernel_size': 1}
            self.descriptor.add_layer(ch_layer, ch_params, name=ch_name)
            if layer is not None:
                self.descriptor.connect_layers(layer_name, ch_name)
                self.descriptor.connect_layers(ch_name, target)
                self.descriptor.disconnect_layers(layer_name, target)

            else:
                self.descriptor.connect_layers(origin, ch_name)
                self.descriptor.connect_layers(ch_name, target)
                self.descriptor.disconnect_layers(origin, target)

            self.input_shapes[ch_name] = self.input_shapes[target]
            self.input_channels[ch_name] = self.input_channels[target]
            self.input_channels[target] = min_channels

    def get_min_input_size(self, target: str, origins: List[str]):
        """ Get the minimum input size that enables concatenation.

            Parameters
            ----------
            target : string
                Name of the input layer

            origins : list[string]
                Names of output layers

            Returns
            -------
            min_dimensions : int
                Minimum size

            min_channels : int
                Minimum channels

        """
        if self.descriptor.layers[target][0] is nn.Linear:
            return self.descriptor.layers[target][1]['in_features'], 1

        # min_channels = float('inf')
        # min_dimensions = float('inf')

        # for node in origins:
        #     channels = self.get_output_channels(node)
        #     dimensions = self.get_output_size(node)[0]
        #     min_dimensions = min(dimensions, min_dimensions)
        #     min_channels = min(channels, min_channels)

        min_channels = 0
        min_dimensions = 0

        if MIN_MAX_DIMENSIONS == 'MIN':
            min_channels = 1e+10
            min_dimensions = 1e+10

        for node in origins:
            channels = self.get_output_channels(node)
            dimensions = self.get_output_size(node)[0]
            if MIN_MAX_DIMENSIONS == 'MAX':
                min_dimensions = max(dimensions, min_dimensions)
                min_channels = max(channels, min_channels)
            else:
                min_dimensions = min(dimensions, min_dimensions)
                min_channels = min(channels, min_channels)

        if 'kernel_size' in self.descriptor.layers[target][1]:
            kernel = self.descriptor.layers[target][1]['kernel_size']
            if min_dimensions < kernel:
                min_dimensions = kernel

        return min_dimensions, min_channels

    def get_output_size(self, node: str):
        """Calculate the output size of a layer

        Parameters
        ----------
        node : string
            Name of the layer

        Returns
        -------
        The layer's output size and dimensions

        """
        node_class = self.descriptor.layers[node][0]
        params = self.descriptor.layers[node][1]

        if node == self.first_layer:
            return self.input_shapes[node], self.data_shape

        if node_class is Flatten:
            return (self.input_channels[node] *
                    self.input_shapes[node] ** self.data_shape), 1

        elif node_class is nn.Linear:
            return self.descriptor.layers[node][1]['out_features'], 1

        elif node_class is nn.BatchNorm2d:
            return self.input_shapes[node], 2

        elif node_class is Conv2d151:
            stride = params['stride'] if 'stride' in params else 1
            first_h = get_layer_out_size(self.input_shapes[node],
                                         1,
                                         0,
                                         stride,
                                         1)
            first_w = get_layer_out_size(self.input_shapes[node],
                                         5,
                                         0,
                                         stride,
                                         1)
            second_h = get_layer_out_size(first_h,
                                          5,
                                          0,
                                          stride,
                                          1)
            second_w = get_layer_out_size(first_w,
                                          1,
                                          0,
                                          stride,
                                          1)
            return second_w, 2

        kernel = params['kernel_size'] if 'kernel_size' in params else 1
        padding = params['padding'] if 'padding' in params else 0
        dilation = params['dilation'] if 'dilation' in params else 1
        stride = params['stride'] if 'stride' in params else 1

        # print(node_class)
        if issubclass(node_class, _ConvTransposeMixin):
            out_pad = params['output_padding']
            min_sz = get_transpose_out_size(self.input_shapes[node],
                                            kernel,
                                            padding,
                                            dilation,
                                            stride,
                                            out_pad)
            if node_class is nn.ConvTranspose2d:
                return min_sz, 2
            else:
                return min_sz, 1

        elif issubclass(node_class, _ConvNd) or issubclass(node_class, _MaxPoolNd) or issubclass(node_class, _AvgPoolNd):
            min_sz = get_layer_out_size(self.input_shapes[node],
                                        kernel,
                                        padding,
                                        dilation,
                                        stride)

            min_sz = kernel if min_sz <= 0 else min_sz

        else:
            min_sz = self.input_shapes[node]

        if (node_class is nn.Conv2d) or (node_class is nn.MaxPool2d) or (node_class is nn.AvgPool2d):
            return min_sz, 2
        else:
            return min_sz, self.data_shape

    def get_output_channels(self, node: str):
        """Calculate the output channels of a layer

        Parameters
        ----------
        node : string
            Name of the layer

        Returns
        -------
        The layer's output channels

        """
        params = self.descriptor.layers[node][1]
        if 'out_channels' in params:
            return params['out_channels']
        if self.descriptor.layers[node][0] is Flatten:
            return 1
        elif self.descriptor.layers[node][0] is Identity:
            return self.input_channels[node]
        elif self.descriptor.layers[node][0] is nn.BatchNorm2d:
            return self.descriptor.layers[node][1]['num_features']
        return self.input_channels[node]

    def remove_connection(self, origin: str, target: str):
        """Remove a connection from the network

        Parameters
        ----------
        origin : string
            Name of the origin layer

        target : string
            Name of the target layer

        """

        while target in self.connections[origin]:
            self.connections[origin].remove(target)
        while origin in self.in_connections[target]:
            self.in_connections[target].remove(origin)

    def add_connection(self, origin: str, target: str):
        """Add a connection to the network

        Parameters
        ----------
        origin : string
            Name of the origin layer

        target : string
            Name of the target layer

        """
        if origin not in self.connections:
            self.connections[origin] = []
            self.in_connections[origin] = []
        if target not in self.connections:
            self.connections[target] = []
            self.in_connections[target] = []

        self.in_connections[target].append(origin)
        self.connections[origin].append(target)

    def __name_all_paths(self, layers_in: Dict[str, List],
                         acyclic: bool = True):
        """Find all the possible paths from the input layer to the output layer
           and name the nodes according to their depth.

        Parameters
        ----------
        layers_in : dict
            Contains the layer name as key. The dictionary's items
            are the layer's type and its parameters.

        acyclic : bool
            If true, only acyclic graphs are considered.

        Returns
        ----------

        A dictionary with updated names
        """

        layers = layers_in

        def set_layer_level(layer, level):
            new_key = str(level).zfill(3)+'_'+layer
            layers[new_key] = layers.pop(layer)
            self.connections[new_key] = self.connections.pop(layer)
            self.in_connections[new_key] = self.in_connections.pop(layer)

            if layer == self.first_layer:
                self.first_layer = new_key
            elif layer == self.last_layer:
                self.last_layer = new_key
            return new_key

        paths = None
        if acyclic:
            paths = self.get_acyclic_paths()
        else:
            paths = self.get_direct_paths()

        renamed = {}
        paths = sorted(paths, reverse=True, key=len)
        for path in paths:
            level = 1
            for node in path:
                if node not in renamed:
                    if node == self.last_layer:
                        level += 1
                    if node == self.first_layer:
                        level -= 1
                    new_node = set_layer_level(node, level)
                    renamed[node] = new_node

                else:
                    level = int(renamed[node].split('_')[0])
                level += 1

        for new_key in self.connections:
            for i in range(len(self.connections[new_key])):
                node = self.connections[new_key][i]
                if node in renamed:
                    self.connections[new_key][i] = renamed[node]

        for new_key in self.in_connections:
            for i in range(len(self.in_connections[new_key])):
                node = self.in_connections[new_key][i]
                if node in renamed:
                    self.in_connections[new_key][i] = renamed[node]
        return layers

    def to_networkx(self):
        """Create a networkx graph from the network

        Returns
        -------
        G : MultiDiGraph

        """

        G = nx.MultiDiGraph()
        for start in self.connections:
            ends = self.connections[start]
            for end in ends:
                G.add_edge(start, end)

        # names = {}
        # for layer in self.descriptor.layers:
        #     names[layer] = str(layer)+' '+
        #                    str(self.descriptor.layers[layer][1])

        # names['0_-2out'] = 'INPUTS'
        # H = nx.relabel.relabel_nodes(G, names)
        return G

    def get_direct_paths(self):
        """Return all the direct paths from the input to the output layer

        Returns
        -------
        list with all the direct paths
        """
        G = self.to_networkx()
        try:
            paths = nx.all_simple_paths(G, self.first_layer, self.last_layer)
        except nx.NodeNotFound:
            paths = [[]]
        return [p for p in paths]

    def get_acyclic_paths(self):
        """Return all the direct acyclic paths from the input to the output layer

        Returns
        -------
        list with all the direct paths
        """
        G = self.to_networkx()
        # recs = self.get_recursions()

        try:
            paths = nx.all_simple_paths(G, self.first_layer, self.last_layer)
        except nx.NodeNotFound:
            paths = [[]]

        rets = []
        for p in paths:
            if p not in rets:
                rets.append(p)
        return rets

    def get_recursions(self):
        """Return all the recursions in the network

        Returns
        -------
        list with all the recursions
        """
        G = self.to_networkx()
        cycles = nx.simple_cycles(G)
        return [sorted(c) for c in cycles]

    def plot(self, title: str = None):
        import matplotlib.pyplot as plt
        import numpy as np

        spacing_h = 0.5
        spacing_w = 0.1
        half_spacing = 0.25

        def my_layout(G, paths, recursions):
            nodes = G.nodes
            lengths = [-len(x) for x in paths]
            sorted_ = np.argsort(lengths)

            positions = dict()
            h = 0
            w = 0

            for index in sorted_:
                h = 0
                added = False
                path = paths[index]
                for node in path:
                    if node not in positions:
                        positions[node] = (w, h)
                        added = True
                        h -= spacing_h
                    else:
                        if h > positions[node][1]:
                            h = positions[node][1]

                if added:
                    if w >= 0:
                        w += spacing_w
                    w *= -1

            h = 0
            for node in nodes:
                if node not in positions:
                    positions[node] = (w, h)
                    h -= spacing_h

            f_l = self.first_layer
            l_l = self.last_layer
            if f_l in positions:
                positions[f_l] = (positions[f_l][0],
                                  positions[f_l][1]+spacing_h)
            if l_l in positions:
                positions[l_l] = (positions[l_l][0],
                                  positions[l_l][1]-spacing_h)

            recursed_nodes = []
            for path in recursions:
                last = sorted(path)[-1]
                if last not in recursed_nodes:
                    positions[last] = (positions[last][0]+half_spacing,
                                       positions[last][1])
                    recursed_nodes.append(last)
            return positions

        G = self.to_networkx()
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        plt.figure()
        plt.title(title)
        in_path = self.get_direct_paths()
        recs = self.get_recursions()
        # pos = graphviz_layout(G, root='-2')
        pos = my_layout(G, in_path, recs)
        nx.draw(G, pos=pos, node_color='b',
                node_shape='s',
                node_size=800,
                with_labels=False,
                linewidths=0)

        nodes = set()
        for p in in_path:
            for node in p:
                nodes.add(node)
        for p in recs:
            for node in p:
                nodes.add(node)

        labels = {}

        for n in nodes:
            name = type(self.layers[n]).__name__
            if ('Conv' in name) or ('Pool' in name):
                name += str(self.layers[n].kernel_size)
            labels[n] = name
            # s = n.split('_')
            # label = ''
            # if len(s) == 2:
            #     label = s[1]
            # else:
            #     label = s[2]
            # label = ''.join([i for i in label if not i.isdigit()])

            # label = label.replace('-', '')
            # if not label == 'scale':
            #     label = label.replace('scale', '')
            # labels[n] = label

        nx.draw_networkx_nodes(G, pos=pos,
                               node_color='w',
                               node_shape='s',
                               node_size=800,
                               nodelist=list(nodes),
                               with_labels=False,
                               linewidths=0)

        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

        plt.show()


class GraphRankNet(nn.Module):
    """Basic class, implementing a deep neural net.
    """

    def __init__(self, net_descriptor: NeuralDescriptor, num_classes: int,
                 input_dimensions: int,
                 sort: bool = False,
                 untrained: bool = False):
        """Generates the pytorch graph of the network.

        Parameters
        ----------
        net_descriptor : NeuralDescriptor
            A descriptor of the netowrk's structure.
            (see :class:`~.neural_descriptor.NeuralDescriptor`)

        num_classes : int
            The number of classes (output layer neurons) of the network.

        input_dimensions : int
            Size of feature vector.

        sort : bool
            If set to True, the depth of each layer will be
            added before its name.
        """
        super().__init__()

        self.dummy_param = nn.Parameter(torch.empty(0))

        self.descriptor = copy.deepcopy(net_descriptor)
        self.connections = self.descriptor.connections
        self.in_connections = self.descriptor.incoming_connections
        self.input_shapes = {}
        self.first_layer = self.descriptor.first_layer
        self.last_layer = self.descriptor.last_layer

        self.functional = True
        if len(self.in_connections[self.last_layer]) == 0:
            self.functional = False
            return

        if len(self.connections[self.first_layer]) == 0:
            self.functional = False
            return

        self.layers = dict()
        self.num_classes = num_classes
        # Get all paths

        layer_dict = self.__name_all_paths(self.descriptor.layers)

        self.input_shapes[self.first_layer] = input_dimensions

        self.scale_all_layers()
        # Instantiate all layers
        for key in layer_dict.keys():
            # Get layer type and parameters
            layer, params = layer_dict[key]
            # Get instance
            # print(layer, params)
            layer_instance = layer(**params)
            if untrained:
                for param in layer_instance.parameters():
                    param.requires_grad = False
            # Set as attribute
            setattr(self, key, layer_instance)
            self.layers[key] = getattr(self, key)

        self.flatten = Flatten()
        in_sz = self.__get_flat_out_size_calc()
        self.fc1 = nn.Linear(in_sz*2, in_sz//2)
        self.final_layer = nn.Linear(in_sz//2, self.num_classes)
        self.final_activation = nn.LogSoftmax(dim=1)
        if self.num_classes == 1:
            self.fc1 = nn.Linear(in_sz, in_sz//2)
            self.fc2 = nn.Linear(in_sz//2, in_sz//4)
            self.final_layer = nn.Linear(in_sz//4, self.num_classes)
            self.final_activation = nn.Identity()

        # self.fc1 = nn.Linear(in_sz, in_sz//2)
        # self.final_layer = nn.Linear(in_sz, self.num_classes)

        # self.final_layer = nn.Linear(in_sz*2, self.num_classes)

    def __get_flat_out_size_calc(self):

        in_sz = self.get_output_size(self.last_layer)
        return in_sz

    def __internal_fwd(self, g):
        """ Implement the forward pass
        """
        device = self.dummy_param.device
        with g.local_scope():
            # print('FORWARD', '*'*30)
            x = g.ndata['x'].to(device).float()
            x = self.layers[self.first_layer](x)
            outs = {self.first_layer: x}

            keys = self.connections.keys()
            # Example: key='layer_1_1'
            for key in sorted(keys):
                # print('*')
                if key is self.first_layer:
                    continue
                # from_layer = [layer_0_1, layer_0_2]
                other_in = None
                inputs = self.in_connections[key]
                for from_layer in set(inputs):
                    if int(from_layer.split('_')[0]) > int(key.split('_')[0]):
                        # tmp_in = self.hidden_outs[from_layer]
                        continue

                    else:
                        # tmp_in = layer_0_1(x)
                        tmp_in = outs[from_layer]

                    # TODO: THIS "if" is for DEBUG ONLY!!!!
                    # if not tmp_in.shape[-1] == self.get_output_size(from_layer):
                    #     print(from_layer, '->', key)
                    #     print('E:', self.get_output_size(from_layer),
                    #           'R:', tmp_in.shape[-1])

                    if other_in is not None:
                        # print('-OTHER INS-', key, inputs)
                        # print(other_in.shape, tmp_in.shape)
                        other_in = torch.cat((tmp_in, other_in), 1)

                    else:
                        other_in = tmp_in

                if type(self.layers[key]).__name__ in ('SAGEConv', 'AvgPooling', 'MaxPooling', 'SortPooling'):
                    outs[key] = self.layers[key](g, other_in)
                else:
                    outs[key] = self.layers[key](other_in)

            return outs[self.last_layer]

    def forward(self, g):

        if self.num_classes > 1:
            g1, g2 = g
            flats1 = self.flatten(self.__internal_fwd(g1))
            flats2 = self.flatten(self.__internal_fwd(g2))

            # print(flats1.shape, flats2.shape)

            h_d = torch.cat((flats1, flats2), 1)
            x = F.relu6(self.fc1(h_d))
            x = F.relu6(self.final_layer(x))
            x = self.final_activation(x)
            return x
        else:
            h_d = self.flatten(self.__internal_fwd(g))
            x = F.relu6(self.fc1(h_d))
            x = F.relu6(self.fc2(x))
            x = self.final_layer(x)
            x = self.final_activation(x)
            return x

    def scale_all_layers(self):
        """ Scale all inputs for multi-input layers.
        """
        scaled = set()
        scaled.add(self.first_layer)
        not_scaled = list(self.descriptor.layers.keys())
        not_scaled.remove(self.first_layer)
        self.pure_recursions = set()
        while len(not_scaled) > 0:

            for target in not_scaled:

                tmp = self.in_connections[target]
                origin_set = set(tmp)

                # If the origins are already scaled, proceed
                if origin_set <= scaled:
                    self.scale_target_inputs(target, origin_set)
                    scaled.add(target)
                    not_scaled.remove(target)

    def scale_target_inputs(self, target: str, origins: List[str]):
        """ Scale the selected origins layers, in order to concatenate
            inputs to target.

            Parameters
            ----------
            target : string
                Name of the input layer

            origins : list[string]
                Names of output layers
        """

        min_sz = self.get_min_input_size(target, origins)

        if 'in_channels' in self.descriptor.layers[target][1]:
            self.descriptor.layers[target][1]['in_channels'] = min_sz
        elif 'num_features' in self.descriptor.layers[target][1]:
            self.descriptor.layers[target][1]['num_features'] = min_sz
        elif 'in_feats' in self.descriptor.layers[target][1]:
            self.descriptor.layers[target][1]['in_feats'] = min_sz

        self.input_shapes[target] = min_sz

    def get_min_input_size(self, target: str, origins: List[str]):
        """ Get the minimum input size that enables concatenation.

            Parameters
            ----------
            target : string
                Name of the input layer

            origins : list[string]
                Names of output layers

            Returns
            -------
            min_dimensions : int
                Minimum size


        """

        min_dimensions = 0

        for node in origins:
            dimensions = self.get_output_size(node)
            min_dimensions += dimensions

        return min_dimensions

    def get_output_size(self, node: str):
        """Calculate the output size of a layer

        Parameters
        ----------
        node : string
            Name of the layer

        Returns
        -------
        The layer's output size

        """
        node_class = self.descriptor.layers[node][0]

        if node == self.first_layer:
            return self.input_shapes[node][0]

        if node_class is Flatten:
            return self.input_shapes[node]

        elif node_class is nn.Linear:
            return self.descriptor.layers[node][1]['out_features']

        elif node_class is nn.BatchNorm1d:
            return self.input_shapes[node]

        elif node_class in types_dict['activations']:
            return self.input_shapes[node]

        elif 'out_feats' in self.descriptor.layers[node][1]:
            return self.descriptor.layers[node][1]['out_feats']
        elif 'k' in self.descriptor.layers[node][1]:
            return self.descriptor.layers[node][1]['k'] * self.input_shapes[node]
        else:
            return self.input_shapes[node]

    def remove_connection(self, origin: str, target: str):
        """Remove a connection from the network

        Parameters
        ----------
        origin : string
            Name of the origin layer

        target : string
            Name of the target layer

        """

        while target in self.connections[origin]:
            self.connections[origin].remove(target)
        while origin in self.in_connections[target]:
            self.in_connections[target].remove(origin)

    def add_connection(self, origin: str, target: str):
        """Add a connection to the network

        Parameters
        ----------
        origin : string
            Name of the origin layer

        target : string
            Name of the target layer

        """
        if origin not in self.connections:
            self.connections[origin] = []
            self.in_connections[origin] = []
        if target not in self.connections:
            self.connections[target] = []
            self.in_connections[target] = []

        self.in_connections[target].append(origin)
        self.connections[origin].append(target)

    def __name_all_paths(self, layers_in: Dict[str, List],
                         acyclic: bool = True):
        """Find all the possible paths from the input layer to the output layer
           and name the nodes according to their depth.

        Parameters
        ----------
        layers_in : dict
            Contains the layer name as key. The dictionary's items
            are the layer's type and its parameters.

        acyclic : bool
            If true, only acyclic graphs are considered.

        Returns
        ----------

        A dictionary with updated names
        """

        layers = layers_in

        def set_layer_level(layer, level):
            new_key = str(level).zfill(3)+'_'+layer
            layers[new_key] = layers.pop(layer)
            self.connections[new_key] = self.connections.pop(layer)
            self.in_connections[new_key] = self.in_connections.pop(layer)

            if layer == self.first_layer:
                self.first_layer = new_key
            elif layer == self.last_layer:
                self.last_layer = new_key
            return new_key

        paths = None
        if acyclic:
            paths = self.get_acyclic_paths()
        else:
            paths = self.get_direct_paths()

        renamed = {}
        paths = sorted(paths, reverse=True, key=len)
        for path in paths:
            level = 1
            for node in path:
                if node not in renamed:
                    if node == self.last_layer:
                        level += 1
                    if node == self.first_layer:
                        level -= 1
                    new_node = set_layer_level(node, level)
                    renamed[node] = new_node

                else:
                    level = int(renamed[node].split('_')[0])
                level += 1

        for new_key in self.connections:
            for i in range(len(self.connections[new_key])):
                node = self.connections[new_key][i]
                if node in renamed:
                    self.connections[new_key][i] = renamed[node]

        for new_key in self.in_connections:
            for i in range(len(self.in_connections[new_key])):
                node = self.in_connections[new_key][i]
                if node in renamed:
                    self.in_connections[new_key][i] = renamed[node]
        return layers

    def to_networkx(self):
        """Create a networkx graph from the network

        Returns
        -------
        G : MultiDiGraph

        """

        G = nx.MultiDiGraph()
        for start in self.connections:
            ends = self.connections[start]
            for end in ends:
                G.add_edge(start, end)

        # names = {}
        # for layer in self.descriptor.layers:
        #     names[layer] = str(layer)+' '+
        #                    str(self.descriptor.layers[layer][1])

        # names['0_-2out'] = 'INPUTS'
        # H = nx.relabel.relabel_nodes(G, names)
        return G

    def get_direct_paths(self):
        """Return all the direct paths from the input to the output layer

        Returns
        -------
        list with all the direct paths
        """
        G = self.to_networkx()
        try:
            paths = nx.all_simple_paths(G, self.first_layer, self.last_layer)
        except nx.NodeNotFound:
            paths = [[]]
        return [p for p in paths]

    def get_acyclic_paths(self):
        """Return all the direct acyclic paths from the input to the output layer

        Returns
        -------
        list with all the direct paths
        """
        G = self.to_networkx()
        # recs = self.get_recursions()

        try:
            paths = nx.all_simple_paths(G, self.first_layer, self.last_layer)
        except nx.NodeNotFound:
            paths = [[]]

        rets = []
        for p in paths:
            if p not in rets:
                rets.append(p)
        return rets

    def get_recursions(self):
        """Return all the recursions in the network

        Returns
        -------
        list with all the recursions
        """
        G = self.to_networkx()
        cycles = nx.simple_cycles(G)
        return [sorted(c) for c in cycles]

    def plot(self, title: str = None):
        import matplotlib.pyplot as plt
        import numpy as np

        spacing_h = 0.5
        spacing_w = 0.1
        half_spacing = 0.25

        def my_layout(G, paths, recursions):
            nodes = G.nodes
            lengths = [-len(x) for x in paths]
            sorted_ = np.argsort(lengths)

            positions = dict()
            h = 0
            w = 0

            for index in sorted_:
                h = 0
                added = False
                path = paths[index]
                for node in path:
                    if node not in positions:
                        positions[node] = (w, h)
                        added = True
                        h -= spacing_h
                    else:
                        if h > positions[node][1]:
                            h = positions[node][1]

                if added:
                    if w >= 0:
                        w += spacing_w
                    w *= -1

            h = 0
            for node in nodes:
                if node not in positions:
                    positions[node] = (w, h)
                    h -= spacing_h

            f_l = self.first_layer
            l_l = self.last_layer
            if f_l in positions:
                positions[f_l] = (positions[f_l][0],
                                  positions[f_l][1]+spacing_h)
            if l_l in positions:
                positions[l_l] = (positions[l_l][0],
                                  positions[l_l][1]-spacing_h)

            recursed_nodes = []
            for path in recursions:
                last = sorted(path)[-1]
                if last not in recursed_nodes:
                    positions[last] = (positions[last][0]+half_spacing,
                                       positions[last][1])
                    recursed_nodes.append(last)
            return positions

        G = self.to_networkx()
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
        plt.figure()
        plt.title(title)
        in_path = self.get_direct_paths()
        recs = self.get_recursions()
        # pos = graphviz_layout(G, root='-2')
        pos = my_layout(G, in_path, recs)
        nx.draw(G, pos=pos, node_color='b',
                node_shape='s',
                node_size=800,
                with_labels=False,
                linewidths=0)

        nodes = set()
        for p in in_path:
            for node in p:
                nodes.add(node)
        for p in recs:
            for node in p:
                nodes.add(node)

        labels = {}

        for n in nodes:
            name = type(self.layers[n]).__name__
            labels[n] = name
            # s = n.split('_')
            # label = ''
            # if len(s) == 2:
            #     label = s[1]
            # else:
            #     label = s[2]
            # label = ''.join([i for i in label if not i.isdigit()])

            # label = label.replace('-', '')
            # if not label == 'scale':
            #     label = label.replace('scale', '')
            # labels[n] = label

        nx.draw_networkx_nodes(G, pos=pos,
                               node_color='w',
                               node_shape='s',
                               node_size=800,
                               nodelist=list(nodes),
                               with_labels=False,
                               linewidths=0)

        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

        plt.show()
