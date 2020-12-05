import random as rnd
import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
from evaluation import Evaluator
from enums import ConnectMode, ModuleType
from nord.neural_nets import NeuralDescriptor
from config import NODE_INPUT_TAG, NODE_OUTPUT_TAG, UNEVALUATED_FITNESS, NODE_INTERNAL_COUNT_RANGE

class NeuralModule:

    def __init__(self, depth: int, evaluator: Evaluator):
        # Constructor variables.
        self.depth = depth
        self.evaluator = evaluator
        self.fitness = UNEVALUATED_FITNESS
        self.module_type = ModuleType.NEURAL_LAYER
        
        # Randomize seed.
        self.random_seed = rnd.randint(0,100)
        self.child_count = rnd.choice(NODE_INTERNAL_COUNT_RANGE)

        # Assign layer to module.
        self._assign_layer()

        if self.depth == 1: self.change_module_type(ModuleType.ABSTRACT_MODULE)

    def _create_children(self):
        """ Randomly assign neural layers to the network graph. """
        self.child_modules = {child_idx : NeuralModule(self.depth + 1, self.evaluator) for child_idx in range(self.child_count)}
        return

    def _assign_layer(self):
        """ Assign a layer to this module. """
        available_layers = self.evaluator.get_available_layers()
        self.layer = rnd.choice(available_layers)

    def _init_abstract_graph(self):
        """ Create a random graph for this module. """
        self.abstract_graph = nx.DiGraph()

        # Internal nodes.
        self.abstract_graph.add_nodes_from(range(self.child_count))

        nodes = set(self.abstract_graph.nodes())

        input_k = rnd.randint(1, self.child_count)
        input_nodes = rnd.sample(nodes,k=input_k)

        output_k = rnd.randint(1, self.child_count)
        output_nodes = rnd.sample(nodes,k=output_k)

        input_edges = [(NODE_INPUT_TAG, idx) for idx in input_nodes]
        output_edges = [(idx, NODE_OUTPUT_TAG) for idx in output_nodes]

        # Add input & output nodes.
        self.abstract_graph.add_nodes_from([NODE_INPUT_TAG, NODE_OUTPUT_TAG])
        
        # Connect internal nodes to the input & output.
        self.abstract_graph.add_edges_from(input_edges)
        self.abstract_graph.add_edges_from(output_edges)

        # Get nodes with no incoming connections.
        no_input_nodes = [k for k,v in self.abstract_graph.in_degree() if v == 0 and k != NODE_INPUT_TAG]
        no_output_nodes = [k for k,v in self.abstract_graph.out_degree() if v == 0 and k != NODE_OUTPUT_TAG]
        for node in no_input_nodes: self._add_node_edges(node, 1, False, ConnectMode.IN)
        for node in no_output_nodes: self._add_node_edges(node, 1, False, ConnectMode.OUT)
        
    def _add_node_edges(self, node: int, count: int, use_external: bool, mode: ConnectMode):
        """
        Adds a number of edges between the given node and a random node.

        Parameters
        ----------
        node: int
            The node ID.
        
        count: int
            The number of edges to add. This is restricted to the max number of
            nodes that can be connected to the given node.
        
        use_external: bool
            Whether to use input/output nodes.
        
        mode: ConnectMode
            Should the edges be incoming or outgoing?
        """
        nodes = set(self.abstract_graph.nodes())
        nodes.remove(node)
        if use_external == False or mode == ConnectMode.IN: nodes.remove(NODE_INPUT_TAG)
        if use_external == False or mode == ConnectMode.OUT: nodes.remove(NODE_OUTPUT_TAG)

        # Restrict maximum count.        
        count = min(count, len(nodes))

        nodes_to_connect = rnd.sample(nodes,k=count)
        if mode == ConnectMode.IN:
            edges = [(input_node, node) for input_node in nodes_to_connect]
        else:
            edges = [(node, output_node) for output_node in nodes_to_connect]

        self.abstract_graph.add_edges_from(edges)

    # NOTE: Not used.
    def _remove_graph_cycles(self):
        """ Removes cycles from a graph by randomly deleting edges that belong in a cycle. """
        edges_removed = 0
        while True:
            cycles = list(nx.simple_cycles(self.abstract_graph))
            if cycles == []: break
            cycle = cycles[0]
            # Pick a random edge.
            random_idx = rnd.randint(0, len(cycle) - 2)
            edge_start = cycle[random_idx]
            edge_end = cycle[random_idx + 1]
            # Remove it.
            self.abstract_graph.remove_edge(edge_start, edge_end)
            edges_removed += 1
            print(f"Removed ({edge_start},{edge_end}) edge.")

        print(f"Removed {edges_removed} edge(s).")

    def _graph_has_cycles(self, graph):
        """ 
        Checks if a graph has (simple) cycles. 
        
        Parameters
        ----------
        graph: nx.Digraph
            A networkx graph.

        Returns
        -------
        has_cycles: bool
            Whether or not the abstract graph has (simple) cycles. 
        """
        return len(list(nx.simple_cycles(graph))) > 0

    def mutate_node(self):
        """
        Performs mutation by finding a neural node in the graph and converting
        it to an abstract module.
        """
        selected_node = rnd.choice(self.child_modules)
        if selected_node.module_type == ModuleType.ABSTRACT_MODULE:
            selected_node.mutate_node()
        elif selected_node.module_type == ModuleType.NEURAL_LAYER:
            selected_node.change_module_type(ModuleType.ABSTRACT_MODULE)

    def mutate_connection(self):
        """
        Performs a mutation by finding adding an edge to an abstract graph of a
        random depth in the module.
        
        Returns
        -------
        success: bool
            Whether the operation was successfull or not.
        """
        # Get all nodes in the abstract graph that are abstract modules.
        abstract_nodes = [node for node in self.child_modules.values() if node.module_type == ModuleType.ABSTRACT_MODULE]
        can_add_edge_to_self = True
        while True:
            visit_child = rnd.random() < 0.5
            # Check that the randomness decided we will be visiting a child(alternatively, 
            # we can't add an an edge to this layer) & there are children to visit.
            if (visit_child == True or can_add_edge_to_self == False) and len(abstract_nodes) > 0:
                # Select a child code.
                selected_node = rnd.choice(abstract_nodes)
                child_success = selected_node.mutate_connection()
                # If we could not add an edge to this part of the graph, remove 
                # it as a choice.
                if child_success == False:
                    abstract_nodes.remove(selected_node)
                else:
                    # Edge connection successful.
                    return True                
            elif can_add_edge_to_self == True:
                # Add the edge on this module's abstract graph.
                # Get all possible edges.
                possible_edges = list(it.product(self.abstract_graph.nodes,repeat=2))
                # Remove all existing edges.
                possible_edges = [edge for edge in possible_edges if edge not in self.abstract_graph.edges()]
                # Remove all the self loops.
                possible_edges = [(source, dest) for source, dest in possible_edges if source != dest]
                # Remove all edges that start from the output or end in the input.
                possible_edges = [(source, dest) for source, dest in possible_edges if source != NODE_OUTPUT_TAG and dest != NODE_INPUT_TAG]
                # TODO: Decide whether an (INPUT, OUTPUT) edge is desirable.
                # If there are no possible edges to add to the graph, quit.
                if possible_edges == []: can_add_edge_to_self = False
                while len(possible_edges) > 0:
                    # Get a random edge and add it to the abstract graph,testing it for cycles.
                    source,dest = rnd.choice(possible_edges)
                    temp_graph = self.abstract_graph.copy()
                    temp_graph.add_edge(source, dest)
                    # If this has cycles, remove it from the possible edges list.
                    if self._graph_has_cycles(temp_graph) == True:
                        possible_edges.remove((source,dest))
                    else:
                        # Set the abstract graph with the new connection.
                        self.abstract_graph = temp_graph
                        return True
                # We did not manage to add an edge.
                can_add_edge_to_self = False
            else:
                # An edge cannot be added at any point in the graph.
                return False
    
    def change_module_type(self, new_type: ModuleType):
        """
        Changes the module's type.

        Parameters
        ----------
        new_type: ModuleType
            The new module type.
        """
        if (self.module_type == new_type): return
        self.module_type = new_type
        if new_type == ModuleType.ABSTRACT_MODULE:
            self._create_children()
            # Try to create the abstract graph, until a random topology with no 
            # cycles is made.
            while True:
                self._init_abstract_graph()
                if self._graph_has_cycles(self.abstract_graph) == False: break
                
    def mutate(self):
        """ Perform mutation. """
        self.mutate_node()
        _ = self.mutate_connection()

    def show_net(self):
        """ Draws the neural network. """
        nx.draw_spring(self.abstract_graph, with_labels=True, labels = self.child_modules)
        plt.show()
    
    def get_graph(self):
        """
        Iterate through all children, getting the subgraphs and create the full graph
        for this node.

        Returns
        -------
        full_graph: nx.Digraph
            A networkx directed graph.

        layer_names: dict(int->str)
            A dictionary containing the layer names(types).
        
        input_idx: int
            The index of the input node for the graph.

        output_idx: int
            The index of the output node for the graph.        
        """
        full_graph = nx.DiGraph()
        layer_names = {}
        # We use this index to add nodes to the full graph.
        full_graph_idx = 0
        # This holds the associations between each abstract node and its input
        # output in the smaller graphs.
        subgraph_connections_dict = {}
        # Add the input and output nodes manually.
        subgraph_connections_dict[NODE_INPUT_TAG] = {NODE_INPUT_TAG : -1, NODE_OUTPUT_TAG : full_graph_idx}
        input_idx = full_graph_idx
        full_graph.add_node(input_idx)
        layer_names[full_graph_idx] = NODE_INPUT_TAG
        full_graph_idx += 1
        
        subgraph_connections_dict[NODE_OUTPUT_TAG] = {NODE_INPUT_TAG : full_graph_idx, NODE_OUTPUT_TAG : -1}
        output_idx = full_graph_idx
        full_graph.add_node(full_graph_idx)
        layer_names[full_graph_idx] = NODE_OUTPUT_TAG
        full_graph_idx += 1
        
        for child_idx, child in self.child_modules.items():
            # Extract this child's graph and attach it to the corresponding node
            # of this graph.
            if child.module_type == ModuleType.ABSTRACT_MODULE:
                child_graph, child_layer_names, child_input_idx, child_output_idx = child.get_graph()
                # Extract nodes.
                child_nodes = set(child_graph.nodes())
                child_to_full_node_idx_dict = {}
                for child_node in child_nodes:
                    # Save the relation between the child node and its node on the
                    # full graph.
                    child_to_full_node_idx_dict[child_node] = full_graph_idx
                    # Also get the layer name for each node.
                    layer_names[full_graph_idx] = child_layer_names[child_node]
                    full_graph_idx += 1
                    # Add nodes to full graph.
                    full_graph.add_nodes_from(child_to_full_node_idx_dict.values())
                
                # Extract edges.
                child_edges = set(child_graph.edges())
                full_edges = []
                for child_source,child_dest in child_edges:
                    full_source = child_to_full_node_idx_dict[child_source]
                    full_dest = child_to_full_node_idx_dict[child_dest]
                    full_edges.append((full_source, full_dest))
                # Add child edges to full graph.
                full_graph.add_edges_from(full_edges)

                # Register subgraph input and output nodes in order to connect it
                # with the other subgraphs.
                subgraph_connections_dict[child_idx] = {
                    NODE_INPUT_TAG  : child_to_full_node_idx_dict[child_input_idx],
                    NODE_OUTPUT_TAG : child_to_full_node_idx_dict[child_output_idx]
                }
                child_layer_names[child_to_full_node_idx_dict[child_input_idx]] = NODE_INPUT_TAG
                child_layer_names[child_to_full_node_idx_dict[child_output_idx]] = NODE_OUTPUT_TAG

            elif child.module_type == ModuleType.NEURAL_LAYER:
                # Just assign a new index to this node for the full graph and register
                # the input and output.
                new_node_idx = full_graph_idx
                layer_names[full_graph_idx] = child.layer
                full_graph_idx += 1
                subgraph_connections_dict[child_idx] = {
                    NODE_INPUT_TAG  : new_node_idx,
                    NODE_OUTPUT_TAG : new_node_idx
                }

        # Now connect all the subgraphs together.
        external_edges = []
        delete_keys = set()
        for abstract_source, abstract_dest in set(self.abstract_graph.edges()):
            full_source_dict = subgraph_connections_dict[abstract_source]
            full_dest_dict = subgraph_connections_dict[abstract_dest]

            # Find the node indices in the full graph.
            source_output_node = full_source_dict[NODE_OUTPUT_TAG]
            dest_input_node = full_dest_dict[NODE_INPUT_TAG]

            output_nodes = [source_output_node]
            input_nodes = [dest_input_node]

            # If the output node is part of a complex graph, get the in edges
            # and remove the artificial node by connecting the edges that lead
            # to this node with the dest_input_node.
            if NODE_OUTPUT_TAG in layer_names[source_output_node]:
                in_edges = full_graph.in_edges(source_output_node)
                output_nodes = [start for start,_ in in_edges]
                # Mark node for deletion.
                delete_keys.add(source_output_node)
            
            # If the input node is part of a complex graph, get the out edges
            # and remove the artificial node by connecting the edges that start
            # from this node with the source_output_node.
            if NODE_INPUT_TAG in layer_names[dest_input_node]:
                out_edges = full_graph.out_edges(dest_input_node)
                input_nodes = [end for _,end in out_edges]
                # Mark node for deletion.
                delete_keys.add(dest_input_node)

            # Create the proper connections.
            external_connections = list(it.product(output_nodes,input_nodes))
            external_edges.extend(external_connections)
        
        # Use the external edges to connect internal nodes of different subgraphs.
        full_graph.add_edges_from(external_edges)
        for key in set(delete_keys): 
            full_graph.remove_node(key)
            layer_names.pop(key)

        if self.depth == 1:
            nx.draw_spring(full_graph,with_labels=True,labels=layer_names)
            plt.show()

        return full_graph, layer_names, input_idx, output_idx

    def to_descriptor(self):
        """
        Creates the descriptor object that represents the network module.

        Returns
        -------
        descriptor: NeuralDescriptor
            A descriptor object.
        """
        # Create descriptor.
        descriptor = NeuralDescriptor()
        # Get full graph data.
        full_graph, layer_types, input_idx, output_idx = self.get_graph()
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
