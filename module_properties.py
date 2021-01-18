# Author: Aris Christoforidis

from enums import ModuleType
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from config import TEMP_MODULE_TTL

class ModuleProperties:

    def __init__(self, module_type, layer, abstract_graph, child_module_properties, total_nodes, total_edges):
        self.module_type = module_type
        self.layer = layer
        self.abstract_graph = abstract_graph
        self.child_module_properties = child_module_properties
        # These are only used to calculate the complexity of the graph described 
        # by the module properties object.
        self.total_nodes = total_nodes
        self.total_edges = total_edges
        self.cached_hash = None

    def __hash__(self):
        """
        Hashes the module properties object.

        Returns
        -------
        hash: int
            The integer hash of the object.
        """
        if self.cached_hash == None:
            if self.module_type == ModuleType.NEURAL_LAYER:
                attribute_container = (self.module_type, self.layer)
            else:
                # Hash the abstract graph.
                abstract_graph_hash = weisfeiler_lehman_graph_hash(self.abstract_graph)
                # Get the hashes of the children.
                child_module_hashes = [hash(child_properties) for child_properties in self.child_module_properties]            
                # Create a list with the attributes of self and children and convert it to a tuple to make it hashable.
                attribute_container = [self.module_type, self.layer, abstract_graph_hash]
                attribute_container.extend(child_module_hashes)
                attribute_container = tuple(attribute_container)
            # Get the hash and cache it.
            self.cached_hash = hash(attribute_container)
        
        return self.cached_hash
