# Author: Aris Christoforidis

from enums import ModuleType
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx import is_isomorphic
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
        self.complexity = total_nodes + total_edges
        self.cached_hash = None

    def __eq__(self, other):
        """
        Equality comparator for the module properties object.

        Parameters
        ----------
        other: ModuleProperties
            A ModuleProperties object.

        Returns
        -------
        eq: bool
            True if objects are equal, False, otherwise.
        """
        # Occurs at the first comparison of the best module on module_manager.
        if other == None: return False
        
        modules_equal = self.module_type == other.module_type
        layer_equal = self.layer == other.layer
        if self.module_type == ModuleType.NEURAL_LAYER: return modules_equal and layer_equal
        graph_equal = is_isomorphic(self.abstract_graph, other.abstract_graph)
        children_equal = len(self.child_module_properties) == len(other.child_module_properties) and sum([x == y for x,y in zip(self.child_module_properties, other.child_module_properties)]) == len(self.child_module_properties)
        return modules_equal and layer_equal and graph_equal and children_equal

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
