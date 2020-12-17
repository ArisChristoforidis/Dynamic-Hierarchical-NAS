# Author: Aris Christoforidis

from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

class ModuleProperties:

    def __init__(self, module_type, layer, abstract_graph, child_module_properties):
        self.module_type = module_type
        self.layer = layer
        self.abstract_graph = abstract_graph
        self.child_module_properties = child_module_properties

    def __hash__(self):
        """
        Hashes the module properties object.

        Returns
        -------
        hash: int
            The integer hash of the object.
        """
        # Hash the abstract graph.
        abstract_graph_hash = weisfeiler_lehman_graph_hash(self.abstract_graph)
        # Get the hashes of the children.
        child_module_hashes = [hash(child_properties) for child_properties in self.child_module_properties]            
        # Create a list with the attributes of self and children and convert it to a tuple to make it hashable.
        attribute_container = [self.module_type, self.layer, abstract_graph_hash].extend(child_module_hashes)
        attribute_container = tuple(attribute_container)
        return hash(attribute_container)