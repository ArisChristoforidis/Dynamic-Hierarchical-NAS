# Author: Aris Christoforidis

class ModuleProperties:

    def __init__(self, module_type, layer, abstract_graph, child_module_properties):
        self.module_type = module_type
        self.layer = layer
        self.abstract_graph = abstract_graph
        self.child_module_properties = child_module_properties

    def __hash__(self):
        # TODO: Pass all graphs, use weisfeiler_lehman_graph_hash to get graph strings. 
        for child_node in self.abstract_graph:
            continue
        # TODO: Use the rest of the info to make a tuple and then hash it.