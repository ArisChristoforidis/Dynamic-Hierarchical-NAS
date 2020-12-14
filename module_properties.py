# Author: Aris Christoforidis

class ModuleProperties:

    def __init__(self, module_type, layer, abstract_graph, child_module_properties):
        self.module_type = module_type
        self.layer = layer
        self.abstract_graph = abstract_graph
        self.child_module_properties = child_module_properties
