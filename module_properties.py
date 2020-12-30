# Author: Aris Christoforidis

from enums import ModuleType
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from config import TEMP_MODULE_TTL

class ModuleProperties:

    def __init__(self, module_type, layer, abstract_graph, child_module_properties):
        self.module_type = module_type
        self.layer = layer
        self.abstract_graph = abstract_graph
        self.child_module_properties = child_module_properties
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
                self.cached_hash = hash(self.module_type)
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


class PropertiesInfo:
    
    def __init__(self, temp_info = None):
        if temp_info == None:
            self.occurence_count = 0
            self.average_fitness = 0
        else:
            self.occurence_count = temp_info.occurence_count
            self.average_fitness = temp_info.average_fitness

    def get_total_fitness(self):
        return self.occurence_count * self.average_fitness

    def record(self, fitness):
        """
        Records a new fitness observation from a neural module.

        Parameters
        ----------
        fitness: The neural module fitness.
        """
        self.occurence_count += 1
        # Recalculate the average fitness.
        self.average_fitness = (self.get_total_fitness() + fitness) / self.occurence_count

class TempPropertiesInfo(PropertiesInfo):

    def __init__(self):
        super().__init__()
        self.time_to_leave = TEMP_MODULE_TTL
    
    def on_generation_increase(self):
        """
        Call this when a generation change(increase) occurs.

        Returns
        -------
        delete: bool
            True if the module should be deleted because it stayed in the temp
            list too long(TTL expired), False otherwise.
        """
        if self.time_to_leave > 0:
            self.time_to_leave -= 1
        return self.time_to_leave == 0
        