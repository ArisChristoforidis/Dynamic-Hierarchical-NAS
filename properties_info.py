# Author: Aris Christoforidis.
from config import TEMP_MODULE_TTL

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
        # Recalculate the average fitness.
        self.average_fitness = (self.get_total_fitness() + fitness) / (self.occurence_count + 1)
        self.occurence_count += 1

class TempPropertiesInfo(PropertiesInfo):

    def __init__(self, complexity):
        super().__init__()

        self.time_to_leave = TEMP_MODULE_TTL * complexity
    
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
        