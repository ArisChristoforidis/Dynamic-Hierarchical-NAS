from enum import Enum

class ConnectMode(Enum):
    IN = 0,
    OUT = 1

class ModuleType(Enum):
    NEURAL_LAYER = 0,
    ABSTRACT_MODULE = 1


class SaveMode(Enum):
    PICKLE = 0,
    CONSOLE_LOG = 1
    

