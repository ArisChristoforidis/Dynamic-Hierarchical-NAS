# Author: Aris Christoforidis.

from nas_state import NasState
from enums import SaveMode
from config import BEST_NETWORK_LABEL, BEST_NETWORK_SCORE_LABEL
import dill
from enums import ModuleType
import networkx as nx
import matplotlib.pyplot as plt

BASE_PATH = "saved_checkpoints"
FOLDER_PATH = "activity_recognition"
CHECKPOINT_PATH = "nas_state_activity_recognition_21_laptop_hierarchical.dill"

def main():
    with open(f"{BASE_PATH}/{FOLDER_PATH}/{CHECKPOINT_PATH}",'rb') as state_file:
        state = dill.load(state_file)

    best_network_data = state.performance_supervisor.best_networks
    for generation, data in best_network_data.items():
        module =  data[BEST_NETWORK_LABEL]
        accuracy = data[BEST_NETWORK_SCORE_LABEL]
        print(f"Generation: {generation} | Complexity: {module.module_properties.complexity} | Accuracy: {accuracy:.4f}")
        module.show_full_graph()
        sub_modules = [module]
        while len(sub_modules) > 0:
            m = sub_modules.pop()
            print(f"{m.module_type} | {m.layer} | {m.module_properties.cached_hash}")
            print("-" * 50)
            if m.module_type == ModuleType.ABSTRACT_MODULE:
                children = list(m.child_modules.values())
                sub_modules.extend(children)
                indices = 0
                layer_names = {}
                layer_names[indices] = "input"
                for k,v in m.child_modules.items():
                    print(f"{k}: {v.module_type} | {v.layer} | {v.module_properties.cached_hash}")
                    indices += 1
                    layer_names[k+1] = f"{v.layer} | {v.module_properties.cached_hash}"
                
                indices += 1
                layer_names[indices] = "output"

                g = m.abstract_graph

                nx.draw_spring(g,with_labels=True,labels=layer_names)
                plt.show()


    
if __name__ == "__main__":
    main()