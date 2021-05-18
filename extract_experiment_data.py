import dill
from enums import SaveMode
from nas_state import NasState

BASE_PATH = "saved_checkpoints"
FOLDER_PATH = "activity_recognition"
CHECKPOINT_PATH = "nas_state_activity_recognition_21_laptop_hierarchical.dill"

def main():
    with open(f"{BASE_PATH}/{FOLDER_PATH}/{CHECKPOINT_PATH}",'rb') as state_file:
        state = dill.load(state_file)

    supervisor = state.performance_supervisor
    print(supervisor.average_fitness)
    return 1

if __name__ == "__main__":
    main()