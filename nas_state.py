# Author: Aris Christoforidis.

import glob
import os
import sys
import pickle
import dill
import bz2
from copy import deepcopy
from tqdm import tqdm
from config import NAS_STATE_SAVE_BASE_PATH, CHUNK_SIZE
from enums import SaveMode
from evaluation import Evaluator
from module_manager import ModuleManager
from metrics import PerformanceSupervisor

class NasState:

    def __init__(self, name: str, evaluator: Evaluator, performance_supervisor: PerformanceSupervisor):
        """
        Holds references to all critical components of the search and saves them
        when needed.
        """
        self.name = name
        self.evaluator = evaluator
        self.performance_supervisor = performance_supervisor
        # Initialized after the first generation.
        self.module_manager = None
        self.population = None
        self.generation = 0
    
    def save(self, generation: int, population: list, module_manager: ModuleManager, mode: SaveMode, ovewrite: bool = True):
        """
        Saves the nas state.
        
        Parameters
        ----------
        generation: int
            The current generation.

        mode: SaveMode
            Determines the save method.

        overwrite: bool
            If True overwrites the old state.
        """
        # These need to be copies because their references are altered during a 
        # generation. We want the intact population and module_manager as is before
        # entering the next generation.
        self.population = deepcopy(population)
        self.module_manager = deepcopy(module_manager)
        if mode == SaveMode.PICKLE:
            self._save_pickle(generation, ovewrite)
        elif mode == SaveMode.CONSOLE_LOG:
            self._save_log(generation, ovewrite)
        print(f"Checkpoint saved(Generation {generation})")
    
    def _update_generation(self, generation):
        """
        Updates the current generation.

        Parameters
        ----------
        generation: int
            The current generation.
        """
        if self.generation >= generation: return
        self.generation = generation

    def _save_pickle(self, generation: int, overwrite: bool):
        """
        Saves the nas state using the pickle protocol.

        Parameters
        ----------
        generation: int
            The current generation.

        overwrite: bool
            If True overwrites the old state.
        """
        self._update_generation(generation+1)
        save_path = f"{NAS_STATE_SAVE_BASE_PATH}/nas_state_{self.name}_{self.generation}.dill"
        with open(save_path,'wb') as save_file:
            dill.dump(self, save_file)
            '''
            # TODO: Check if this a viable method of saving.
            state_bytes = pickle.dumps(self)
            for idx in tqdm(range(0,len(state_bytes),CHUNK_SIZE),desc='Saving state'):
                chunk = state_bytes[idx:idx+CHUNK_SIZE]
                save_file.write(chunk)
            '''
        # Delete old file.
        if overwrite == True and self.generation > 1:
            old_path = f"{NAS_STATE_SAVE_BASE_PATH}/nas_state_{self.name}_{self.generation-1}.dill"
            os.remove(old_path)
    
    def _save_log(self, generation: int, overwrite: bool):
        """
        Saves the nas state to a log file.

        Parameters
        ----------
        generation: int
            The current generation.

        overwrite: bool
            If True overwrites the old state.
        """

        # TODO: Implement.
        raise NotImplementedError

    @staticmethod
    def load(name: str, mode: SaveMode):
        """
        Loads the most recent nas state with the given name.

        Parameters
        ----------
        name: str
            The data name.

        mode: SaveMode
            Determines the save method.
        
        Returns
        -------
        nas_state: NasState
            The loaded nas state.
        """
        if mode == SaveMode.PICKLE:
            state = NasState._load_pickle(name)
        else:
            state = NasState._load_log(name)
        
        print(f"Loaded a nas state for {name}.(Generation {state.generation})")
        return state

    @staticmethod
    def _get_state_file_path(name: str, mode: SaveMode):
        """
        Gets the most recent state save file path.

        Parameters
        ----------
        name: str
            The data name.

        mode: SaveMode
            Determines the save method.
        
        Returns
        -------
        state_file_path: str
            The requested state file.
        """
        ext = '.dill' if mode == SaveMode.PICKLE else '.log'
        file_paths = glob.glob(f"{NAS_STATE_SAVE_BASE_PATH}/*")
        # Get all files that contain the name of the data set.
        file_paths = [path for path in file_paths if name in path and path.endswith(ext)]
        # Get the latest file path.
        try:
            latest_file_path = max(file_paths, key=os.path.getctime)
        except ValueError:
            raise Exception('A checkpoint file could not be loaded!')
        
        return latest_file_path

    @staticmethod
    def _load_pickle(name: str):
        """
        Loads a nas state from the most recent pickle file.
        
        Parameters
        ----------
        name: str
            The data name.

        Returns
        -------
        nas_state: NasState
            The loaded nas state.
        """
        file_path = NasState._get_state_file_path(name, SaveMode.PICKLE)
        with open(file_path,'rb') as state_file:
            state = dill.load(state_file)
        print(f"Loaded state from {file_path.split('/')[-1]}")
        return state

    @staticmethod
    def _load_log(name: str):
        """
        Loads a nas state from the most recent log file.
        
        Parameters
        ----------
        name: str
            The data name.

        Returns
        -------
        nas_state: NasState
            The loaded nas state.
        """
        raise NotImplementedError
