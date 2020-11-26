# Author: Aris Christoforidis.

from neural_module import NeuralModule
from evaluation import Evaluator, NasBenchEvaluator
from communication import Communicator
from mpi4py import MPI


def main():
    # Establish communication.
    communicator = Communicator()
    
    # Switch to this to use nasbench.
    #evaluator = NasBenchEvaluator()
    evaluator = Evaluator()
    n = NeuralModule(1,evaluator)
    # Do some example mutations.
    n.mutate()
    n.mutate()
    n.mutate()
    n.mutate()
    
    print(f"Size: {communicator._get_size()}")
    return

if __name__ == "__main__":
    main()