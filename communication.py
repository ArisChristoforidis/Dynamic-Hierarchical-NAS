from mpi4py import MPI

class Communicator:


    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        print(self.rank)

    def _get_size(self):
        return self.comm.size