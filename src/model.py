import qiskit.opflow


class Model:

    @staticmethod
    def get_hamiltonian(lattice_size):
        size = lattice_size[0]*lattice_size[1]
        if size <= 1:
            raise InvalidSizeError
        #TODO


class InvalidSizeError(RuntimeError):
    """ Lattice Size should be n>1 """
    pass