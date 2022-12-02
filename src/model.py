import qiskit.opflow


class Model:

    @staticmethod
    def get_hamiltonian(size):
        if size <= 1:
            raise InvalidSizeError
        #TODO


class InvalidSizeError(RuntimeError):
    """ Lattice Size should be n>1 """
    pass