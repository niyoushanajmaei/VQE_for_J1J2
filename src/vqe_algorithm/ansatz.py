
class Ansatz:
    def __init__(self, N):
        self.circuit = None

    def _get_ansatz_w(self, N):
        """
        should return the ansatz circuit
        """
        pass

    def get_parameters(self) -> list:
        """
        should return a list of parameters of the ansatz
        """
        pass

    def update_parameters(self, new_parameters):
        """
        Should update the ansatz using the given new parameters
        """
        pass
