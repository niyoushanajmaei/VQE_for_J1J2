
from src.vqe_algorithm.ansatz import Ansatz

class FuelnerHartmannAnsatz(Ansatz):
    def __init__(self):
        self.circuit = self._get_ansatz()

    def _get_ansatz(self):
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
