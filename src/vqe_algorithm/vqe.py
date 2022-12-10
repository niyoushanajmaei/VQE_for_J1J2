from src.vqe_algorithm.ansatze.FuelnerHartmannAnsatz import FuelnerHartmannAnsatz
from src.vqe_algorithm.ansatze.twoLocalAnsatz import TwoLocalAnsatz
from src.model import Model
from src.vqe_algorithm.optimizers.custom_optimizer import CustomOptimizer

class VQE:
    def __init__(self, m, n, J1, J2, h=0, simulation=True, ansatz="FuelnerHartmann", open_bound=True):
        """
        :param m, n : mxn lattice
        :param J1, J2: nearest neighbour and next nearest neighbor interactions respectively
        :param h: magnetic field, taken to be 0 for now
        :param ansatz: can be one of the following: "FuelnerHartmann", "TwoLocal"
        :param simulation: if True, run on simulation backend, otherwise, on quantum backend
        :param open_bound: if Ture, the lattice is open bound, otherwise it's periodic bound

        :raise raises UnidentifiedAnsatzError
        """

        self.m = m
        self.n = n
        self.N = m * n

        if ansatz == "FeulnerHartmann":
            self.ansatz = FuelnerHartmannAnsatz()
        elif ansatz == "TwoLocal":
            self.ansatz = TwoLocalAnsatz()
        else:
            raise UnidentifiedAnsatzError

        if open_bound:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_open(m, n, J1, J2, h=h)
        else:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_periodic(m, n, J1, J2, h=h)
        self.hamiltonianMatrix = self.hamiltonian.to_matrix()

        self.optimizer = CustomOptimizer()

        self.simulation = simulation


class UnidentifiedAnsatzError(RuntimeError):
    """
    Raised when the given ansatz name is unidentified
    """
    pass
