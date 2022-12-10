from qiskit import Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.utils import QuantumInstance

from src.vqe_algorithm.ansatze.FuelnerHartmannAnsatz import FuelnerHartmannAnsatz
from src.vqe_algorithm.ansatze.twoLocalAnsatz import TwoLocalAnsatz
from src.model import Model
from src.vqe_algorithm.haminlonian_expectator import HamiltonianExpectator
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
            self.ansatz = FuelnerHartmannAnsatz(self.N)
        elif ansatz == "TwoLocal":
            self.ansatz = TwoLocalAnsatz(self.N)
        else:
            raise UnidentifiedAnsatzError

        if open_bound:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_open(m, n, J1, J2, h=h)
        else:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_periodic(m, n, J1, J2, h=h)
        self.hamiltonianMatrix = self.hamiltonian.to_matrix()

        self.optimizer = CustomOptimizer()
        self.hamiltonian_expectator = HamiltonianExpectator()

        self.simulation = simulation

    def run_vqe(self, iterations: int):
        """
        wrapper class for running the vqe algorithm

        :param iterations: number of iterations of the algorithm

        :return:
        """
        seed = 500
        qi = QuantumInstance(self.get_backend(simulate=self.simulation), seed_transpiler=seed, seed_simulator=seed)

        for i in range(iterations):
            hamiltonian_expectation = self.hamiltonian_expectator.get_hamiltonian_expectation(qi, self.ansatz)
            new_parameters = self.optimizer.optimize(self.ansatz.get_parameters(), hamiltonian_expectation)
            self.ansatz.update_parameters(new_parameters)


    def get_backend(self, simulate: bool = True):
        """
        returns an Aer simulator if simulate = True
        and an IBMQ backend if simulate = False

        In case simulate is False, it is assumed that the user has stored their IBM Quantum account
        information locally ahead of time using IBMQ.save_account(TOKEN).
        TOKEN here is the API token you obtain from your IBM Quantum account.
        """
        if simulate:
            backend = Aer.get_backend('aer_simulator')
        else:
            IBMQ.load_account()  # Load account from disk
            provider = IBMQ.get_provider(hub='ibm-q')
            filtered_backends = provider.backends(filters=lambda x: not x.configuration().simulator
                                                                    and x.status().operational)
            backend = least_busy(filtered_backends)
        return backend


class UnidentifiedAnsatzError(RuntimeError):
    """
    Raised when the given ansatz name is unidentified
    """
    pass
