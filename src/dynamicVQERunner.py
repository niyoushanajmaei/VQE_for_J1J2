import numpy as np
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA, SLSQP
from qiskit.utils import algorithm_globals, QuantumInstance

from src.VQERunner import UnidentifiedAnsatzError, VQERunner, InvalidOptimizerError
from src.model import Model
from src.vqe_algorithm.ansatz import Ansatz
from src.vqe_algorithm.ansatze.FuelnerHartmannAnsatz import FuelnerHartmannAnsatz
from src.vqe_algorithm.ansatze.twoLocalAnsatz import TwoLocalAnsatz


class DynamicVQERunner:
    def __init__(self,  m, n, J1, J2, h=0, periodic_hamiltonian = False, ansatz="TwoLocal"):
        self.seed = 50
        if ansatz == "FeulnerHartmann":
            self.ansatz = FuelnerHartmannAnsatz(m * n)
        elif ansatz == "TwoLocal":
            self.ansatz = TwoLocalAnsatz(m * n)
        else:
            raise UnidentifiedAnsatzError
        self.m = m
        self.n = n
        self.N = m * n
        if periodic_hamiltonian:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_periodic(m, n, J1, J2, h=h)
        else:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_open(m, n, J1, J2, h=h)
        self.hamiltonianMatrix = self.hamiltonian.to_matrix()

        self.optimizer = SLSQP(maxiter=1000)

        self.simulation = True

    def run_dynamic_vqe(self, monitor=True, small_gradient_deletion=False, small_gradient_add_to_end=False,
                        random_pseudo_removal=False, add_layers_fresh=False, add_layers_duplicate=False,
                        large_gradient_add=False):
        """
        Runs the VQE algorithm

        :param monitor: if True, the convergence plot is saved
        :param small_gradient_deletion: if Ture, "alpha" of percent gates with the smallest parameter gradient are
            removed from that layer
        :param small_gradient_add_to_end: if True, "alpha" percent of gates with the smallest parameter gradient are
            added removed and added to the end of the ansatz instead
        :param large_gradient_add: if True, "beta" percent of gates with the largest parameter gradient are either
            duplicated in place, or added to the end of the ansatz
        :param add_layers_duplicate: if True, a duplicate layer of the last layer is added every "step_iter"
            iterations
        :param add_layers_fresh: if True, an uninitialized layer is added every "step_iter" iterations
        :param random_pseudo_removal: if True, at each iteration, a randomly chosen "gamma" percent of the gates are
            temporarily removed from the ansatz

        """
        # variables for how often the dynamic actions should be done
        step_iter_small_gradient = 10
        step_iter_random_pseudo_removal = 10
        step_iter_large_gradient = 10
        step_iter_add_layer = 20

        seed = self.seed
        algorithm_globals.random_seed = seed
        qi = QuantumInstance(VQERunner.get_backend(simulate=self.simulation), seed_transpiler=seed, seed_simulator=seed)

        ansatz = self.ansatz.circuit

        if monitor:
            counts = []
            values = []

            def store_intermediate_results(evalCount, parameters, mean, std):
                counts.append(evalCount)
                values.append(mean)
                print(f"Current Estimated Energy: {mean}")

            vqe = VQE(ansatz, optimizer=self.optimizer, callback=store_intermediate_results, quantum_instance=qi, include_custom=True)
        else:
            vqe = VQE(ansatz, optimizer=self.optimizer, quantum_instance=qi, include_custom=True)

        result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)

        if monitor:
            counts = [np.asarray(counts)]
            values = [np.asarray(values)]
            optimizers = [self.optimizer]
            VQERunner.plotConvergences(counts, values, optimizers)

        return result


def add_layers_duplicate(initial_ansatz: Ansatz) -> Ansatz:
    """
    Adds another layer to the end of the ansatz. The added ansatz is a duplicate of the last layer of the initial ansatz,
    Both parameter-wise and gate wise.

    :param initial_ansatz: the initial ansatz

    :return: the final ansatz
    """



