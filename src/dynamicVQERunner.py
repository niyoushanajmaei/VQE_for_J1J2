import numpy as np
import matplotlib.pyplot as plt

from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA, ADAM
from qiskit.utils import algorithm_globals, QuantumInstance

from src.VQERunner import UnidentifiedAnsatzError, VQERunner, InvalidOptimizerError
from src.model import Model
from src.vqe_algorithm.ansatze.FuelnerHartmannAnsatz import FuelnerHartmannAnsatz
from src.vqe_algorithm.ansatze.twoLocalAnsatz import TwoLocalAnsatz

from time import localtime, strftime

# TODO:
# - add offsets to the ansatz modifiers, to prevent them from running together?
#       But this would require stopping the optimizer more often...
#       or convert to a probability based model for modifications


class DynamicVQERunner:
    def __init__(self,  m, n, J1, J2, h=0, periodic_hamiltonian = False, ansatz="TwoLocal", optimizer="SLSQP", totalMaxIter = 1000):
        self.seed = 50
        if ansatz == "FeulnerHartmann":
            self.ansatz = FuelnerHartmannAnsatz(m * n)
        elif ansatz == "TwoLocal":
            self.ansatz = TwoLocalAnsatz(m * n)
        else:
            raise UnidentifiedAnsatzError

        self.optimizer = optimizer
        self.m = m
        self.n = n
        self.N = m * n
        if periodic_hamiltonian:
            self.periodicity = "periodic"
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_periodic(m, n, J1, J2, h=h)
        else:
            self.periodicity = "open"
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_open(m, n, J1, J2, h=h)
        self.hamiltonianMatrix = self.hamiltonian.to_matrix()
        self.simulation = True
        self.totalMaxIter = totalMaxIter

    def run_dynamic_vqe(self, small_gradient_deletion=False, small_gradient_add_to_end=False,
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

        # number of iterations before stopping the optimizer for modifications
        step_iter = 10
        step_iter = min(step_iter, self.totalMaxIter)

        seed = self.seed
        algorithm_globals.random_seed = seed
        qi = QuantumInstance(VQERunner.get_backend(simulate=self.simulation), seed_transpiler=seed, seed_simulator=seed)

        ansatz = self.ansatz.circuit
        initialTheta = self.ansatz.theta
        finalTheta = self.ansatz.theta

        if self.optimizer == "SLSQP":
            opt = SLSQP(maxiter=step_iter)
        elif self.optimizer == "SPSA":
            # TODO for SPSA, investigate the usage of a user defined gradient
            opt = SPSA(maxiter=step_iter)
        elif self.optimizer == "AMSGRAD":
            lr = 0.009
            opt = ADAM(maxiter=step_iter, lr=lr, amsgrad=True)
        else:
            raise InvalidOptimizerError
        print(f"Using {self.optimizer} optimizer with {self.totalMaxIter} total iterations, stopping every {step_iter} iterations for modifications")

        counts = []
        values = []

        def store_intermediate_results(evalCount, parameters, mean, std):
            counts.append(evalCount)
            values.append(mean)
            print(f"Current Estimated Energy: {mean}")

        for i in range(0, self.totalMaxIter, step_iter):
            vqe = VQE(ansatz, optimizer=opt, initial_point=initialTheta, callback=store_intermediate_results, quantum_instance=qi, include_custom=True)
            result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)
            print("optimizer stopped here")
            finalTheta = result.optimal_point.tolist()
            # do modifications?
            # still have initial theta at this stage, can process finalTheta and initialTheta
            if initialTheta:
                pass
            initialTheta = finalTheta
            self.ansatz.update_parameters(finalTheta)

        # save convergence plot for the run
        counts = [np.asarray(counts)]
        values = [np.asarray(values)]
        optimizers = [self.optimizer]
        fileName = f"{self.periodicity}-{self.m}x{self.n}-{self.ansatz}-{self.optimizer}-{self.totalMaxIter}iters"
        self.plotConvergences(counts, values, optimizers, fileName=fileName)

        return result

    def plotConvergences(self, counts, values, optimizers, fileName="convergenceGraph.png"):
        """
        plots the convergence plots for a list of counts and values

        :param: counts and values should be a list of np arrays
            optimizers is a list of the name of the optimizers corresponding to each set of counts and values
        """
        plt.figure(figsize=(12, 8))
        for i, optimizer in enumerate(optimizers):
            plt.plot(values[i], label=optimizer)

        # plotting exact value
        plt.axhline(y=Model.getExactEnergy(self.hamiltonianMatrix), color='r', linestyle='-', label="exact energy")

        plt.xlabel('Eval count')
        plt.ylabel('Energy')
        plt.title('Energy convergence plot')
        plt.legend(loc='upper right')
        plt.savefig(f"graphs/{fileName} - {strftime('%Y-%m-%d %H%M', localtime())}")