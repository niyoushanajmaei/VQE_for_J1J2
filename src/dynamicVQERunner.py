import numpy as np
from matplotlib import pyplot as plt
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA, ADAM, COBYLA
from qiskit.utils import algorithm_globals, QuantumInstance

from src.VQERunner import UnidentifiedAnsatzError, VQERunner, InvalidOptimizerError
from src.model import Model
from src.vqe_algorithm.ansatz import Ansatz
from src.vqe_algorithm.ansatze.FeulnerHartmannAnsatz import FeulnerHartmannAnsatz
from src.vqe_algorithm.ansatze.twoLocalAnsatz import TwoLocalAnsatz

from time import localtime, strftime

# TODO:
# - add offsets to the ansatz modifiers, to prevent them from running together?
#       But this would require stopping the optimizer more often...
#       or convert to a probability based model for modifications


class DynamicVQERunner:

    def __init__(self,  m, n, J1, J2, seed=50, ansatz_rep=7, h=0, periodic_hamiltonian = False, ansatz="TwoLocal", optimizer="SLSQP", totalMaxIter = 1000):
        self.seed = seed
        self.ansatz = None
        self.initialise_ansatz(ansatz,n*m, ansatz_rep)
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
        # It takes way too long to compute the ground state energy for 3x4
        # so I hard coded it in
        # But this is only for the open system
        # TODO: add similar hardcoding for periodic bc as well
        if self.N == 12:
            self.exactEnergy = -22.138
        else:
            self.exactEnergy = Model.getExactEnergy(self.hamiltonianMatrix)

    def run_dynamic_vqe(self, step_iter=np.inf, small_gradient_deletion=False, small_gradient_add_to_end=False,
                        random_pseudo_removal=False, add_layers_fresh=False, add_layers_duplicate=False,
                        large_gradient_add=False):
        """
        Runs the VQE algorithm
        # For now, setting add_layers and large_gradient_adds to True simultaneously is not implemented

        :param small_gradient_deletion: if True, "alpha" of percent gates with the smallest parameter gradient are
            removed from that layer
        :param small_gradient_add_to_end: if True, "alpha" percent of gates with the smallest parameter gradient are
            removed and added to the end of the ansatz instead
        :param large_gradient_add: if True, "beta" percent of gates with the largest parameter gradient are either
            duplicated in place, or added to the end of the ansatz
        :param add_layers_duplicate: if True, a duplicate layer of the last layer is added every "step_iter"
            iterations
        :param add_layers_fresh: if True, an uninitialized layer is added every "step_iter" iterations
        :param initial_reps: The number of repititions in the ansatz to begin with
        :param random_pseudo_removal: if True, at each iteration, a randomly chosen "gamma" percent of the gates are
            temporarily removed from the ansatz

        """
        final_reps = self.ansatz.reps
        # number of iterations before adding a layer to the ansatz
        if add_layers_fresh and not large_gradient_add:
            if self.ansatz.name == "TwoLocal":
                initial_reps = 4  # 40 parameters
            elif self.ansatz.name == "FeulnerHartmann":
                initial_reps = 1  # 39 parameters
            else:
                raise UnidentifiedAnsatzError
            self.initialise_ansatz(self.ansatz.name, self.ansatz.N, initial_reps) # reinitialize ansatz
            if initial_reps != final_reps:
                step_iter = int(self.totalMaxIter / (final_reps - initial_reps))
            else:
                step_iter = self.totalMaxIter
        elif large_gradient_add and add_layers_fresh:
            raise SimultaneousGradientAndLayer
        else:
            # number of iterations before stopping the optimizer for modifications
            step_iter = min(step_iter, self.totalMaxIter)

        seed = self.seed
        print(f"Running with seed: {seed}")
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
        elif self.optimizer == "COBYLA":
            opt = COBYLA(maxiter=step_iter, tol=1e-6)
        else:
            raise InvalidOptimizerError
        print(f"Using {self.optimizer} optimizer with {self.totalMaxIter} total iterations, stopping every {step_iter} iterations for modifications")

        counts = []
        values = []

        def store_intermediate_results(evalCount, parameters, mean, std):
            counts.append(evalCount)
            values.append(mean)
            # print(f"Current Estimated Energy: {mean}")

        for i in range(0, self.totalMaxIter, step_iter):
            # print(initialTheta)
            vqe = VQE(ansatz, optimizer=opt, initial_point=initialTheta, callback=store_intermediate_results,
                      quantum_instance=qi, include_custom=True)
            if initialTheta:
                print("estimate after ansatz modification:", vqe.get_energy_evaluation(operator=self.hamiltonian)(initialTheta))
            result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)
            print(f"iteration {i+step_iter}/{self.totalMaxIter}: current estimate: {result.optimal_value}")
            finalTheta = result.optimal_point.tolist()

            if large_gradient_add:
                # do modifications?
                # still have initial theta at this stage, can process finalTheta and initialTheta
                self.ansatz.update_parameters(finalTheta)
                if initialTheta:
                    paramGrad = np.abs(np.array(finalTheta) - np.array(initialTheta))
                    mx = np.max(paramGrad)
                    assert mx  # make sure it isn't zero, since we will be using it to normalize paramGrad
                    # print(paramGrad)
                    paramGrad = paramGrad / mx
                    # print(paramGrad)
                    self.ansatz.add_large_gradient_gate_end(paramGrad)
                initialTheta = self.ansatz.get_parameters()
            if add_layers_fresh:
                if i+step_iter < self.totalMaxIter:
                    # change both the ansatz and the theta accordingly
                    self.initialise_ansatz(self.ansatz.name, self.ansatz.N, self.ansatz.reps+1)
                    ansatz = self.ansatz.circuit
                    finalTheta = self.ansatz.add_fresh_parameter_layer(finalTheta)
                    self.ansatz.update_parameters(finalTheta)
                    print(f"Added one layer to the ansatz, current reps: {self.ansatz.reps}, current number of parameter:"
                          f" {len(finalTheta)}")
                    initialTheta = finalTheta

        self.ansatz.update_parameters(finalTheta)
        # save convergence plot for the run
        counts = [np.asarray(counts)]
        values = [np.asarray(values)]
        optimizers = [self.optimizer]
        fileName = f"{self.periodicity}-{self.m}x{self.n}-{self.ansatz}-{self.optimizer}-{self.totalMaxIter}iters"
        self.plotConvergences(counts, values, optimizers, fileName=fileName)

        return result

    def initialise_ansatz(self, name, N, reps):
        if name == "FeulnerHartmann":
            self.ansatz = FeulnerHartmannAnsatz(N, reps)
        elif name == "TwoLocal":
            self.ansatz = TwoLocalAnsatz(N, reps)
        else:
            raise UnidentifiedAnsatzError

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
        plt.axhline(y=self.exactEnergy, color='r', linestyle='-', label="exact energy")

        plt.xlabel('Eval count')
        plt.ylabel('Energy')
        plt.title('Energy convergence plot')
        plt.plot([], [], ' ', label=f"Final VQE Estimate: {np.round(values[0][-1], 6)}")
        plt.legend(loc='upper right')
        # plt.annotate()
        plt.savefig(f"graphs/{fileName}-{strftime('%Y-%m-%d %H%M', localtime())}")

    def add_layers_duplicate(initial_ansatz: Ansatz) -> Ansatz:
        """
        Adds another layer to the end of the ansatz. The added ansatz is a duplicate of the last layer of the initial ansatz,
        Both parameter-wise and gate wise.

        :param initial_ansatz: the initial ansatz

        :return: the final ansatz
        """

class SimultaneousGradientAndLayer(RuntimeError):
    """
        The feature of having the gradient-based dynamic VQE and the gradual addition of layers is not implemented yet
    """
    pass

