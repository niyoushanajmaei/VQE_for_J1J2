import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA, ADAM
from qiskit.circuit.library import TwoLocal

from src.vqe_algorithm.ansatze.FuelnerHartmannAnsatz import FuelnerHartmannAnsatz
from src.vqe_algorithm.ansatze.twoLocalAnsatz import TwoLocalAnsatz
from src.model import Model

from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy



class VQERunner:

    def __init__(self, m, n, J1, J2, h=0, periodic_hamiltonian = False, simulation = True, seed=50, ansatz="FeulnerHartmann", optimizer="SLSQP"):
        """
        For running on back-end "SPSA" is used as optimizer
        For simualtion "SLSQP" is used as optimizer

        QISKIT Documentation:
        SPSA can be used in the presence of noise, and it is therefore indicated in situations involving measurement
        uncertainty on a quantum computation when finding a minimum. If you are executing a variational algorithm using
        a Quantum ASseMbly Language (QASM) simulator or a real device, SPSA would be the most recommended choice among
        the optimizers provided here.
        """
        self.seed = seed
        if ansatz == "FeulnerHartmann":
            self.ansatz = FuelnerHartmannAnsatz(m*n)
        elif ansatz == "TwoLocal":
            self.ansatz = TwoLocalAnsatz(m*n)
        else:
            raise UnidentifiedAnsatzError
        self.m = m
        self.n = n
        self.N = m*n
        if periodic_hamiltonian:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_periodic(m, n, J1, J2, h=h)
        else:
            self.hamiltonian = Model.getHamiltonian_J1J2_2D_open(m, n, J1, J2, h=h)
        self.hamiltonianMatrix = self.hamiltonian.to_matrix()
        self.optimizer = optimizer
        # if simulation:
        #     self.optimizer = "SLSQP"
        # else:
        #     self.optimizer = "SPSA"
        self.simulation = simulation

    def runVQE(self, monitor=True):
        """
        Runs the VQE algorithm

        :return: A Dictionary of results of the run
        Example return object: {   'aux_operator_eigenvalues': None,
                                     'cost_function_evals': 65,
        'eigenstate': {'01': 0.9921567416492215, '10': 0.125},
        'eigenvalue': (-1.8572750175571595+0j),
        'optimal_circuit': None,
        'optimal_parameters': {   ParameterVectorElement(θ[5]): 1.5683259454122547,
                              ParameterVectorElement(θ[2]): 0.5470754193210003,
                              ParameterVectorElement(θ[3]): 6.092947857528147,
                              ParameterVectorElement(θ[0]): 4.2965205340503685,
                              ParameterVectorElement(θ[4]): -2.5983258639687397,
                              ParameterVectorElement(θ[7]): 0.36020735708081203,
                              ParameterVectorElement(θ[1]): 4.426962242132452,
                              ParameterVectorElement(θ[6]): -4.717618177195121},
        'optimal_point': array([ 4.29652053,  4.42696224,  0.54707542,  6.09294786, -2.59832586,
        1.56832595, -4.71761818,  0.36020736]),
        'optimal_value': -1.8572750175571595,
        'optimizer_evals': None,
        'optimizer_result': None,
        'optimizer_time': 0.10461235046386719}
        """

        seed = self.seed
        algorithm_globals.random_seed = seed
        qi = QuantumInstance(self.get_backend(simulate=self.simulation), seed_transpiler=seed, seed_simulator=seed)


        ansatz = self.ansatz.circuit
        print(ansatz)


        if self.optimizer == "SLSQP":
            iter = 1000
            opt = SLSQP(maxiter=iter)
            print(f"Using SLSQP optimizer with {iter} iterations")
        elif self.optimizer == "SPSA":
            # TODO for SPSA, investigate the usage of a user defined gradient
            iter = 10
            opt = SPSA(maxiter=iter)
            print(f"Using SPSA optimizer with {iter} iterations")
        elif self.optimizer == "AMSGRAD":
            iter = 50
            lr = 0.5
            opt = ADAM(maxiter=iter, lr=lr, amsgrad=True)
        else:
            raise InvalidOptimizerError

        if monitor:
            counts = []
            values = []

            def store_Intermediate_Results(evalCount, parameters, mean, std):
                counts.append(evalCount)
                values.append(mean)
                print(f"Current Estimated Energy: {mean}")

            vqe = VQE(ansatz, optimizer=opt, callback=store_Intermediate_Results, quantum_instance=qi,include_custom=True)
        else:
            vqe = VQE(ansatz, optimizer=opt, quantum_instance=qi, include_custom=True)

        result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)

        if monitor:
            counts = [np.asarray(counts)]
            values = [np.asarray(values)]
            optimizers = [self.optimizer]
            self.plotConvergences(counts, values, optimizers)
        return result

    def plotConvergences(self, counts, values, optimizers, fileName="convergenceGraph.png"):
        """
        plots the convergence plots for a list of counts and values

        :param: counts and values should be a list of np arrays
            optimizers is a list of the name of the optimizers corresponding to each set of counts and values
        """
        plt.figure(figsize=(12, 8))
        for i, optimizer in enumerate(optimizers):
            plt.plot(counts[i], values[i], label=optimizer)

        # plotting exact value
        plt.axhline(y=Model.getExactEnergy(self.hamiltonianMatrix), color='r', linestyle='-', label="exact energy")

        plt.xlabel('Eval count')
        plt.ylabel('Energy')
        plt.title('Energy convergence plot')
        plt.legend(loc='upper right')
        plt.savefig(f"graphs/{fileName}")

    def compare_Optimizers_And_Ansatze(self):
        """
        Runs the VQE algorithm with a list of optimizers and plots the convergence graphs

        """
        twolocal = TwoLocalAnsatz(self.N)
        fuelner = FuelnerHartmannAnsatz(self.N)
        ansatze = {"twoLocal": twolocal.circuit}
        optimizers = [SLSQP(maxiter=1000), SPSA(maxiter=500), ADAM(maxiter=50, lr=0.6), ADAM(maxiter=50, amsgrad=True, lr=0.6)]

        seed = self.seed
        algorithm_globals.random_seed = seed
        backend = self.get_backend(simulate=self.simulation)
        print(f"Using {backend.name()}")
        qi = QuantumInstance(backend, seed_transpiler=seed, seed_simulator=seed)

        for name, ansatz in ansatze.items():
            convergeCnts = np.empty([len(optimizers)], dtype=object)
            convergeVals = np.empty([len(optimizers)], dtype=object)
            optimizerNames = []
            for i, optimizer in enumerate(optimizers):
                print(f"Running for {name} and {type(optimizer).__name__}")
                counts = []
                values = []

                def store_Intermediate_Results(evalCount, parameters, mean, std):
                    counts.append(evalCount)
                    values.append(mean)

                vqe = VQE(ansatz, optimizer, callback=store_Intermediate_Results, quantum_instance=qi, include_custom=True)
                result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)

                convergeCnts[i] = np.asarray(counts)
                convergeVals[i] = np.asarray(values)
                optimizerNames.append(type(optimizer).__name__)

            self.plotConvergences(convergeCnts, convergeVals, optimizerNames, fileName=f"{name}")


    def tune_lr_iter_for_optimizer(self):
        twolocal = TwoLocalAnsatz(self.N)
        fuelner = FuelnerHartmannAnsatz(self.N)
        ansatz = twolocal.circuit

        seed = self.seed
        algorithm_globals.random_seed = seed
        backend = self.get_backend(simulate=self.simulation)
        print(f"Using {backend.name()}")
        qi = QuantumInstance(backend, seed_transpiler=seed, seed_simulator=seed)

        best_result = None
        best_lr = None
        for lr in np.arange(0.001, 0.7, 0.001):
            optimizer = ADAM(lr=lr, maxiter=1000)
            print(f"Running for lr={lr}")
            vqe = VQE(ansatz, optimizer, quantum_instance=qi,
                          include_custom=True)
            result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)
            optimal_value = result.optimal_value
            print(f"optimal value for lr={lr} was {optimal_value}")
            if best_result is None or optimal_value < best_result:
                best_result = optimal_value
                best_lr = lr

        print(f"Done. Overall best result was: {best_result} for lr={best_lr}")


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


class InvalidOptimizerError(RuntimeError):
    """
    Raised when an Invalid optimizer value is used
    """
    pass

class UnidentifiedAnsatzError(RuntimeError):
    """
    Raised when the given ansatz name is unidentified
    """
    pass
