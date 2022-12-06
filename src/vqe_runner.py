import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.circuit.library import TwoLocal

from src.ansatze.two_local_ansatz import TwoLocalAnsatz
from src.model import Model



class VqeRunner:

    def __init__(self, m, n, J1, J2, h=0, simulation = True, seed=50, ansatz="sample_ansatz"):
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
        self.ansatz = ansatz
        self.m = m
        self.n = n
        self.N = m*n
        self.hamiltonian = Model.getHamiltonian_J1J2_2D(m, n, J1, J2, h=h)
        self.hamiltonian_matrix = self.hamiltonian.to_matrix()
        if simulation:
            self.optimizer = "SLSQP"
        else:
            self.optimizer = "SPSA"

    def run_vqe(self, monitor=True):
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
        qi = QuantumInstance(Aer.get_backend('aer_simulator'), seed_transpiler=seed, seed_simulator=seed)


        ansatz:TwoLocal = TwoLocalAnsatz.get_ansatz(self.N)
        print(ansatz)


        if self.optimizer == "SLSQP":
            slsqp = SLSQP(maxiter=1000)
        elif self.optimizer == "SPSA":
            # TODO for SPSA, investigate the usage of a user defined gradient
            slsqp = SPSA(maxiter=500)
        else:
            raise UnvalidOptimizerError

        if monitor:
            counts = []
            values = []

            def store_intermediate_results(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)

            vqe = VQE(ansatz, optimizer=slsqp, callback=store_intermediate_results, quantum_instance=qi,include_custom=True)
        else:
            vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi, include_custom=True)

        result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)

        if monitor:
            counts = [np.asarray(counts)]
            values = [np.asarray(values)]
            optimizers = [self.optimizer]
            self.plot_convergences(counts, values, optimizers)
        return result

    def plot_convergences(self, counts, values, optimizers, file_name="convergence_graph.png"):
        """
        plots the convergence plots for a list of counts and values

        :param: counts and values should be a list of np arrays
            optimizers is a list of the name of the optimizers corresponding to each set of counts and values
        """
        plt.figure(figsize=(12, 8))
        for i, optimizer in enumerate(optimizers):
            plt.plot(counts[i], values[i], label=optimizer)

        # plotting exact value
        plt.axhline(y=Model.get_exact_energy(self.hamiltonian_matrix), color='r', linestyle='-', label="exact energy")

        plt.xlabel('Eval count')
        plt.ylabel('Energy')
        plt.title('Energy convergence plot')
        plt.legend(loc='upper right')
        plt.savefig(f"graphs/{file_name}")

    def compare_optimizers_and_ansatze(self):
        """
        Runs the VQE algorithm with a list of optimizers and plots the convergence graphs

        """

        ansatze = {"two_local": TwoLocalAnsatz.get_ansatz(self.N)}
        optimizers = [SLSQP(maxiter=1000), SPSA(maxiter=500)]

        seed = self.seed
        algorithm_globals.random_seed = seed
        backend = Aer.get_backend('aer_simulator')
        qi = QuantumInstance(backend, seed_transpiler=seed, seed_simulator=seed)

        for name, ansatz in ansatze.items():
            converge_cnts = np.empty([len(optimizers)], dtype=object)
            converge_vals = np.empty([len(optimizers)], dtype=object)
            optimizer_names = []
            for i, optimizer in enumerate(optimizers):
                counts = []
                values = []

                def store_intermediate_results(eval_count, parameters, mean, std):
                    counts.append(eval_count)
                    values.append(mean)

                vqe = VQE(ansatz, optimizer, callback=store_intermediate_results, quantum_instance=qi, include_custom=True)
                result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)

                converge_cnts[i] = np.asarray(counts)
                converge_vals[i] = np.asarray(values)
                optimizer_names.append(type(optimizer).__name__)

            self.plot_convergences(converge_cnts, converge_vals, optimizer_names, file_name=f"{name}")


class UnvalidOptimizerError(RuntimeError):
    """
    Raised when an unvalid optimizer value is used
    """
    pass