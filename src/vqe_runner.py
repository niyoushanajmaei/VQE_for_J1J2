from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import QuantumCircuit
from src.ansatze.sample_ansatz import SampleAnsatz
from src.model import Model


class VqeRunner:
    def __init__(self, lattice_size, simulation = True, seed=50, ansatz="sample_ansatz"):
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
        self.hamiltonian = Model.get_hamiltonian(lattice_size)
        if simulation:
            self.optimizer = "SLSQP"
        else:
            self.optimizer = "SPSA"

    def run_vqe(self) -> dict:
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

        ansatz:QuantumCircuit = SampleAnsatz()
        if self.optimizer == "SLSQP":
            slsqp = SLSQP(maxiter=1000)
        elif self.optimizer == "SPSA":
            # TODO for SPSA, investigate the usage of a user defined gradient
            slsqp = SPSA(maxiter=100)
        else:
            raise UnvalidOptimizerError
        vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi, include_custom=True)
        result = vqe.compute_minimum_eigenvalue(operator=self.hamiltonian)
        optimal_value1 = result.optimal_value
        return result


class UnvalidOptimizerError(RuntimeError):
    """
    Raised when an unvalid optimizer value is used
    """
    pass