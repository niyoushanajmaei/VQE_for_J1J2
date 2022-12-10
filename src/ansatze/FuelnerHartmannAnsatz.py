from qiskit.circuit import QuantumCircuit
# from qiskit.circuit.library import TwoLocal
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
class FuelnerHartmannAnsatz(QuantumCircuit):
    def __init__(self):
        pass

    @staticmethod
    def XXYYZZBlock(theta):
        """
            Return a circuit with RXX RYY and RZZ gates
            applied to two qubits, each with same angle theta
        """

    @staticmethod
    def getAnsatz(N):
        """
           Ansatz described in https://arxiv.org/abs/2205.11198
        """

