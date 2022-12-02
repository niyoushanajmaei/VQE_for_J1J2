from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal


class SampleAnsatz(QuantumCircuit):
    def __init__(self):
        pass

    @staticmethod
    def get_ansatz() -> TwoLocal:
        pass
