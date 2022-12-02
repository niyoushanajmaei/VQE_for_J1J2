from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

class SampleAnsatz(QuantumCircuit):
    def __init__(self):
        pass

    # move this to another class that generates required ansaetze?
    @staticmethod
    def getTwoLocalAnsatz(N, rotation_blocks=['ry'], entanglement_blocks= ['cx'], entanglement='linear', reps=2):
        """
            N: size of system. for a mxn lattice, N = m*n
            rotation_blocks and entanglement_blocks: set of gates to use in the twolocal circuit for rotation and entanglement
            entanglement: defines entanglement strategy
                possible values: {full, linear, circular, pairwise, sca}

            returns Qiskit's twolocal circuit for N qubits
        """
        twoLocalAnsatz = TwoLocal(N, *rotation_blocks, *entanglement_blocks, entanglement, reps, insert_barriers=True)
        return twoLocalAnsatz

    @staticmethod
    def get_ansatz(N):
        """
            N: size of system. for a mxn lattice, N = m*n
        """
        return SampleAnsatz.getTwoLocalAnsatz(N)
