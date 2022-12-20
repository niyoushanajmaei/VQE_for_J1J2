from qiskit.circuit.library import TwoLocal
from src.vqe_algorithm.ansatz import Ansatz

class TwoLocalAnsatz(Ansatz):
    def __init__(self, N):
        super().__init__(N)
        self.circuit = self._get_ansatz_w(N)
        self.theta = None
        self.N = N

    def __str__(self):
        return f"TwoLocal-{len(theta)} params"

    def _getTwoLocalAnsatz(self, N, rotation_blocks=['ry'], entanglement_blocks= ['cx'], entanglement='linear', reps=2):
        """
            N: size of system. for a mxn lattice, N = m*n
            rotation_blocks and entanglement_blocks: set of gates to use in the twolocal circuit for rotation and entanglement
            entanglement: defines entanglement strategy
                possible values: {full, linear, circular, pairwise, sca}

            returns Qiskit's twolocal circuit for N qubits
        """
        twoLocalAnsatz = TwoLocal(N, *rotation_blocks, *entanglement_blocks, entanglement, reps, insert_barriers=True)
        return twoLocalAnsatz

    def _get_ansatz_w(self, N):
        """
            N: size of system. for a mxn lattice, N = m*n
        """
        return self._getTwoLocalAnsatz(N)

    def get_parameters(self) -> list:
        """
        should return a list of parameters of the ansatz
        """
        pass

    def update_parameters(self, new_parameters):
        """
        Should update the ansatz using the given new parameters
        """
        pass
