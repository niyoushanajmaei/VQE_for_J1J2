import numpy as np
from qiskit.circuit.library import TwoLocal
from src.vqe_algorithm.ansatz import Ansatz

class TwoLocalAnsatz(Ansatz):
    def __init__(self, N, reps=10):
        super().__init__(N, reps)
        self.circuit = self._get_ansatz_w(N, reps)
        self.theta = None
        self.N = N
        self.name = "TwoLocal"
        self.reps = reps

    def __str__(self):
        return f"TwoLocal-{len(self.theta)} params"

    def _getTwoLocalAnsatz(self, N, reps, rotation_blocks=['ry'], entanglement_blocks= ['cx'], entanglement='linear'):
        """
            number of parameters:  (1+reps)*N

            N: size of system. for a mxn lattice, N = m*n
            rotation_blocks and entanglement_blocks: set of gates to use in the twolocal circuit for rotation and entanglement
            entanglement: defines entanglement strategy
                possible values: {full, linear, circular, pairwise, sca}

            returns Qiskit's twolocal circuit for N qubits
        """
        twoLocalAnsatz = TwoLocal(N, *rotation_blocks, *entanglement_blocks, entanglement, reps, insert_barriers=True)
        countParamGates = 0  #count the number of gates with parameters
        # needed when dynamicVQERunner calls to include a particular gate again due to a large gradient or so
        for gate in twoLocalAnsatz.data:
            if (gate[0].params):
                print('\ngate name:', gate[0].name)
                print('qubit(s) acted on:', gate[1])
                print('other parameters (such as angles):', gate[0].params)
                countParamGates+=1
        print(countParamGates)
        # print(numParams)
        return twoLocalAnsatz

    def _get_ansatz_w(self, N, reps):
        """
            N: size of system. for a mxn lattice, N = m*n
        """
        return self._getTwoLocalAnsatz(N, reps)

    def get_parameters(self) -> list:
        """
        should return a list of parameters of the ansatz
        """
        return self.theta

    def update_parameters(self, new_parameters):
        """
        Should update the ansatz using the given new parameters
        """
        self.theta = new_parameters

    def add_fresh_parameter_layer(self, current_params: list) -> list:
        """
        Should add a layers of zeros to the end of the current_params list.

        :param current_params: the list of the current parameters of the circuit
        :return: the list of the final parameters of the circuit
        """
        current_params.extend(list(np.zeros(self.N)))
        return current_params
