import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from src.vqe_algorithm.ansatz import Ansatz


class FuelnerHartmannAnsatz(Ansatz):
    def __init__(self, N, reps=7):
        if N != 9:
            raise NotImplementedError

        super().__init__(N, reps)
        self.circuit = self._get_ansatz_w(N, reps)
        self.theta = None
        self.N = N
        self.name = "FuelnerHartmann"
        self.reps = reps

    def __str__(self):
        return f"FuelnerHartmann-{len(self.theta)} params"

    def _get_ansatz_w(self, N, reps):
        """
        Wrapper class for _get_ansatz

        :param N: size of the lattice
        """
        return self._get_ansatz(N,reps)

    @staticmethod
    def XXYYZZ(theta_i):
        # make an instruction set/gate out of XXYYZZ to make it simpler to work with the XXYYZZ gates
        qc = QuantumCircuit(2)
        qc.rxx(theta_i, 0, 1)
        qc.ryy(theta_i, 0, 1)
        qc.rzz(theta_i, 0, 1)
        XXYYZZ = qc.to_gate()
        return qc.to_gate(label = 'XXYYZZ')

    def _get_ansatz(self, N, reps):
        """
           Ansatz described in https://arxiv.org/abs/2205.11198

           Implemented here only for a 3x3 system. Hardcoded.
           Hassle to implement it for an arbitrary size system.
           Need to do it for a 4x3 system as well though; will probably hardcode that as well. Should be easier.
        """

        assert N == 9
        # FOR 3x3 ONLY:
        """
            numParams: 12 nearest neighbour links => 12 values of theta for each layer
                also, the authors include a rotation about Z for each qubit, so 9 parameters per layer
                initially, an X and Y rotation for each qubit, so 18 parameters for those
            theta: the parameters
                should this be set as an input? or return this?
            indParam: indicates how many parameters have been used up so far
        """
        numParams = 18 + 21*reps
        theta = ParameterVector('theta', numParams)
        indParam = 0
        # describe sets of gates:
        # can probably be put into 3 sets of 4 instead of 4 sets of 3, but this is more symmetric
        XXYYZZGatesList = (
            ((0,1),(4,5),(6,7)),
            ((1,2),(3,4),(7,8)),
            ((0,3),(4,7),(2,5)),
            ((3,6),(1,4),(5,8))
        )

        qc = QuantumCircuit(N)
        for i in range(N):
            qc.rx(theta[indParam], i)
            indParam += 1
        qc.barrier()
        for i in range(N):
            qc.ry(theta[indParam], i)
            indParam += 1
        qc.barrier()
        for i in range(reps):
            for i in range(N):
                qc.rz(theta[indParam], i)
                indParam += 1
            qc.barrier()
            for gate in XXYYZZGatesList:
                for qubits in gate:
                    qc.append(FuelnerHartmannAnsatz.XXYYZZ(theta[indParam]), [qubits[0], qubits[1]])
                    indParam += 1
                qc.barrier()
        # print(numParams - indParam)
        return qc

    def get_parameters(self) -> list:
        """
        should return a list of parameters of the ansatz
        """
        pass

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
        if self.N != 9:
            raise NotImplementedError

        current_params.extend(list(np.zeros(21)))
        return current_params
