from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from src.vqe_algorithm.ansatz import Ansatz

# TODO: Make the XXYYZZ circuit block a gate (https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.to_gate.html#qiskit.circuit.QuantumCircuit.to_gate)
# or to instruction (https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.to_instruction.html#qiskit.circuit.QuantumCircuit.to_instruction)

class FuelnerHartmannAnsatz(Ansatz):
    def __init__(self, N):
        self.circuit = self._get_ansatz_w(N)

    def _get_ansatz_w(self, N):
        """
        Wrapper class for _get_ansatz

        :param N: size of the lattice
        """
        return self._get_ansatz(N)


    def _get_ansatz(self, N, nLayers = 7):
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
        numParams = 18 + 21*nLayers
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
        for i in range(nLayers):
            for i in range(N):
                qc.rz(theta[indParam], i)
                indParam += 1
            qc.barrier()
            for gate in XXYYZZGatesList:
                for qubits in gate:
                    qc.rxx(theta[indParam], qubits[0], qubits[1])
                    qc.ryy(theta[indParam], qubits[0], qubits[1])
                    qc.rzz(theta[indParam], qubits[0], qubits[1])
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
        pass
