import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from src.vqe_algorithm.ansatz import Ansatz


class FeulnerHartmannAnsatz(Ansatz):
    def __init__(self, N, reps=7):
        if N not in [9,12]:
            raise NotImplementedError

        super().__init__(N, reps)
        self.circuit = self._get_ansatz_w(N, reps)
        self.theta = None
        self.N = N
        self.name = "FeulnerHartmann"
        self.reps = reps

    def __str__(self):
        return f"FeulnerHartmann-{len(self.theta)} params"

    def _get_ansatz_w(self, N, reps):
        """
        Wrapper class for _get_ansatz

        :param N: size of the lattice
        """
        return self._get_ansatz(N, reps)

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

        assert N in [9,12]
        # FOR 3x3 and 3x4 ONLY:
        # CURRENTLY GIVES CIRCUIT ONLY FOR 3x4; 4x3 NOT DIFFERENTIATED
        """
            numParams: 12 nearest neighbour links => 12 values of theta for each layer
                also, the authors include a rotation about Z for each qubit, so 9 parameters per layer
                initially, an X and Y rotation for each qubit, so 18 parameters for those
            theta: the parameters
                should this be set as an input? or return this?
            indParam: indicates how many parameters have been used up so far
        """
        if N == 9:
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
                        qc.append(FeulnerHartmannAnsatz.XXYYZZ(theta[indParam]), [qubits[0], qubits[1]])
                        indParam += 1
                    qc.barrier()
            # print(numParams - indParam)

        elif N == 12:
            numParams = 24 + 29*reps
            theta = ParameterVector('theta', numParams)
            indParam = 0
            # describe sets of gates:
            # can probably be put into 3 sets of 4 instead of 4 sets of 3, but this is more symmetric
            XXYYZZGatesList = (
                ((0,1),(5,6),(10,11)),
                ((1,2),(6,7),(8,9)),
                ((2,3),(4,5),(9,10)),
                ((0,4),(5,9),(2,6),(7,11)),
                ((4,8),(1,5),(6,10),(3,7))
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
                        qc.append(FeulnerHartmannAnsatz.XXYYZZ(theta[indParam]), [qubits[0], qubits[1]])
                        indParam += 1
                    qc.barrier()
            # print(numParams - indParam)
            # qc.draw(output='mpl', filename=f"{len(theta)}")

        return qc

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

    def add_large_gradient_gate_end(self, paramGrad):
        """
            Given the parameter gradient, selects the gate(s) with largest gradient(s)
            ***TO THE END OF THE CIRCUIT***

            :paramGrad: parameter gradient after latest optimization

            Does not return anything: the new (extended) parameters are updated within the ansatz object
            and the dynamicVQERunner copy of parameters needs to reassigned to this updated list

            Currently it only adds a copy of the gate with largest gradient to the end.
            - Copy multiple gates, say top "beta" percent of gates with highest gradient copied
            - parameters for the new gate: initialized to
                - 0
                - *the same value as its parent, or
                - to a random value?
        """
        qc = self.circuit
        theta = self.theta

        indexOfMax = np.argmax(paramGrad)    # to "reach" the gate with highest gradient
        parammedGateIndex = -1  # index the gates that have parameters
        for gate in qc.data:
            if (gate[0].params):
                parammedGateIndex+=1
                if (parammedGateIndex == indexOfMax):
                    newParam = Parameter(f"theta[{len(theta)}]")
                    theta.append(0)
                    match gate[0].name:
                        case "rx":
                            qc.rx(newParam, gate[1])
                        case "ry":
                            qc.ry(newParam, gate[1])
                        case "rz":
                            qc.rz(newParam, gate[1])
                        case _:
                            qc.append(FeulnerHartmannAnsatz.XXYYZZ(newParam), gate[1])

                    qc.draw(output='mpl', filename=f"{len(theta)}")
                    self.circuit = qc
                    self.theta = theta
                    break

    def add_fresh_parameter_layer(self, current_params: list) -> list:
        """
        Should add a layers of zeros to the end of the current_params list.

        :param current_params: the list of the current parameters of the circuit
        :return: the list of the final parameters of the circuit
        """
        if self.N not in [9,12]:
            raise NotImplementedError

        if self.N == 9:
            current_params.extend(list(np.zeros(21)))
        elif self.N == 12:
            current_params.extend(list(np.zeros(29)))
        return current_params
