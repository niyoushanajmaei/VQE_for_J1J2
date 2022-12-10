import qiskit.opflow
from qiskit.opflow import X, Y, Z, I
import numpy as np

class Model:

    @staticmethod
    def getIndex(i, j, m, n):
        """
            mapping for spins: i,j (0<=i
                maps to (i*n+j)
        """
        assert 0 <= i < m and 0 <= j < n
        return i * n + j


    @staticmethod
    def getHamiltonian_J1J2_2D(m, n, J1, J2, h=0):
        """
            m x n lattice of spins
            J1: nearest neighbour interaction
            J2: next to nearest neighbour interaction
            h: magnetic field, taken to be 0 for now

            H = - J1 ΣSi.Sj - J2 ΣSi.Sj - h ΣSi

            corner cases: 1D, 2x2 don't work
            1D takes neighbours that don't exist
            and 2D has multiple repetitions of same bonds
        """

        if m < 1 or n < 1 or (m == 1 and n == 1):
            raise InvalidSizeError

        N = m * n
        H1 = 0
        H2 = 0

        # contribution of nearest neighbour, spins X, Y and Z:
        # for pauli in [pauli_x, pauli_y, pauli_z]:
        for pauli in [X, Y, Z]:
            for i in range(m):
                for j in range(n):
                    indexCurr = Model.getIndex(i, j, m, n)
                    indexEast = Model.getIndex(i, (j + 1) % n, m, n)
                    total = 0
                    for ind in range(0, N):
                        if (ind == indexCurr
                                or ind == indexEast):
                            curr = pauli
                        else:
                            # curr = np.identity((2,2))
                            curr = I
                        if not total:
                            total = curr
                        else:
                            # total = np.kron(total, curr)
                            total = total ^ curr
                    # print(total)
                    if not H1:
                        H1 = total
                    else:
                        H1 += total
                    indexSouth = Model.getIndex((i + 1) % m, j, m, n)
                    total = 0
                    for ind in range(0, N):
                        if ind == indexCurr or ind == indexSouth:
                            curr = pauli
                        else:
                            curr = I
                        if not total:
                            total = curr
                        else:
                            # total = np.kron(total, curr)
                            total = total ^ curr
                    H1 += total

        # contribution of next to nearest neighbour, spins X, Y and Z:
        # for pauli in [pauli_x, pauli_y, pauli_z]:
        for pauli in [X, Y, Z]:
            for i in range(m):
                for j in range(n):
                    indexCurr = Model.getIndex(i, j, m, n)
                    indexNorthEast = Model.getIndex((i - 1) % m, (j + 1) % n, m, n)
                    total = 0
                    for ind in range(0, N):
                        if ind == indexCurr or ind == indexNorthEast:
                            curr = pauli
                        else:
                            # curr = np.identity((2,2))
                            curr = I
                        if not total:
                            total = curr
                        else:
                            # total = np.kron(total, curr)
                            total = total ^ curr
                    # print(total)
                    if not H2:
                        H2 = total
                    else:
                        H2 += total
                    indexSouthEast = Model.getIndex((i + 1) % m, (j + 1) % n, m, n)
                    total = 0
                    for ind in range(0, N):
                        if ind == indexCurr or ind == indexSouthEast:
                            curr = pauli
                        else:
                            curr = I
                        if not total:
                            total = curr
                        else:
                            # total = np.kron(total, curr)
                            total = total ^ curr
                    H2 += total

        return H1 * J1 + J2 * H2


    @staticmethod
    def getExactEnergy(hamiltonianMatrix):
        # TODO: approximate the eigenvalue for larger systems
        exactEnergy = np.min(np.linalg.eigvals(hamiltonianMatrix))
        return exactEnergy


class InvalidSizeError(RuntimeError):
    """ Lattice Size should be n>1 """
    pass