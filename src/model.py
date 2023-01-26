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
    def getHamiltonian_J1J2_2D_periodic(m, n, J1, J2, h=0):
        """
            periodic => periodic boundary is considered
            spins on the last row are nearest neighbours to spins on the first row
            and similarly for the first and last columns

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
                            curr = I
                        if not total:
                            total = curr
                        else:
                            total = total ^ curr
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
                            total = total ^ curr
                    H1 += total

        # contribution of next to nearest neighbour, spins X, Y and Z:
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
                            curr = I
                        if not total:
                            total = curr
                        else:
                            total = total ^ curr
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
                            total = total ^ curr
                    H2 += total

        return J1 * H1 + J2 * H2

    @staticmethod
    def getHamiltonian_J1J2_2D_open(m, n, J1, J2, h=0):
        """
            open => no periodic boundary interactions
            i.e., spin on first row is not a nearest neighbour to the spin on the last row
            and similarly for the first and last columns

            m x n lattice of spins
            J1: nearest neighbour interaction
            J2: next to nearest neighbour interaction
            h: magnetic field, taken to be 0 for now
            
            H = - J1 ΣSi.Sj - J2 ΣSi.Sj - h ΣSi
        """

        if m < 1 or n < 1 or (m == 1 and n == 1):
            raise InvalidSizeError

        N = m*n
        H1 = 0
        H2 = 0

        # contribution of nearest neighbour, spins X, Y and Z:
        for pauli in [X, Y, Z]:
            for i in range(m):
                for j in range(n):
                    indexCurr = Model.getIndex(i, j, m, n)
                    if j < n-1: #not last column
                        index_East = Model.getIndex(i, j+1, m, n)
                        total = 0
                        for ind in range(0, N):
                            if (ind == indexCurr 
                                or ind == index_East):
                                curr = pauli
                            else:
                                curr = I
                            if not total:
                                total = curr
                            else:
                                total = total^curr
                        if not H1:
                            H1 = total
                        else:
                            H1 += total
                    if i < m-1: #not last row
                        index_South = Model.getIndex(i+1, j, m, n)
                        total = 0
                        for ind in range(0, N):
                            if (ind == indexCurr or ind == index_South):
                                curr = pauli
                            else:
                                curr = I
                            if not total:
                                total = curr
                            else:
                                total = total^curr
                        if not H1:
                            H1 = total
                        else:
                            H1 += total
                    
        # contribution of next to nearest neighbour, spins X, Y and Z:
        for pauli in [X, Y, Z]:
            for i in range(m):
                for j in range(n):
                    indexCurr = Model.getIndex(i, j, m, n)
                    if (i > 0 and j < n-1):
                        index_NorthEast = Model.getIndex(i-1, j+1, m, n)
                        total = 0
                        for ind in range(0, N):
                            if (ind == indexCurr or ind == index_NorthEast):
                                curr = pauli
                            else:
                                curr = I
                            if not total:
                                total = curr
                            else:
                                total = total^curr
                        if not H2:
                            H2 = total
                        else:
                            H2 += total
                    if (i < m-1 and j < n-1):
                        index_SouthEast = Model.getIndex(i+1, j+1, m, n)
                        total = 0
                        for ind in range(0, N):
                            if (ind == indexCurr or ind == index_SouthEast):
                                curr = pauli
                            else:
                                curr = I
                            if not total:
                                total = curr
                            else:
                                total = total^curr
                        if not H2:
                            H2 = total
                        else:
                            H2 += total
                   
        return J1 * H1 + J2 * H2
                
    # no more corner cases: both 1D, 2x2 work

    @staticmethod
    def getExactEnergy(hamiltonianMatrix):
        # TODO: approximate the eigenvalue for larger systems
        eigenValues = np.sort(np.real(np.linalg.eigvals(hamiltonianMatrix)))
        exactEnergy = eigenValues[0]
        # print(eigenValues)
        return exactEnergy


class InvalidSizeError(RuntimeError):
    """ Lattice Size should be n>1 """
    pass