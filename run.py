from src.VQERunner import VQERunner
from src.model import Model
from src.vqe_algorithm.vqe import VQE


def test_with_qiskit():
    seed = 50
    ansatz = "TwoLocal"
    # ansatz = "FeulnerHartmann"

    m = 3
    n = 3
    J1 = 1
    J2 = 0.5

    # print(Model.getHamiltonian_J1J2_2D(m,n,J1,J2))

    vqe_runner = VQERunner(m, n, J1, J2, h=0, simulation=True, seed=seed, ansatz=ansatz)
    result = vqe_runner.runVQE(monitor=True)

    print(result)

    # vqe_runner.compare_optimizers_and_ansatze()

    exactResult = Model.getExactEnergy(VQERunner.hamiltonianMatrix)
    print(f"exactResult: {exactResult}")


def test_with_vqe_algorithm ():
    m = 2
    n = 2
    J1 = 1
    J2 = 0.5

    vqe = VQE(m, n, J1, J2, h=0, simulation=True, ansatz="FuelnerHartmann", open_bound=True)
    result = vqe.run_vqe()

    print(result)

    exactResult = Model.getExactEnergy(vqe.hamiltonianMatrix)
    print(f"exactResult: {exactResult}")


if __name__=="__main__":
    test_with_qiskit()

