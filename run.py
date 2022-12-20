import time

from src.VQERunner import VQERunner
from src.model import Model
from src.vqe_algorithm.vqe import VQE


def test_with_qiskit():
    start = time.time()
    seed = 50
    # ansatz = "TwoLocal"
    ansatz = "FeulnerHartmann"
    optimizer = "AMSGRAD"

    m = 3
    n = 3
    J1 = 1
    J2 = 0.5

    # print(Model.getHamiltonian_J1J2_2D(m,n,J1,J2))

    vqe_runner = VQERunner(m, n, J1, J2, h=0, periodic_hamiltonian=False, simulation=True, seed=seed, ansatz=ansatz, optimizer=optimizer)
    result = vqe_runner.runVQE(monitor=True)

    print(f"The algorithm took {time.time()-start:.2f}s")
    print(result)

    print(result.optimal_point.tolist())

    # vqe_runner.compare_optimizers_and_ansatze()

    exactResult = Model.getExactEnergy(vqe_runner.hamiltonianMatrix)
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


def test_compare_ansatze():
    start = time.time()
    seed = 50
    ansatz = "TwoLocal"
    # ansatz = "FeulnerHartmann"

    m = 3
    n = 3
    J1 = 1
    J2 = 0.5

    # print(Model.getHamiltonian_J1J2_2D(m,n,J1,J2))

    vqe_runner = VQERunner(m, n, J1, J2, h=0, periodic_hamiltonian=False, simulation=True, seed=seed, ansatz=ansatz)
    vqe_runner.compare_Optimizers_And_Ansatze()

    exactResult = Model.getExactEnergy(vqe_runner.hamiltonianMatrix)
    print(f"exactResult: {exactResult}")


def tune_adam():
    start = time.time()
    seed = 50
    # ansatz = "TwoLocal"
    ansatz = "FeulnerHartmann"

    m = 3
    n = 3
    J1 = 1
    J2 = 0.5

    vqe_runner = VQERunner(m, n, J1, J2, h=0, periodic_hamiltonian=False, simulation=True, seed=seed, ansatz=ansatz)
    vqe_runner.tune_lr_iter_for_optimizer()


if __name__ == "__main__":
    tune_adam()
    # test_compare_ansatze()

