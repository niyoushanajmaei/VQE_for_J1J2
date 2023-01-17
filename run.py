import time

from matplotlib import pyplot as plt

from src.VQERunner import VQERunner
from src.model import Model
from src.dynamicVQERunner import DynamicVQERunner
import numpy as np


def test_with_qiskit():
    start = time.time()
    seed = 50
    ansatz = "TwoLocal"
    # ansatz = "FeulnerHartmann"
    optimizer = "SLSQP"

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


def testDynamicRunner():
    start = time.time()
    seed = 50
    # ansatz in {"TwoLocal", "FeulnerHartmann"}
    ansatz = "FeulnerHartmann"
    layers = 2
    # optimizer in {"SLSQP", "SPSA", "AMSGRAD", "COBYLA"}
    optimizer = "SLSQP"

    m = 3
    n = 3
    J1 = 1
    J2 = 0.5

    vqe_runner = DynamicVQERunner(m, n, J1, J2, h=0, seed=seed, ansatz_rep=layers, periodic_hamiltonian=False, ansatz=ansatz, optimizer=optimizer, totalMaxIter=1000)
    result = vqe_runner.run_dynamic_vqe(step_iter=100, large_gradient_add=True)
    #result = vqe_runner.run_dynamic_vqe(add_layers_fresh=True)
    print(result)

    print(f"The algorithm took {time.time()-start:.2f}s")

    print(f"exactResult: {vqe_runner.exactEnergy}")


def tune_number_of_layers_for_adam():
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


def check_local_minima_hypothesis():
    # Running the TwoLocal Ansatz with random seeds

    layers = 7
    ansatz = "FeulnerHartmann"
    optimal_values = []
    num = 50
    num_bins = 20

    for i in range(num):
        if i % 10 == 0:
            print(f"running for iter {i}")
        seed = np.random.randint(100*num, size=1)[0]

        # optimizer in {"SLSQP", "SPSA", "AMSGRAD", "COBYLA"}
        optimizer = "SLSQP"

        m = 3
        n = 4
        J1 = 1
        J2 = 0.5

        # print(Model.getHamiltonian_J1J2_2D(m,n,J1,J2))

        vqe_runner = DynamicVQERunner(m, n, J1, J2, h=0, seed=seed, ansatz_rep=layers, periodic_hamiltonian=False, ansatz=ansatz,
                                      optimizer=optimizer, totalMaxIter=1000)
        # result = vqe_runner.run_dynamic_vqe(step_iter=10 ,large_gradient_add=True)
        result = vqe_runner.run_dynamic_vqe(add_layers_fresh=True)
        optimal_values.append(result.optimal_value)

        with open(f"results/local_minima/3x4/dynamic/dynamic_results_TL_open_{layers}", "a") as f:
            f.write(f"{result.optimal_value}, ")

    print(optimal_values)
    print(f"minimum: {min(optimal_values)}")

    plt.figure(figsize=(12, 8))
    plt.hist(optimal_values, bins=num_bins, color='green')

    plt.title(f'Distribution of results for running the {ansatz} ansatz with {layers} layers, {num} times.')
    plt.savefig(f"results/local_minima/3x4/dynamic/dynamic_distribution_TL_open_{layers}")

    with open(f"results/local_minima/3x4/dynamic/dynamic_results_TL_open_{layers}", "a") as f:
        f.write(f"optimal values: ")
        for e in optimal_values:
            f.write(f"{e}, ")
        f.write(f"\nminimum achieved: {min(optimal_values)}")



if __name__ == "__main__":
    # tune_adam()
    # test_compare_ansatze()
    testDynamicRunner()
    #check_local_minima_hypothesis()
