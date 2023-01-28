import time

from matplotlib import pyplot as plt

from src.VQERunner import VQERunner
from src.model import Model
from src.dynamicVQERunner import DynamicVQERunner
import numpy as np
from scipy import stats

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
    n = 4
    J1 = 1
    J2 = 0.5

    vqe_runner = DynamicVQERunner(m, n, J1, J2, h=0, seed=seed, ansatz_rep=layers, periodic_hamiltonian=False, ansatz=ansatz, optimizer=optimizer, totalMaxIter=1000)
    result = vqe_runner.run_dynamic_vqe(step_iter=100, large_gradient_add=True, gradient_beta=0.2) # pass gradient_beta=None for adding one gate
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
    num = 5
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

        vqe_runner = DynamicVQERunner(m, n, J1, J2, h=0, seed=seed, ansatz_rep=layers, periodic_hamiltonian=False, ansatz=ansatz,
                                      optimizer=optimizer, totalMaxIter=1000)
        # result = vqe_runner.run_dynamic_vqe(step_iter=10 ,large_gradient_add=True)
        result = vqe_runner.run_dynamic_vqe(add_layers_fresh=True)
        optimal_values.append(result.optimal_value)

        with open(f"results/local_minima/3x4/static/dynamic_results_TL_open_{layers}_4", "a") as f:
            f.write(f"{result.optimal_value}, ")

        plt.close()

    print(optimal_values)
    print(f"minimum: {min(optimal_values)}")

    plt.figure(figsize=(12, 8))
    plt.hist(optimal_values, bins=num_bins, color='green')

    plt.title(f'Distribution of results for running the {ansatz} ansatz with {layers} layers, {num} times.')
    plt.savefig(f"results/local_minima/3x4/static/dynamic_results_TL_open_{layers}_4")

    with open(f"results/local_minima/3x4/static/dynamic_results_TL_open_{layers}_4", "a") as f:
        f.write(f"optimal values: ")
        for e in optimal_values:
            f.write(f"{e}, ")
        f.write(f"\nminimum achieved: {min(optimal_values)}")


def interrupt_with_no_mod_test():
    start = time.time()
    seed = 50
    # ansatz in {"TwoLocal", "FeulnerHartmann"}
    ansatz = "FeulnerHartmann"
    layers = 7
    # optimizer in {"SLSQP", "SPSA", "AMSGRAD", "COBYLA"}
    optimizer = "COBYLA"

    m = 3
    n = 4
    J1 = 1
    J2 = 0.5

    vqe_runner = DynamicVQERunner(m, n, J1, J2, h=0, seed=seed, ansatz_rep=layers, periodic_hamiltonian=False,
                                  ansatz=ansatz, optimizer=optimizer, totalMaxIter=1000)
    result = vqe_runner.run_interrupt_test(step_iter=5000, random_restart=False)
    print(result)

    print(f"The algorithm took {time.time() - start:.2f}s")

    print(f"exactResult: {vqe_runner.exactEnergy}")


def analysis_box_plots():
    with open("stat_data/cobyla_with_layer_adding.txt","r") as f:
        data_with_layer_adding = f.read().split(',')

    with open("stat_data/cobyla_without_layer_adding.txt","r") as f:
        data_without_layer_adding = f.read().split(',')

    exact = -22.138

    # box plots
    data_dict = {'without layer adding': data_without_layer_adding, 'with layer adding': data_with_layer_adding}

    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values())
    ax.set_xticklabels(data_dict.keys())
    line = plt.axhline(exact, c='r')
    line.set_label("Exact Energy")
    plt.legend()
    plt.ylim([-22.2, -20.0])

    plt.savefig('results/COBYLA_layer_adding/analysis/COBYLA_BoxPlot.png')
    plt.close()


def analysis_t_test():
    with open("stat_data/cobyla_with_layer_adding.txt", "r") as f:
        data_with_layer_adding = f.read().split(',')

    with open("stat_data/cobyla_without_layer_adding.txt", "r") as f:
        data_without_layer_adding = f.read().split(',')

    exact = -22.138

    mean_with_layer_adding = np.mean(data_with_layer_adding)
    std_with_layer_adding = np.std(data_with_layer_adding)
    mean_without_layer_adding = np.mean(data_without_layer_adding)
    std_without_layer_adding = np.std(data_without_layer_adding)

    with open("results/COBYLA_layer_adding/analysis/COBYLA_t-test.txt", "w") as f:
        f.write(f"without layer adding: mean={mean_without_layer_adding}, std={std_without_layer_adding}\n")
        f.write(f"with layer adding: mean={mean_with_layer_adding}, std={std_with_layer_adding}\n")
        # Welch's t-test
        f.write(str(stats.ttest_ind(data_with_layer_adding, data_without_layer_adding, equal_var=False)))


def slsqp_cobyla_box_plots():
    with open("stat_data/cobyla_with_layer_adding.txt", "r") as f:
        cobyla_with_layer_adding = f.read().split(',')
        cobyla_with_layer_adding = list(map(float, cobyla_with_layer_adding))

    with open("stat_data/cobyla_without_layer_adding.txt", "r") as f:
        cobyla_without_layer_adding = f.read().split(',')
        cobyla_without_layer_adding = list(map(float, cobyla_without_layer_adding))

    with open("stat_data/slsqp_with_layer_adding.txt", "r") as f:
        slsqp_with_layer_adding = f.read().split(',')
        slsqp_with_layer_adding = list(map(float, slsqp_with_layer_adding))

    with open("stat_data/slsqp_without_layer_adding.txt", "r") as f:
        slsqp_without_layer_adding = f.read().split(',')
        slsqp_without_layer_adding = list(map(float, slsqp_without_layer_adding))

    exact = -22.138

    # box plots
    data_dict1 = {'cobyla without layer adding': cobyla_without_layer_adding,
         'cobyla with layer adding': cobyla_with_layer_adding}
    data_dict2 = { 'slsqp without layer adding': slsqp_without_layer_adding,
                 'slsqp with layer adding': slsqp_with_layer_adding}

    fig, ax = plt.subplots()
    ax.boxplot(data_dict2.values())
    ax.set_xticklabels(data_dict2.keys())
    line = plt.axhline(exact, c='r', linestyle=":")
    line.set_label("Exact Energy")
    plt.legend()
    plt.ylim([-22.2, -21.8])

    plt.savefig('stat_data/slsqp_box_plot.png')
    plt.close()




if __name__ == "__main__":
    #tune_adam()
    #test_compare_ansatze()
    #testDynamicRunner()
    #check_local_minima_hypothesis()
    #interrupt_with_no_mod_test()
    #analysis_box_plots()
    #analysis_t_test()
    slsqp_cobyla_box_plots()
    #slsqp_cobyla_t_test()
