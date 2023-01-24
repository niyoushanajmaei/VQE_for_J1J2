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
    layers = 7
    # optimizer in {"SLSQP", "SPSA", "AMSGRAD", "COBYLA"}
    optimizer = "SLSQP"

    m = 3
    n = 4
    J1 = 1
    J2 = 0.5

    vqe_runner = DynamicVQERunner(m, n, J1, J2, h=0, seed=seed, ansatz_rep=layers, periodic_hamiltonian=False, ansatz=ansatz, optimizer=optimizer, totalMaxIter=1000)
    #result = vqe_runner.run_dynamic_vqe(step_iter=100, large_gradient_add=True, gradient_beta=0.1) # pass gradient_beta=None for adding one gate
    result = vqe_runner.run_dynamic_vqe(add_layers_fresh=True)
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
    num = 9
    num_bins = 20

    for i in range(num):
        if i % 10 == 0:
            print(f"running for iter {i}")
        seed = np.random.randint(100*num, size=1)[0]

        # optimizer in {"SLSQP", "SPSA", "AMSGRAD", "COBYLA"}
        optimizer = "COBYLA"

        m = 3
        n = 4
        J1 = 1
        J2 = 0.5

        vqe_runner = DynamicVQERunner(m, n, J1, J2, h=0, seed=seed, ansatz_rep=layers, periodic_hamiltonian=False, ansatz=ansatz,
                                      optimizer=optimizer, totalMaxIter=50000)
        # result = vqe_runner.run_dynamic_vqe(step_iter=10 ,large_gradient_add=True)
        result = vqe_runner.run_dynamic_vqe(add_layers_fresh=True)
        optimal_values.append(result.optimal_value)

        with open(f"results/COBYLA_layer_adding/dynamic_results_TL_open_{layers}", "a") as f:
            f.write(f"{seed},{result.optimal_value},\n")

        plt.close()

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
                                  ansatz=ansatz, optimizer=optimizer, totalMaxIter=50000)
    result = vqe_runner.run_interrupt_test(step_iter=5000, random_restart=False)
    print(result)

    print(f"The algorithm took {time.time() - start:.2f}s")

    print(f"exactResult: {vqe_runner.exactEnergy}")


def analysis_box_plots():
    data_with_layer_adding = [-22.04285762504142, -22.043051631536475, -22.04895012299256, -22.046507969620638,
                              -22.055939121984956, -22.038953054445393, -22.036816805406264, -22.049267354263222,
                              -22.02147927129723, -22.02173917433623, -22.03748908017386, -22.034350190786153,
                              -22.004321225132006, -22.03838300729901, -22.043082394138807, -22.018789160515954,
                              -22.040763126087693, -22.031026739803547, -22.046744614415246, -22.054011496757465,
                              -22.04681166036868, -22.01876879302441, -22.022424345706686, -22.028495542581346,
                              -22.0331529224908, -22.023166523560953, -22.031526589867823, -21.858514861437293,
                              -22.05837374675301, -22.02925826826033]
    data_without_layer_adding = [-21.618530732586365, -21.77193049540494, -21.70085885399603, -21.71811533078925,
                                 -21.731695841841024, -21.373700294426055, -21.77882427409699, -21.509270286134853,
                                 -21.655463592564406, -21.790462830042827, -21.346307934171893, -21.515015189418286,
                                 -21.63542725556709, -21.632506477532665, -21.732434649625926, -21.553376717610043,
                                 -21.59711848953663, -20.106703324418167, -21.694702593180377, -21.671431372292826,
                                 -21.691958863706233, -21.65798436557029, -21.181150253104924, -21.51224816066623,
                                 -21.66259454427485, -21.61722904073145, -21.532362946278372, -21.704361648329638,
                                 -21.492660873484482, -21.6167506497758]
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
    data_with_layer_adding = [-22.04285762504142, -22.043051631536475, -22.04895012299256, -22.046507969620638,
                              -22.055939121984956, -22.038953054445393, -22.036816805406264, -22.049267354263222,
                              -22.02147927129723, -22.02173917433623, -22.03748908017386, -22.034350190786153,
                              -22.004321225132006, -22.03838300729901, -22.043082394138807, -22.018789160515954,
                              -22.040763126087693, -22.031026739803547, -22.046744614415246, -22.054011496757465,
                              -22.04681166036868, -22.01876879302441, -22.022424345706686, -22.028495542581346,
                              -22.0331529224908, -22.023166523560953, -22.031526589867823, -21.858514861437293,
                              -22.05837374675301, -22.02925826826033]
    data_without_layer_adding = [-21.618530732586365, -21.77193049540494, -21.70085885399603, -21.71811533078925,
                                 -21.731695841841024, -21.373700294426055, -21.77882427409699, -21.509270286134853,
                                 -21.655463592564406, -21.790462830042827, -21.346307934171893, -21.515015189418286,
                                 -21.63542725556709, -21.632506477532665, -21.732434649625926, -21.553376717610043,
                                 -21.59711848953663, -20.106703324418167, -21.694702593180377, -21.671431372292826,
                                 -21.691958863706233, -21.65798436557029, -21.181150253104924, -21.51224816066623,
                                 -21.66259454427485, -21.61722904073145, -21.532362946278372, -21.704361648329638,
                                 -21.492660873484482, -21.6167506497758]
    exact = -22.138

    mean_with_layer_adding = np.mean(data_with_layer_adding)
    var_with_layer_adding = np.var(data_with_layer_adding)
    mean_without_layer_adding = np.mean(data_without_layer_adding)
    var_without_layer_adding = np.var(data_without_layer_adding)

    with open("results/COBYLA_layer_adding/analysis/COBYLA_t-test.txt","w") as f:
        f.write(f"without layer adding: mean={mean_without_layer_adding}, var={var_without_layer_adding}")
        f.write(f"with layer adding: mean={mean_with_layer_adding}, var={var_with_layer_adding}")
        # Welch's t-test
        f.write(str(stats.ttest_ind(data_with_layer_adding, data_without_layer_adding, equal_var=False)))



if __name__ == "__main__":
    #tune_adam()
    #test_compare_ansatze()
    #testDynamicRunner()
    #check_local_minima_hypothesis()
    #interrupt_with_no_mod_test()
    #analysis_box_plots()
    analysis_t_test()
