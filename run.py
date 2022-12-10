from src.VQERunner import VQERunner
from src.model import Model


if __name__=="__main__":
    seed = 50
    ansatz = "sample_ansatz"
    model = "j1j2"

    m = 2
    n = 2
    J1 = 1
    J2 = 0.5

    # print(Model.getHamiltonian_J1J2_2D(m,n,J1,J2))

    vqe_runner = VQERunner(m, n, J1, J2, h=0, simulation=False, seed=seed, ansatz=ansatz)
    result = vqe_runner.run_vqe(monitor=True)

    print(result)

    #vqe_runner.compare_optimizers_and_ansatze()

    exactResult = Model.getExactEnergy(VQERunner.hamiltonianMatrix)
    print(f"exactResult: {exactResult}")

