from qiskit import Aer, QuantumCircuit
from src.VQERunner import VQERunner
from src.model import Model


if __name__=="__main__":
    seed = 50
    ansatz = "sample_ansatz"
    model = "j1j2"

    m = 3
    n = 3
    J1 = 1
    J2 = 0.5

    # print(Model.getHamiltonian_J1J2_2D(m,n,J1,J2))

    vqeRunner = VQERunner(m, n, J1, J2, h=0, simulation=True, seed=seed, ansatz=ansatz)
    #result = vqe_runner.run_vqe(monitor=True)

    #print(result)

    vqeRunner.compare_Optimizers_And_Ansatze()

    exactResult = Model.getExactEnergy(vqeRunner.hamiltonianMatrix)
    print(f"exactResult: {exactResult}")

    #test_backend()
