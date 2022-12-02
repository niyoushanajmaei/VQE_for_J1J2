from src.vqe_runner import VqeRunner
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

    vqe_runner = VqeRunner(m, n, J1, J2, h=0, simulation=True, seed=seed, ansatz=ansatz)
    result = vqe_runner.run_vqe(monitor=True)
    print(result)
