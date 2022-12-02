from src.vqe_runner import VqeRunner

if __name__=="__main__":
    seed = 50
    ansatz = "sample_ansatz"
    model = "j1j2"
    lattice_size = [3,3]

    vqe_runner = VqeRunner(lattice_size, simulation=True, seed=seed, ansatz=ansatz)
    results = vqe_runner.run_vqe()
    print(results)
