from qiskit import Aer, QuantumCircuit
from src.vqe_runner import VqeRunner
from src.model import Model


def test_backend():
    # Create circuit
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.measure_all()

    backend = Aer.get_backend('aer_simulator')
    backend._configuration.max_shots = 1
    shots = 10000

    job = backend.run(circ, shots=shots)
    counts = job.result().get_counts(0)
    print(counts)


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
    #result = vqe_runner.run_vqe(monitor=True)

    #print(result)

    vqe_runner.compare_optimizers_and_ansatze()

    exact_result = Model.get_exact_energy(vqe_runner.hamiltonian_matrix)
    print(f"exact_result: {exact_result}")

    #test_backend()
