from qiskit import QuantumCircuit, Aer, execute

def run_quantum_experiment(b_measurement_basis):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    if b_measurement_basis == 'Z':
        qc.measure(1, 1)
    elif b_measurement_basis == 'X':
        qc.h(1)
        qc.measure(1, 1)
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1)
    result = job.result().get_counts()
    key = list(result.keys())[0]
    return int(key[1]), int(key[0])  # (A, B)
