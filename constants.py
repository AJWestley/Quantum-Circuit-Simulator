import numpy as np

# Computational Basis

class Basis:
    ket_0 = np.array([1 + 0j, 0 + 0j])
    ket_1 = np.array([0 + 0j, 1 + 0j])

# Quantum Gates

class QuantumGates:
    
    Hadamard = np.sqrt(0.5) * np.array([
        [1, 1],
        [1, -1]
    ])
    
    Pauli_X = np.array([
        [0, 1],
        [1, 0]
    ])
    
    Pauli_Y = np.array([
        [0, -1j],
        [1j, 0]
    ])
    
    Pauli_Z = np.array([
        [1, 0],
        [0, -1]
    ])
    
    Identity = np.array([
        [1, 0],
        [0, 1]
    ])
    
    S = np.array([
        [1, 0],
        [0, 1j]
    ])
    
    T = np.array([
        [1, 0],
        [0, np.exp(1j * np.pi / 4)]
    ])