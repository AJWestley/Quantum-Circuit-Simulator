import numpy as np

class Gates:
    '''A class containing the matrices for various quantum gates'''
    
    Hadamard = np.sqrt(0.5) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex)
    
    Pauli_X = np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)
    
    Pauli_Y = np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex)
    
    Pauli_Z = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)
    
    Identity = np.array([
        [1, 0],
        [0, 1]
    ], dtype=complex)
    
    S = np.array([
        [1, 0],
        [0, 1j]
    ], dtype=complex)
    
    T = np.array([
        [1, 0],
        [0, np.exp(1j * np.pi / 4)]
    ], dtype=complex)