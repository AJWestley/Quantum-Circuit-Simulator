from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from constants import Basis, QuantumGates
from utilities import bloch_axes, bloch_vector

class Qubit():
    def __init__(self, state: np.ndarray = Basis.ket_0) -> None:
        self.state = state
    
    # Quantum Operations
    
    def X(self) -> None:
        ''' Applies a Pauli X Gate to the qubit. '''
        self.state = QuantumGates.Pauli_X.dot(self.state)
    
    def Y(self) -> None:
        ''' Applies a Pauli Y Gate to the qubit. '''
        self.state = QuantumGates.Pauli_Y.dot(self.state)
    
    def Z(self) -> None:
        ''' Applies a Pauli Z Gate to the qubit. '''
        self.state = QuantumGates.Pauli_Z.dot(self.state)
    
    def I(self) -> None:
        ''' Applies an Identity Gate to the qubit. '''
        self.state = QuantumGates.Identity.dot(self.state)
    
    def H(self) -> None:
        ''' Applies a Hadamard Gate to the qubit. '''
        self.state = QuantumGates.Hadamard.dot(self.state)
    
    def S(self) -> None:
        ''' Applies an S Gate to the qubit. '''
        self.state = QuantumGates.S.dot(self.state)
    
    def T(self) -> None:
        ''' Applies a T Gate to the qubit. '''
        self.state = QuantumGates.T.dot(self.state)
    
    # Arbitrary Rotations
    
    def Rx(self, angle: float, unit: Literal['Radians', 'Degrees'] = 'Radians') -> None:
        ''' Rotates the qubit along the x-axis by a given angle. '''
        
        if unit == 'Degrees':
            angle = np.deg2rad(angle)
        R = np.array([
            [np.cos(angle / 2), -1j * np.sin(angle / 2)],
            [-1j * np.sin(angle / 2), np.cos(angle / 2)]
        ])
        self.state = R.dot(self.state)
    
    def Ry(self, angle: float, unit: Literal['Radians', 'Degrees'] = 'Radians') -> None:
        ''' Rotates the qubit along the y-axis by a given angle. '''
        
        if unit == 'Degrees':
            angle = np.deg2rad(angle)
        R = np.array([
            [np.cos(angle / 2), -np.sin(angle / 2)],
            [np.sin(angle / 2), np.cos(angle / 2)]
        ])
        self.state = R.dot(self.state)
    
    def Rz(self, angle: float, unit: Literal['Radians', 'Degrees'] = 'Radians') -> None:
        ''' Rotates the qubit along the z-axis by a given angle. '''
        if unit == 'Degrees':
            angle = np.deg2rad(angle)
        R = np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ])
        self.state = R.dot(self.state)
    
    def flip(self) -> None:
        ''' Flips the qubit using the Pauli X Gate. '''
        self.X()
    
    # Observations and Probabilities
    
    def observe(self) -> np.ndarray:
        ''' Returns a 1 or 0 according the the qubit's current state then collapses the qubit to the observed state. '''
        if np.random.uniform(0, 1) < self.p0():
            self.state = Basis.ket_0
        else:
            self.state = Basis.ket_1
        return self.state
    
    def sample(self, n = 1) -> np.ndarray | int:
        ''' Generates n sample observations without collapsing the qubit's state. '''
        if n < 1:
            raise ValueError('Cannot sample less than once.')
        return np.random.binomial(1, self.p1(), n)
    
    def p0(self) -> float:
        ''' Returns the probability of observing a 0 in the qubit's current state. '''
        return np.power(np.abs(self.alpha()), 2)
    
    def p1(self) -> float:
        ''' Returns the probability of observing a 1 in the qubit's current state. '''
        return np.power(np.abs(self.alpha()), 2)
    
    def alpha(self) -> complex:
        ''' Returns the linear coefficient of the |0⟩ state '''
        return self.state[0]
    
    def beta(self) -> complex:
        ''' Returns the linear coefficient of the |1⟩ state '''
        return self.state[1]
    
    def reset(self) -> None:
        ''' Returns the qubit to the |0⟩ state. '''
        self.state = Basis.ket_0
    
    # Display
    
    def __str__(self) -> str:
        if self.beta().real <= 1e-12 and self.beta().imag <= 1e-12:
            return "|0⟩"
        if self.alpha().real <= 1e-12 and self.alpha().imag <= 1e-12:
            return "|1⟩"
        return f"{self.alpha()}|0⟩ + {self.beta()}|1⟩"
    
    def show(self) -> None:
        ''' Plots the qubit's current state on a Bloch Sphere. '''
        
        fig, ax = bloch_axes()
        
        origin = np.array([0, 0, 0])
        vec = bloch_vector(self.alpha(), self.beta())
        
        ax.quiver(*origin, *vec, color='r')
        
        plt.show()

q = Qubit()

q.Ry(15, 'Degrees')
q.H()
q.Rz(30, 'Degrees')
q.Ry(-15, 'Degrees')

q.show()