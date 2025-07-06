import numpy as np
from gates import *

class QuantumState:
    def __init__(self, state: np.ndarray | int = None):
        '''Initializes a quantum state.
        Parameters:
        state (np.ndarray | int): The initial quantum state as a numpy array or the number of qubits to initialize.
        If an integer is provided, it initializes the state to |0...0> with the specified number of qubits.

        Raises:
        ValueError: If the provided state is not a valid quantum state or if the number of qubits is less than 1.
        TypeError: If the provided state is not a numpy array or an integer.
        Example:
        >>> qs = QuantumState(3)  # Initializes a 3-qubit state |000>
        >>> qs = QuantumState(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex))  # Initializes a specific state
        >>> qs = QuantumState()  # Initializes a single qubit state |0>.
        '''
        if state is None:
            state = initialize_state(1)
        elif isinstance(state, int):
            if state < 1:
                raise ValueError("Number of qubits must be at least 1.")
            state = initialize_state(state)
        self.num_qubits = get_num_qubits(state)
        self.state = state
        
    def __str__(self):
        return f"QuantumState({self.state})"
    
    def __repr__(self):
        return f"QuantumState({self.state})"
    
    def __eq__(self, other):
        if not isinstance(other, QuantumState):
            return False
        return np.array_equal(self.state, other.state)
    
    def __ne__(self, other):
        if not isinstance(other, QuantumState):
            return True
        return not np.array_equal(self.state, other.state)
    
    def __len__(self):
        '''
        Returns the number of basis states in the quantum state.
        
        Returns:
        int: The number of basis states, which is 2 raised to the power of the number of qubits.
        '''
        return 2 ** self.num_qubits
    
    def __getitem__(self, index: int) -> complex:
        '''
        Returns the amplitude of the specified basis state.
        
        Parameters:
        index (int): The index of the basis state to retrieve.
        
        Returns:
        complex: The amplitude of the specified basis state.
        '''
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range for quantum state.")
        return self.state[index]
    
    def __setitem__(self, index: int, value: complex):
        '''
        Sets the amplitude of the specified basis state.
        
        Parameters:
        index (int): The index of the basis state to set.
        value (complex): The new amplitude for the specified basis state.
        '''
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range for quantum state.")
        self.state[index] = value
    
    def get_probabilities(self) -> np.ndarray:
        '''
        Returns the probabilities of measuring each basis state.
        
        Returns:
        np.ndarray: An array of probabilities for each basis state.
        '''
        return np.abs(self.state).astype(float) ** 2
    
    def U(self, gate: np.ndarray, qubit: int = 0) -> 'QuantumState':
        '''
        Applies a quantum gate to the quantum state.

        Parameters:
        gate (np.ndarray): The quantum gate to apply, must be a unitary matrix.
        qubit (int, optional): The index of the qubit to apply the gate to. Default 0.

        Returns:
        QuantumState: A new quantum state after applying the gate.
        '''

        k = int(np.log2(gate.shape[0]))
        end = qubit + k

        if qubit < 0 or end > self.num_qubits:
            raise ValueError(f"Cannot apply gate to qubit range [{qubit}, {end}); out of bounds.")
        if gate.shape != (2**k, 2**k):
            raise ValueError("Gate must be a square matrix of size 2^k × 2^k.")

        validate_gate(gate, k)  # Optional: checks unitarity etc.

        # Build full operator: I ⊗ ... ⊗ G ⊗ ... ⊗ I
        I = np.eye(2, dtype=complex)

        # Create operator list, right-to-left (LSB = qubit 0)
        ops = []
        for i in reversed(range(self.num_qubits)):
            if qubit <= i < end:
                ops.append(None)  # placeholder for multi-qubit block
            else:
                ops.append(I)

        # Insert the gate block
        # Replace the first occurrence of k consecutive None with the full gate
        gate_inserted = False
        i = 0
        while i <= len(ops) - k:
            if all(x is None for x in ops[i:i+k]):
                ops[i:i+k] = [gate]
                gate_inserted = True
                break
            i += 1

        if not gate_inserted:
            raise RuntimeError("Failed to embed the gate correctly in the tensor product.")

        # Kronecker product chain
        full_gate = ops[0]
        for op in ops[1:]:
            full_gate = np.kron(full_gate, op)

        # Apply to state
        new_state = full_gate @ self.state
        return QuantumState(new_state)
        
    def X(self, qubits: int | list[int] | range):
        '''
        Applies the Pauli-X gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the X gate.
        '''
        
        gate = X_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def Y(self, qubits: int | list[int] | range):
        '''
        Applies the Pauli-Y gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the Y gate.
        '''

        gate = Y_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def Z(self, qubits: int | list[int] | range):
        '''
        Applies the Pauli-Z gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the Z gate.
        '''
        
        gate = Z_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def H(self, qubits: int | list[int] | range):
        '''
        Applies the Hadamard gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the Hadamard gate.
        '''

        gate = H_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def S(self, qubits: int | list[int] | range):
        '''
        Applies the S gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the S gate.
        '''
        
        gate = S_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def T(self, qubits: int | list[int] | range):
        '''
        Applies the T gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the T gate.
        '''
        
        gate = T_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def Sdag(self, qubits: int | list[int] | range):
        '''
        Applies the S-dagger gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the S-dagger gate.
        '''
        
        gate = S_dagger_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def Tdag(self, qubits: int | list[int] | range):
        '''
        Applies the T-dagger gate to a specific qubit.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.

        Returns:
        QuantumState: The new quantum state after applying the T-dagger gate.
        '''
        
        gate = T_dagger_gate(self.num_qubits, qubits)
        return self.U(gate)
    
    def CNOT(self, control_qubit: int, target_qubit: int):
        '''
        Applies the CNOT gate with a control qubit and a target qubit.
        
        Parameters:
        control_qubit (int): The index of the control qubit.
        target_qubit (int): The index of the target qubit.
        '''
        
        gate = CNOT_gate(self.num_qubits, control_qubit, target_qubit)
        return self.U(gate)
    
    def P(self, qubits: int | list[int] | range, phase: float):
        '''
        Applies a phase gate to a specific qubit or qubits.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.
        phase (float): The phase angle in radians to apply.

        Returns:
        QuantumState: The new quantum state after applying the phase gate.
        '''
        
        gate = phase_gate(self.num_qubits, qubits, phase)
        return self.U(gate)
    
    def RX(self, qubits: int | list[int] | range, theta: float):
        '''
        Applies the RX gate to a specific qubit or qubits.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.
        theta (float): The rotation angle in radians.

        Returns:
        QuantumState: The new quantum state after applying the RX gate.
        '''
        
        gate = RX_gate(self.num_qubits, qubits, theta)
        return self.U(gate)
    
    def RY(self, qubits: int | list[int] | range, theta: float):
        '''
        Applies the RY gate to a specific qubit or qubits.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.
        theta (float): The rotation angle in radians.

        Returns:
        QuantumState: The new quantum state after applying the RY gate.
        '''
        
        gate = RY_gate(self.num_qubits, qubits, theta)
        return self.U(gate)
    
    def RZ(self, qubits: int | list[int] | range, theta: float):
        '''
        Applies the RZ gate to a specific qubit or qubits.
        
        Parameters:
        qubits (int | list[int] | range): The index or indices of the qubit(s) to apply the gate to.
        If a range is provided, applies the gate to all qubits in that range.
        theta (float): The rotation angle in radians.

        Returns:
        QuantumState: The new quantum state after applying the RZ gate.
        '''
        
        gate = RZ_gate(self.num_qubits, qubits, theta)
        return self.U(gate)
    
    def SWAP(self, qubit1: int, qubit2: int):
        '''
        Applies the SWAP gate to two specific qubits.
        
        Parameters:
        qubit1 (int): The index of the first qubit.
        qubit2 (int): The index of the second qubit.
        
        Returns:
        QuantumState: The new quantum state after applying the SWAP gate.
        '''
        
        if qubit1 < 0 or qubit1 >= self.num_qubits or qubit2 < 0 or qubit2 >= self.num_qubits:
            raise ValueError("Qubit indices out of range.")
        
        return self.CNOT(qubit1, qubit2).CNOT(qubit2, qubit1).CNOT(qubit1, qubit2)
    
    def CZ(self, control_qubit: int, target_qubit: int):
        '''
        Applies the Controlled-Z (CZ) gate with a control qubit and a target qubit.
        
        Parameters:
        control_qubit (int): The index of the control qubit.
        target_qubit (int): The index of the target qubit.
        
        Returns:
        QuantumState: The new quantum state after applying the CZ gate.
        '''
        
        gate = CZ_gate(self.num_qubits, control_qubit, target_qubit)
        return self.U(gate)
    
    def CY(self, control_qubit: int, target_qubit: int):
        '''
        Applies the Controlled-Y (CY) gate with a control qubit and a target qubit.
        
        Parameters:
        control_qubit (int): The index of the control qubit.
        target_qubit (int): The index of the target qubit.
        
        Returns:
        QuantumState: The new quantum state after applying the CY gate.
        '''
        
        gate = CY_gate(self.num_qubits, control_qubit, target_qubit)
        return self.U(gate)
    
    def CX(self, control_qubit: int, target_qubit: int):
        '''
        Applies the Controlled-X (CX) gate (CNOT) with a control qubit and a target qubit.
        
        Parameters:
        control_qubit (int): The index of the control qubit.
        target_qubit (int): The index of the target qubit.
        
        Returns:
        QuantumState: The new quantum state after applying the CX gate.
        '''
        
        return self.CNOT(control_qubit, target_qubit)
    
    def CRX(self, control_qubit: int, target_qubit: int, theta: float):
        '''
        Applies the Controlled-RX gate with a control qubit and a target qubit.
        
        Parameters:
        control_qubit (int): The index of the control qubit.
        target_qubit (int): The index of the target qubit.
        theta (float): The rotation angle in radians.
        
        Returns:
        QuantumState: The new quantum state after applying the CRX gate.
        '''
        
        gate = CRX_gate(self.num_qubits, control_qubit, target_qubit, theta)
        return self.U(gate)
    
    def CRY(self, control_qubit: int, target_qubit: int, theta: float):
        '''
        Applies the Controlled-RY gate with a control qubit and a target qubit.
        
        Parameters:
        control_qubit (int): The index of the control qubit.
        target_qubit (int): The index of the target qubit.
        theta (float): The rotation angle in radians.
        
        Returns:
        QuantumState: The new quantum state after applying the CRY gate.
        '''
        
        gate = CRY_gate(self.num_qubits, control_qubit, target_qubit, theta)
        return self.U(gate)
    
    def CRZ(self, control_qubit: int, target_qubit: int, theta: float):
        '''
        Applies the Controlled-RZ gate with a control qubit and a target qubit.
        
        Parameters:
        control_qubit (int): The index of the control qubit.
        target_qubit (int): The index of the target qubit.
        theta (float): The rotation angle in radians.
        
        Returns:
        QuantumState: The new quantum state after applying the CRZ gate.
        '''
        
        gate = CRZ_gate(self.num_qubits, control_qubit, target_qubit, theta)
        return self.U(gate)
    
    def measure(self, qubit: int, register: list = None) -> 'QuantumState':
        '''
        Measures the state of a specific qubit and collapses the quantum state.
        
        Parameters:
        qubit (int): The index of the qubit to measure.
        register (list, optional): A list to store the measurement result. If provided, the result will be appended to this list.
        
        Returns:
        QuantumState: A new quantum state after the measurement, with the state collapsed to the measured result.
        '''
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError("Qubit index out of range.")
        
        n = self.num_qubits
        dim = 2 ** n
        state = self.state.copy()
        
        # Masks to identify indices where qubit = 0 or 1
        zero_mask = []
        one_mask = []

        for i in range(dim):
            if ((i >> qubit) & 1) == 0:
                zero_mask.append(i)
            else:
                one_mask.append(i)

        # Compute probabilities
        prob_0 = np.sum(np.abs(state[zero_mask]) ** 2)
        prob_1 = np.sum(np.abs(state[one_mask]) ** 2)

        if prob_0 + prob_1 == 0:
            raise ValueError("Invalid quantum state — total probability is zero.")

        # Sample measurement result
        result = np.random.choice([0, 1], p=[prob_0, prob_1])
        if register is not None:
            register.append(result)

        # Collapse the state
        if result == 0:
            state[one_mask] = 0
            norm = np.sqrt(prob_0)
        else:
            state[zero_mask] = 0
            norm = np.sqrt(prob_1)

        # Renormalize
        if norm == 0:
            raise ValueError("Collapse resulted in zero state.")
        state /= norm

        return QuantumState(state)
    
    def measure_all(self, register: list = None) -> 'QuantumState':
        '''
        Measures all qubits and returns the measurement results.

        Parameters:
        register (list, optional): A list to store the measurement results. If provided, the results will be appended to this list.
        
        Returns:
        QuantumState: A new quantum state after measuring all qubits, with the state collapsed to the measured results.
        '''
        state = self.state.copy()
        n = self.num_qubits
        dim = 2 ** n

        # Compute measurement probabilities
        probs = np.abs(state) ** 2
        if not np.isclose(np.sum(probs), 1):
            raise ValueError("State vector is not normalized.")

        # Sample basis state
        outcome_index = np.random.choice(dim, p=probs)

        # Collapse the state
        new_state = np.zeros_like(state)
        new_state[outcome_index] = 1.0 + 0j

        # Decode outcome to bitstring
        bits = [(outcome_index >> i) & 1 for i in range(n)]
        if register is not None:
            register.extend(reversed(bits))  # reverse to match qubit ordering (0 = rightmost)

        return QuantumState(new_state)
    

def initialize_state(num_qubits: int) -> np.ndarray:
    '''
    Initializes a quantum state in the computational basis |0...0>.
    
    Parameters:
    num_qubits (int): The number of qubits in the quantum state.
    
    Returns:
    np.ndarray: The initial quantum state vector.
    '''
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")
    
    state = np.zeros(2 ** num_qubits, dtype=complex)
    state[0] = 1.0  # Initialize to |0...0>
    
    return state
    
def get_num_qubits(state: np.ndarray) -> int:
    '''
    Determines the number of qubits from the initial quantum state.

    Parameters:
    state (np.ndarray): The initial quantum state vector.

    Returns:
    int: The number of qubits in the quantum state.

    Raises:
    TypeError: If the provided state is not a numpy array.
    ValueError: If the provided state is not a valid quantum state vector or if it is not normalized.
    ValueError: If the length of the state vector is not a power of 2.
    '''
    if not isinstance(state, np.ndarray):
        raise TypeError("Initial state must be a numpy array.")
    if state.ndim != 1:
        raise ValueError("Initial state must be a 1D array.")
    if not np.isclose(np.linalg.norm(state), 1):
         raise ValueError("Initial state must be normalized (norm = 1).")
    
    num_qubits = int(np.log2(len(state)))

    if 2 ** num_qubits != len(state):
        raise ValueError("Initial state must be a valid quantum state vector (length must be a power of 2).")
    
    return num_qubits

