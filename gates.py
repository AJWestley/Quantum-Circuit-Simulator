import numpy as np
from constants import Gates

def X_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates a Pauli-X gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the X gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.Pauli_X)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def Y_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates a Pauli-Y gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the Y gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.Pauli_Y)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def Z_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates a Pauli-Z gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the Z gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.Pauli_Z)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def H_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates a Hadamard gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the Hadamard gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.Hadamard)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def S_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates an S gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubis (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the S gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.S)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def T_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates a T gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the T gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.T)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def I_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates an Identity gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
        This field exists for consistency with other gate functions, but the Identity gate does not change the state.
    
    Returns:
    np.ndarray: The full unitary matrix representing the Identity gate applied to the specified qubit.
    """
    
    return np.eye(2 ** n_qubits, dtype=complex)

def T_dagger_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates a T-dagger gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the T-dagger gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.T.conj())
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def S_dagger_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates an S-dagger gate for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the S-dagger gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.S.conj().T)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def SX_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates an SX gate (square root of X) for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the SX gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.SX)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def SX_dagger_gate(n_qubits: int, qubits: int | list[int] | range) -> np.ndarray:
    """
    Creates an SX-dagger gate (square root of X dagger) for a specific qubit, or qubit range in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    
    Returns:
    np.ndarray: The full unitary matrix representing the SX-dagger gate applied to the specified qubit.
    """
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(Gates.SX.conj().T)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def CNOT_gate(
        n_qubits: int, 
        control: int, 
        target: int
        ) -> np.ndarray:
    """
    Creates a CNOT gate for a specific control and target qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    control (int): The index of the control qubit.
    target (int): The index of the target qubit.
    
    Returns:
    np.ndarray: The full unitary matrix representing the CNOT gate applied to the specified qubits.
    """

    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Projector for control = 0 → identity
    ops0 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(Gates.Identity)
    term0 = kron_op(ops0)

    # Projector for control = 1 → apply X to target
    ops1 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(Gates.Pauli_X)
        else:
            ops1.append(Gates.Identity)
    term1 = kron_op(ops1)

    return term0 + term1

def phase_gate(n_qubits: int, qubits: int | list[int] | range, phase: float) -> np.ndarray:
    """
    Creates a phase gate for a specific qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    phase (float): The phase angle in radians to apply.
    
    Returns:
    np.ndarray: The full unitary matrix representing the phase gate applied to the specified qubit.
    """
    
    P = np.array([[1, 0], [0, np.exp(1j * phase)]], dtype=complex)
    
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(P)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def RX_gate(n_qubits: int, qubits: int | list[int] | range, theta: float) -> np.ndarray:
    """
    Creates an RX gate for a specific qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    theta (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: The full unitary matrix representing the RX gate applied to the specified qubit.
    """
   
    RX = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                          [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(RX)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def RY_gate(n_qubits: int, qubits: int | list[int] | range, theta: float) -> np.ndarray:
    """
    Creates an RY gate for a specific qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    theta (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: The full unitary matrix representing the RY gate applied to the specified qubit.
    """
    
    RY = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                          [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(RY)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def RZ_gate(n_qubits: int, qubits: int | list[int] | range, theta: float) -> np.ndarray:
    """
    Creates an RZ gate for a specific qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubits (int | list[int] | range): The index of the qubit to apply the gate to, or a list/range of qubits.
    theta (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: The full unitary matrix representing the RZ gate applied to the specified qubit.
    """
   
    RZ = np.array([[np.exp(-1j * theta / 2), 0],
                          [0, np.exp(1j * theta / 2)]], dtype=complex)
    
    if isinstance(qubits, int):
        qubits = [qubits]

    for qubit in qubits:
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("Qubit index out of range.")

    ops = []
    for i in reversed(range(n_qubits)):
        if i in qubits:
            ops.append(RZ)
        else:
            ops.append(Gates.Identity)

    full_gate = ops[0]
    for op in ops[1:]:
        full_gate = np.kron(full_gate, op)

    return full_gate

def CZ_gate(n_qubits: int, control: int, target: int) -> np.ndarray:
    """
    Creates a Controlled-Z (CZ) gate for a specific control and target qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    control (int): The index of the control qubit.
    target (int): The index of the target qubit.
    
    Returns:
    np.ndarray: The full unitary matrix representing the CZ gate applied to the specified qubits.
    """
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Projector for control = 0 → identity
    ops0 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(Gates.Identity)
    term0 = kron_op(ops0)

    # Projector for control = 1 → apply X to target
    ops1 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(Gates.Pauli_Z)
        else:
            ops1.append(Gates.Identity)
    term1 = kron_op(ops1)

    return term0 + term1

def CY_gate(n_qubits: int, control: int, target: int) -> np.ndarray:
    """
    Creates a Controlled-Y (CY) gate for a specific control and target qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    control (int): The index of the control qubit.
    target (int): The index of the target qubit.
    
    Returns:
    np.ndarray: The full unitary matrix representing the CY gate applied to the specified qubits.
    """
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Projector for control = 0 → identity
    ops0 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(Gates.Identity)
    term0 = kron_op(ops0)

    # Projector for control = 1 → apply X to target
    ops1 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(Gates.Pauli_Y)
        else:
            ops1.append(Gates.Identity)
    term1 = kron_op(ops1)

    return term0 + term1

def CH_gate(n_qubits: int, control: int, target: int) -> np.ndarray:
    """
    Creates a Controlled-Hadamard (CH) gate for a specific control and target qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    control (int): The index of the control qubit.
    target (int): The index of the target qubit.
    
    Returns:
    np.ndarray: The full unitary matrix representing the CH gate applied to the specified qubits.
    """

    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Projector for control = 0 → identity
    ops0 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(Gates.Identity)
    term0 = kron_op(ops0)

    # Projector for control = 1 → apply H to target
    ops1 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(Gates.Hadamard)
        else:
            ops1.append(Gates.Identity)
    term1 = kron_op(ops1)

    return term0 + term1

def CRX_gate(n_qubits: int, control: int, target: int, theta: float) -> np.ndarray:
    """
    Creates a Controlled-RX gate for a specific control and target qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    control (int): The index of the control qubit.
    target (int): The index of the target qubit.
    theta (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: The full unitary matrix representing the CRX gate applied to the specified qubits.
    """
    
    RX = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                          [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Projector for control = 0 → identity
    ops0 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(Gates.Identity)
    term0 = kron_op(ops0)

    # Projector for control = 1 → apply X to target
    ops1 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(RX)
        else:
            ops1.append(Gates.Identity)
    term1 = kron_op(ops1)

    return term0 + term1

def CRY_gate(n_qubits: int, control: int, target: int, theta: float) -> np.ndarray:
    """
    Creates a Controlled-RY gate for a specific control and target qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    control (int): The index of the control qubit.
    target (int): The index of the target qubit.
    theta (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: The full unitary matrix representing the CRY gate applied to the specified qubits.
    """
    
    RY = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                          [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Projector for control = 0 → identity
    ops0 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(Gates.Identity)
    term0 = kron_op(ops0)

    # Projector for control = 1 → apply X to target
    ops1 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(RY)
        else:
            ops1.append(Gates.Identity)
    term1 = kron_op(ops1)

    return term0 + term1

def CRZ_gate(n_qubits: int, control: int, target: int, theta: float) -> np.ndarray:
    """
    Creates a Controlled-RZ gate for a specific control and target qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    control (int): The index of the control qubit.
    target (int): The index of the target qubit.
    theta (float): The rotation angle in radians.
    
    Returns:
    np.ndarray: The full unitary matrix representing the CRZ gate applied to the specified qubits.
    """
    
    RZ = np.array([[np.exp(-1j * theta / 2), 0],
                          [0, np.exp(1j * theta / 2)]], dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    # Projector for control = 0 → identity
    ops0 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops0.append(P0)
        else:
            ops0.append(Gates.Identity)
    term0 = kron_op(ops0)

    # Projector for control = 1 → apply X to target
    ops1 = []
    for i in reversed(range(n_qubits)):
        if i == control:
            ops1.append(P1)
        elif i == target:
            ops1.append(RZ)
        else:
            ops1.append(Gates.Identity)
    term1 = kron_op(ops1)

    return term0 + term1

def U_gate(n_qubits: int, qubit: int, matrix: np.ndarray) -> np.ndarray:
    """
    Creates a unitary gate for a specific qubit in an n-qubit system.
    
    Parameters:
    n_qubits (int): Total number of qubits in the system.
    qubit (int): The index of the qubit to apply the gate to.
    matrix (np.ndarray): The unitary matrix to apply to the specified qubit.
    
    Returns:
    np.ndarray: The full unitary matrix representing the gate applied to the specified qubit.
    """
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError("Qubit index out of range.")
    
    validate_gate(matrix, n_qubits)
    
    full_gate = np.eye(2 ** n_qubits, dtype=complex)
    
    full_gate[2 ** qubit:2 ** (qubit + 1), 2 ** qubit:2 ** (qubit + 1)] = matrix
    
    return full_gate.reshape((2 ** n_qubits, 2 ** n_qubits))

def validate_gate(gate: np.ndarray, num_qubits: int):
    """
    Validates a quantum gate to ensure it is a valid unitary matrix for the specified number of qubits.
    Parameters:
    gate (np.ndarray): The quantum gate to validate, must be a unitary matrix.
    num_qubits (int): The number of qubits in the quantum state.
    """
    if not isinstance(gate, np.ndarray):
        raise TypeError("Gate must be a numpy array.")
    if gate.ndim != 2:
        raise ValueError("Gate must be a 2D array.")
    if gate.shape[0] != gate.shape[1]:
        raise ValueError("Gate must be a square matrix.")
    if gate.shape[0] != 2 ** num_qubits:
        raise ValueError("Gate size must match the number of qubits in the state.")
    if not np.allclose(gate.conj().T @ gate, np.eye(gate.shape[0])):
        raise ValueError("Gate must be unitary.")
    
def kron_op(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result