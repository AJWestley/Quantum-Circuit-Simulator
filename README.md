# Quantum Circuit Simulator

This module provides a lightweight simulator for quantum circuits, using a `QuantumState` class to represent and manipulate quantum states via unitary gates and measurements. States are represented as complex vectors, and gates are applied via matrix multiplication. The simulator allows circuits to be built in a functional manner through function chaining.

---

## Basic Usage

```python
from quantum_state import QuantumState

# Initialize a 3-qubit state |000⟩
qs = QuantumState(3)

# Build a simple Bell state circuit
qs = qs.H(0).CNOT(0, 1)

# Measure all qubits
result_register = []
collapsed_state = qs.measure_all(register=result_register)
print("Measured bits:", result_register)
```

All methods return a new `QuantumState` object, making function chaining natural and side-effect free.

---

## Method Groups

### Initialization

* `QuantumState(n: int)` — Initialize |0...0⟩ state.
* `QuantumState(state: np.ndarray)` — Load a custom state vector.

---

### Gate Application

All gate methods accept:

* `qubits: int | list[int] | range` for single-qubit gates.
* `control_qubit: int, target_qubit: int` for controlled gates.
* `theta: float` or `phase: float` for parameterized rotations and phase gates.

#### Single-Qubit Gates

```python
qs.X(0)    # Pauli-X
qs.Y([0, 2])
qs.Z(range(3))
qs.H(0)
qs.S(1)
qs.T([0, 2])
qs.Sdag(0)
qs.Tdag(0)
qs.SX(1)
qs.SXdag(1)
```

#### Parameterized Single-Qubit Rotations

```python
qs.RX(0, theta=np.pi/2)
qs.RY([1,2], theta=np.pi/4)
qs.RZ(0, theta=np.pi)
qs.P(0, phase=np.pi)  # Phase gate
```

#### Two-Qubit Gates

```python
qs.CNOT(0, 1)
qs.SWAP(1, 2)
qs.CZ(0, 2)
qs.CY(0, 1)
qs.CX(0, 2)
qs.CH(2, 1)
qs.CSX(0, 1)
qs.DCX(0, 1)
```

#### Controlled Rotations

```python
qs.CRX(0, 1, theta=np.pi/2)
qs.CRY(0, 1, theta=np.pi/2)
qs.CRZ(0, 1, theta=np.pi/2)
```

#### Multi-Qubit Gates

```python
# Toffoli and Approximate Toffoli
qs.CCX(0, 1, 2) # controls = 0 and 1, target = 2
qs.RCCX(0, 1, 2)

# Rotations
qs.RXX(0, 1, pi / 2) # qubits = 0 and 1, theta = pi / 2
qs.RYY(0, 1, pi / 2)
qs.RZZ(0, 1, pi / 2)
```

#### Unary Gates

```python
qs.U(np.array([[1, 1], [1, -1]]), 1)
```

---

### Measurement

```python
qs.measure(0, register)      # Measure one qubit
qs.measure_all(register)     # Measure all qubits
```

Both collapse the state and return a new `QuantumState` consistent with the result.

---

### QASM Translation

You can translate the built circuit into QASM by setting the `qasm_tape=True` in the constructor.
This allows the `read_tape()` function to be used, which will return a string containing QASM code for your circuit.

```python
qs = QuantumState(2, qasm_tape=True)
qasm_code = qs.read_tape()
```

---

## Features to Be Added

* **More Gates** 
    - CSXdag
    - CS
    - CSdag
    - CT
    - CTdag
    - RC3X
    - CSWAP
* **Circuit Visualiser**

---

