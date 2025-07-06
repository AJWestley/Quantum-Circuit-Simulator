import pytest
import numpy as np
from QCircuit.quantum_state import QuantumState

class TestQuantumState:
    def test_initialization_with_int(self):
        qs = QuantumState(2)
        assert isinstance(qs.state, np.ndarray)
        assert qs.state.shape == (4,)
        assert np.allclose(qs.state, np.array([1, 0, 0, 0]))

    def test_initialization_with_array(self):
        qs = QuantumState(np.array([1, 0, 0, 0]))
        assert isinstance(qs.state, np.ndarray)
        assert qs.state.shape == (4,)
        assert np.allclose(qs.state, np.array([1, 0, 0, 0]))
    
    def test_initialization_with_empty(self):
        qs = QuantumState()
        assert isinstance(qs.state, np.ndarray)
        assert qs.state.shape == (2,)
        assert np.allclose(qs.state, np.array([1, 0]))

    def test_initialization_with_invalid_type(self):
        with pytest.raises(TypeError):
            QuantumState("invalid")

    def test_X_int(self):
        qs = QuantumState(2).X(0)
        assert np.allclose(qs.state, np.array([0, 1, 0, 0], dtype=complex))
    
    def test_X_list(self):
        qs = QuantumState(3).X([0, 1])
        expected_state = np.array([0, 0, 0, 1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_X_range(self):
        qs = QuantumState(3).X(range(2))
        expected_state = np.array([0, 0, 0, 1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)

    def test_X_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.X(2)
        with pytest.raises(ValueError):
            qs.X([0, 2])
        with pytest.raises(ValueError):
            qs.X(range(3))
        with pytest.raises(TypeError):
            qs.X("invalid")
        with pytest.raises(ValueError):
            qs.X(-1)
    
    def test_Y_int(self):
        qs = QuantumState(2).Y(0)
        expected_state = np.array([0, 1j, 0, 0])
        assert np.allclose(qs.state, expected_state)

    def test_Y_list(self):
        qs = QuantumState(3).Y([0, 1])
        expected_state = np.array([0, 0, 0, -1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_Y_range(self):
        qs = QuantumState(3).Y(range(2))
        expected_state = np.array([0, 0, 0, -1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)

    def test_Y_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.Y(2)
        with pytest.raises(ValueError):
            qs.Y([0, 2])
        with pytest.raises(ValueError):
            qs.Y(range(3))
        with pytest.raises(TypeError):
            qs.Y("invalid")
        with pytest.raises(ValueError):
            qs.Y(-1)
    
    def test_Z_int(self):
        qs = QuantumState(2).Y(0).Z(0)
        expected_state = np.array([0, -1j, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_Z_list(self):
        qs = QuantumState(3).Y([0, 1]).Z([0, 1])
        expected_state = np.array([0, 0, 0, -1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_Z_range(self):
        qs = QuantumState(3).Y(range(2)).Z(range(2))
        expected_state = np.array([0, 0, 0, -1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_Z_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.Z(2)
        with pytest.raises(ValueError):
            qs.Z([0, 2])
        with pytest.raises(ValueError):
            qs.Z(range(3))
        with pytest.raises(TypeError):
            qs.Z("invalid")
        with pytest.raises(ValueError):
            qs.Z(-1)
    
    def test_H_int(self):
        qs = QuantumState(2).H(0)
        expected_state = np.sqrt(0.5) * np.array([1, 1, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_H_list(self):
        qs = QuantumState(3).H([0, 1])
        expected_state = 0.5 * np.array([1, 1, 1, 1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_H_range(self):
        qs = QuantumState(3).H(range(2))
        expected_state = 0.5 * np.array([1, 1, 1, 1, 0, 0, 0, 0])
        assert np.allclose(qs.state, expected_state)

    def test_H_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.H(2)
        with pytest.raises(ValueError):
            qs.H([0, 2])
        with pytest.raises(ValueError):
            qs.H(range(3))
        with pytest.raises(TypeError):
            qs.H("invalid")
        with pytest.raises(ValueError):
            qs.H(-1)
    
    def test_S_int(self):
        qs = QuantumState(2).H(0).S(0).H(0)
        expected_state = 0.5 * np.array([1 + 1j, 1 - 1j, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_S_list(self):
        qs = QuantumState(2).H([0, 1]).S([0, 1]).H([0, 1])
        expected_state = 0.5 * np.array([1j, 1, 1, -1j])
        assert np.allclose(qs.state, expected_state)
    
    def test_S_range(self):
        qs = QuantumState(2).H(range(2)).S(range(2)).H(range(2))
        expected_state = 0.5* np.array([1j, 1, 1, -1j])
        assert np.allclose(qs.state, expected_state)

    def test_S_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.S(2)
        with pytest.raises(ValueError):
            qs.S([0, 2])
        with pytest.raises(ValueError):
            qs.S(range(3))
        with pytest.raises(TypeError):
            qs.S("invalid")
        with pytest.raises(ValueError):
            qs.S(-1)
    
    def test_T_int(self):
        qs = QuantumState(2).H(0).T(0).H(0)
        expected_state = np.array([0.854 + 0.354j, 0.146 - 0.354j, 0, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_T_list(self):
        qs = QuantumState(2).H([0, 1]).T([0, 1]).H([0, 1])
        expected_state = np.array([0.604 + 0.604j, 0.25 - 0.25j, 0.25 - 0.25j, -0.104 - 0.104j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_T_range(self):
        qs = QuantumState(2).H(range(2)).T(range(2)).H(range(2))
        expected_state = np.array([0.604 + 0.604j, 0.25 - 0.25j, 0.25 - 0.25j, -0.104 - 0.104j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_T_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.T(2)
        with pytest.raises(ValueError):
            qs.T([0, 2])
        with pytest.raises(ValueError):
            qs.T(range(3))
        with pytest.raises(TypeError):
            qs.T("invalid")
        with pytest.raises(ValueError):
            qs.T(-1)
    
    def test_Sdag_int(self):
        qs1 = QuantumState(2).H(0).S(0).Sdag(0)
        qs2 = QuantumState(2).H(0)
        assert np.allclose(qs1.state, qs2.state)

    def test_Sdag_list(self):
        qs1 = QuantumState(2).H([0, 1]).S([0, 1]).Sdag([0, 1])
        qs2 = QuantumState(2).H([0, 1])
        assert np.allclose(qs1.state, qs2.state)

    def test_Sdag_range(self):
        qs1 = QuantumState(2).H(range(2)).S(range(2)).Sdag(range(2))
        qs2 = QuantumState(2).H(range(2))
        assert np.allclose(qs1.state, qs2.state)
    
    def test_Sdag_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.Sdag(2)
        with pytest.raises(ValueError):
            qs.Sdag([0, 2])
        with pytest.raises(ValueError):
            qs.Sdag(range(3))
        with pytest.raises(TypeError):
            qs.Sdag("invalid")
        with pytest.raises(ValueError):
            qs.Sdag(-1)

    def test_Tdag_int(self):
        qs1 = QuantumState(2).H(0).T(0).Tdag(0)
        qs2 = QuantumState(2).H(0)
        assert np.allclose(qs1.state, qs2.state)

    def test_Tdag_list(self):
        qs1 = QuantumState(2).H([0, 1]).T([0, 1]).Tdag([0, 1])
        qs2 = QuantumState(2).H([0, 1])
        assert np.allclose(qs1.state, qs2.state)

    def test_Tdag_range(self):
        qs1 = QuantumState(2).H(range(2)).T(range(2)).Tdag(range(2))
        qs2 = QuantumState(2).H(range(2))
        assert np.allclose(qs1.state, qs2.state)

    def test_Tdag_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.Tdag(2)
        with pytest.raises(ValueError):
            qs.Tdag([0, 2])
        with pytest.raises(ValueError):
            qs.Tdag(range(3))
        with pytest.raises(TypeError):
            qs.Tdag("invalid")
        with pytest.raises(ValueError):
            qs.Tdag(-1)

    def test_SX_int(self):
        qs = QuantumState(2).SX(0)
        expected_state = 0.5 * np.array([1 + 1j, 1 - 1j, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_SX_list(self):
        qs = QuantumState(2).SX([0, 1])
        expected_state = 0.5 * np.array([1j, 1, 1, -1j])
        assert np.allclose(qs.state, expected_state)

    def test_SX_range(self):
        qs = QuantumState(2).SX(range(2))
        expected_state = 0.5 * np.array([1j, 1, 1, -1j])
        assert np.allclose(qs.state, expected_state)

    def test_SX_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.SX(2)
        with pytest.raises(ValueError):
            qs.SX([0, 2])
        with pytest.raises(ValueError):
            qs.SX(range(3))
        with pytest.raises(TypeError):
            qs.SX("invalid")
        with pytest.raises(ValueError):
            qs.SX(-1)

    def test_SXdag_int(self):
        qs1 = QuantumState(2).H(0).SX(0).SXdag(0)
        qs2 = QuantumState(2).H(0)
        assert np.allclose(qs1.state, qs2.state)

    def test_SXdag_list(self):
        qs1 = QuantumState(2).H([0, 1]).SX([0, 1]).SXdag([0, 1])
        qs2 = QuantumState(2).H([0, 1])
        assert np.allclose(qs1.state, qs2.state)

    def test_SXdag_range(self):
        qs1 = QuantumState(2).H(range(2)).SX(range(2)).SXdag(range(2))
        qs2 = QuantumState(2).H(range(2))
        assert np.allclose(qs1.state, qs2.state)
    
    def test_SXdag_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.SXdag(2)
        with pytest.raises(ValueError):
            qs.SXdag([0, 2])
        with pytest.raises(ValueError):
            qs.SXdag(range(3))
        with pytest.raises(TypeError):
            qs.SXdag("invalid")
        with pytest.raises(ValueError):
            qs.SXdag(-1)

    def test_CNOT(self):
        qs = QuantumState(2).H(0).CNOT(0, 1).H(0)
        expected_state = 0.5 * np.array([1, 1, 1, -1])
        assert np.allclose(qs.state, expected_state)
    
    def test_CNOT_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.CNOT(2, 1)
        with pytest.raises(ValueError):
            qs.CNOT(0, 2)
        with pytest.raises(TypeError):
            qs.CNOT([0, 1], 2)
        with pytest.raises(TypeError):
            qs.CNOT("invalid", 1)
        with pytest.raises(ValueError):
            qs.CNOT(0, -1)
        with pytest.raises(ValueError):
            qs.CNOT(0, 0)
    
    def test_CCX(self):
        qs = QuantumState(3).H(range(3)).T([0, 1]).CCX(0, 1, 2).H([0, 1])
        expected_state = np.array([0.427 + 0.427j, 0.177 - 0.177j, 0.177 - 0.177j, -0.073-0.073j,
                                   0.427 + 0.427j, 0.177 - 0.177j, 0.177 - 0.177j, -0.073-0.073j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_CCX_invalid(self):
        qs = QuantumState(3)
        with pytest.raises(ValueError):
            qs.CCX(3, 1, 0)
        with pytest.raises(ValueError):
            qs.CCX(0, 2, 3)
        with pytest.raises(TypeError):
            qs.CCX([0, 1], 2, 0)
        with pytest.raises(TypeError):
            qs.CCX("invalid", 1, 2)
        with pytest.raises(ValueError):
            qs.CCX(0, -1, 2)
        with pytest.raises(ValueError):
            qs.CCX(0, 1, 0)
        
    def test_P_int(self):
        qs = QuantumState(2).H(0).P(0, np.pi / 4)
        expected_state = np.array([0.707, 0.5 + 0.5j, 0, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_P_list(self):
        qs = QuantumState(2).H([0, 1]).P([0, 1], np.pi / 4)
        expected_state = np.array([0.5, 0.354 + 0.354j, 0.354 + 0.354j, 0.5j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_P_range(self):
        qs = QuantumState(2).H(range(2)).P(range(2), np.pi / 4)
        expected_state = np.array([0.5, 0.354 + 0.354j, 0.354 + 0.354j, 0.5j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_P_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.P(2, np.pi / 4)
        with pytest.raises(ValueError):
            qs.P([0, 2], np.pi / 4)
        with pytest.raises(ValueError):
            qs.P(range(3), np.pi / 4)
        with pytest.raises(TypeError):
            qs.P("invalid", np.pi / 4)
        with pytest.raises(TypeError):
            qs.P(0, 'invalid')
    
    def test_RX_int(self):
        qs = QuantumState(2).H(0).RX(0, np.pi / 2)
        expected_state = 0.5 * np.array([1 - 1j, 1 - 1j, 0, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_RX_list(self):
        qs = QuantumState(2).H([0, 1]).RX([0, 1], np.pi / 2)
        expected_state = 0.5 * np.array([-1j, -1j, -1j, -1j])
        assert np.allclose(qs.state, expected_state)
    
    def test_RX_range(self):
        qs = QuantumState(2).H(range(2)).RX(range(2), np.pi / 2)
        expected_state = 0.5 * np.array([-1j, -1j, -1j, -1j])
        assert np.allclose(qs.state, expected_state)
    
    def test_RX_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.RX(2, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RX([0, 2], np.pi / 2)
        with pytest.raises(ValueError):
            qs.RX(range(3), np.pi / 2)
        with pytest.raises(TypeError):
            qs.RX("invalid", np.pi / 2)
        with pytest.raises(TypeError):
            qs.RX(0, 'invalid')
    
    def test_RY_int(self):
        qs = QuantumState(2).H(0).RY(0, np.pi / 3)
        expected_state = np.array([0.259, 0.966, 0, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_RY_list(self):
        qs = QuantumState(2).H(0).RY([0, 1], np.pi / 3)
        expected_state = np.array([0.224, 0.837, 0.129, 0.483])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_RY_range(self):
        qs = QuantumState(2).H(0).RY(range(2), np.pi / 3)
        expected_state = np.array([0.224, 0.837, 0.129, 0.483])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_RY_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.RY(2, np.pi / 3)
        with pytest.raises(ValueError):
            qs.RY([0, 2], np.pi / 3)
        with pytest.raises(ValueError):
            qs.RY(range(3), np.pi / 3)
        with pytest.raises(TypeError):
            qs.RY("invalid", np.pi / 3)
        with pytest.raises(TypeError):
            qs.RY(0, 'invalid')

    def test_RZ_int(self):
        qs = QuantumState(2).H(0).RZ(0, np.pi / 4)
        expected_state = np.array([0.653 - 0.271j, 0.653 + 0.271j, 0, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_RZ_list(self):
        qs = QuantumState(2).H(0).RZ([0, 1], np.pi / 4)
        expected_state = np.array([0.5 - 0.5j, 0.707, 0, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_RZ_range(self):
        qs = QuantumState(2).H(0).RZ(range(2), np.pi / 4)
        expected_state = np.array([0.5 - 0.5j, 0.707, 0, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)

    def test_RZ_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.RZ(2, np.pi / 4)
        with pytest.raises(ValueError):
            qs.RZ([0, 2], np.pi / 4)
        with pytest.raises(ValueError):
            qs.RZ(range(3), np.pi / 4)
        with pytest.raises(TypeError):
            qs.RZ("invalid", np.pi / 4)
        with pytest.raises(TypeError):
            qs.RZ(0, 'invalid')

    def test_RXX(self):
        qs = QuantumState(2).RXX(0, 1, np.pi / 2)
        expected_state = np.array([0.707, 0, 0, -0.707j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)

    def test_RYY(self):
        qs = QuantumState(2).RYY(0, 1, np.pi / 2)
        expected_state = np.array([0.707, 0, 0, 0.707j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)

    def test_RZZ(self):
        qs = QuantumState(2).RZZ(0, 1, np.pi / 2)
        expected_state = np.array([0.707 - 0.707j, 0, 0, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_RXX_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.RXX(2, 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RXX(0, 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RXX([0, 1], 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RXX("invalid", 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RXX(0, -1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RXX(0, 0, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RXX(0, 1, 'invalid')
    
    def test_RYY_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.RYY(2, 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RYY(0, 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RYY([0, 1], 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RYY("invalid", 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RYY(0, -1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RYY(0, 0, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RYY(0, 1, 'invalid')
    
    def test_RZZ_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.RZZ(2, 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RZZ(0, 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RZZ([0, 1], 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RZZ("invalid", 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RZZ(0, -1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.RZZ(0, 0, np.pi / 2)
        with pytest.raises(TypeError):
            qs.RZZ(0, 1, 'invalid')
    
    def test_SWAP(self):
        qs = QuantumState(2).X(0).SWAP(0, 1)
        expected_state = np.array([0, 0, 1, 0])
        assert np.allclose(qs.state, expected_state)
    
    def test_SWAP_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.SWAP(2, 1)
        with pytest.raises(ValueError):
            qs.SWAP(0, 2)
        with pytest.raises(TypeError):
            qs.SWAP([0, 1], 2)
        with pytest.raises(TypeError):
            qs.SWAP("invalid", 1)
        with pytest.raises(ValueError):
            qs.SWAP(0, -1)
        with pytest.raises(ValueError):
            qs.SWAP(0, 0)
    
    def test_CZ(self):
        qs = QuantumState(2).H([0, 1]).CZ(0, 1)
        expected_state = 0.5 * np.array([1, 1, 1, -1])
        assert np.allclose(qs.state, expected_state)
    
    def test_CZ_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.CZ(2, 1)
        with pytest.raises(ValueError):
            qs.CZ(0, 2)
        with pytest.raises(TypeError):
            qs.CZ([0, 1], 2)
        with pytest.raises(TypeError):
            qs.CZ("invalid", 1)
        with pytest.raises(ValueError):
            qs.CZ(0, -1)
        with pytest.raises(ValueError):
            qs.CZ(0, 0)
        
    def test_CY(self):
        qs = QuantumState(2).H([0, 1]).CY(0, 1)
        expected_state = 0.5 * np.array([1, -1j, 1, 1j])
        assert np.allclose(qs.state, expected_state)

    def test_CY_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.CY(2, 1)
        with pytest.raises(ValueError):
            qs.CY(0, 2)
        with pytest.raises(TypeError):
            qs.CY([0, 1], 2)
        with pytest.raises(TypeError):
            qs.CY("invalid", 1)
        with pytest.raises(ValueError):
            qs.CY(0, -1)
        with pytest.raises(ValueError):
            qs.CY(0, 0)
    
    def test_CH(self):
        qs = QuantumState(2).H([0, 1]).CH(0, 1)
        expected_state = np.array([0.5, 0.707, 0.5, 0])
        assert np.allclose(qs.state, expected_state, atol=1e-3)

    def test_CH_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.CH(2, 1)
        with pytest.raises(ValueError):
            qs.CH(0, 2)
        with pytest.raises(TypeError):
            qs.CH([0, 1], 2)
        with pytest.raises(TypeError):
            qs.CH("invalid", 1)
        with pytest.raises(ValueError):
            qs.CH(0, -1)
        with pytest.raises(ValueError):
            qs.CH(0, 0)
    
    def test_CRX(self):
        qs = QuantumState(2).H([0, 1]).CRX(0, 1, np.pi / 2)
        expected_state = np.array([0.5, 0.354 - 0.354j, 0.5, 0.354 - 0.354j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_CRX_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.CRX(2, 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRX(0, 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRX([0, 1], 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRX("invalid", 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRX(0, -1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRX(0, 0, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRX(0, 1, 'invalid')
    
    def test_CRY(self):
        qs = QuantumState(2).H([0, 1]).CRY(0, 1, np.pi / 2)
        expected_state = np.array([0.5, 0, 0.5, 0.707])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_CRY_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.CRY(2, 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRY(0, 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRY([0, 1], 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRY("invalid", 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRY(0, -1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRY(0, 0, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRY(0, 1, 'invalid')

    def test_CRZ(self):
        qs = QuantumState(2).H([0, 1]).CRZ(0, 1, np.pi / 2)
        expected_state = np.array([0.5, 0.354 - 0.354j, 0.5, 0.354 + 0.354j])
        assert np.allclose(qs.state, expected_state, atol=1e-3)
    
    def test_CRZ_invalid(self):
        qs = QuantumState(2)
        with pytest.raises(ValueError):
            qs.CRZ(2, 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRZ(0, 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRZ([0, 1], 2, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRZ("invalid", 1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRZ(0, -1, np.pi / 2)
        with pytest.raises(ValueError):
            qs.CRZ(0, 0, np.pi / 2)
        with pytest.raises(TypeError):
            qs.CRZ(0, 1, 'invalid')
    
    def test_RCCX(self):
        qs = QuantumState(3).H([0, 1]).RCCX(0, 1, 2)
        expected_state = np.array([0.5, 0.5, 0.5, 0,
                                   0, 0, 0, 0.5j])
        assert np.allclose(qs.state, expected_state)
    
    def test_RCCX_invalid(self):
        qs = QuantumState(3)
        with pytest.raises(ValueError):
            qs.RCCX(3, 1, 0)
        with pytest.raises(ValueError):
            qs.RCCX(0, 2, 3)
        with pytest.raises(TypeError):
            qs.RCCX([0, 1], 2, 0)
        with pytest.raises(TypeError):
            qs.RCCX("invalid", 1, 2)
        with pytest.raises(ValueError):
            qs.RCCX(0, -1, 2)
        with pytest.raises(ValueError):
            qs.RCCX(0, 1, 0)