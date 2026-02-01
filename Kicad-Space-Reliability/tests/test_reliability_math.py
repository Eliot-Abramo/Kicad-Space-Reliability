"""
Test Suite for Reliability Math Module

Tests IEC TR 62380 implementation including:
- Pi factor calculations
- Component failure rates
- System reliability calculations
- Input validation

Run with: pytest tests/test_reliability_math.py -v
"""

import pytest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reliability_math import (
    # Pi factors
    pi_thermal_cycles,
    pi_temperature,
    pi_alpha,
    lambda_eos,
    # Component calculations
    lambda_integrated_circuit,
    lambda_diode,
    lambda_transistor,
    lambda_capacitor,
    lambda_resistor,
    lambda_inductor,
    lambda_misc_component,
    calculate_component_lambda,
    # System reliability
    reliability_from_lambda,
    lambda_from_reliability,
    mttf_from_lambda,
    r_series,
    r_parallel,
    r_k_of_n,
    lambda_series,
    # Validation
    validate_temperature,
    validate_positive,
    validate_ratio,
    # Lookup tables
    INTERFACE_EOS_VALUES,
    THERMAL_EXPANSION_SUBSTRATE,
    IC_TYPE_CHOICES,
    ActivationEnergy,
)


class TestPiThermalCycles:
    """Test thermal cycling factor calculations."""

    def test_below_threshold(self):
        """π_n = n^0.76 for n ≤ 8760"""
        n = 5256
        expected = n**0.76
        result = pi_thermal_cycles(n)
        assert abs(result - expected) < 1e-6

    def test_at_threshold(self):
        """π_n = n^0.76 at exactly n = 8760"""
        n = 8760
        expected = n**0.76
        result = pi_thermal_cycles(n)
        assert abs(result - expected) < 1e-6

    def test_above_threshold(self):
        """π_n = 1.7 × n^0.6 for n > 8760"""
        n = 10000
        expected = 1.7 * (n**0.6)
        result = pi_thermal_cycles(n)
        assert abs(result - expected) < 1e-6

    def test_continuity_at_threshold(self):
        """Check reasonable continuity at threshold boundary."""
        below = pi_thermal_cycles(8760)
        above = pi_thermal_cycles(8761)
        # Should be reasonably close (within 10%)
        ratio = above / below
        assert 0.9 < ratio < 1.1

    def test_zero_cycles(self):
        """Handle zero cycles gracefully."""
        result = pi_thermal_cycles(0)
        assert result == 0.0

    def test_negative_cycles(self):
        """Handle negative cycles (return 0)."""
        result = pi_thermal_cycles(-100)
        assert result == 0.0


class TestPiTemperature:
    """Test temperature acceleration factor."""

    def test_reference_temperature(self):
        """At reference temperature, π_T should be close to 1."""
        # For T_ref = 328K (55°C), T_j = 55°C
        result = pi_temperature(55.0, ActivationEnergy.MOS, 328.0)
        assert 0.99 < result < 1.01

    def test_higher_temperature(self):
        """Higher temperature should increase π_T."""
        t_low = pi_temperature(60.0, ActivationEnergy.MOS, 328.0)
        t_high = pi_temperature(100.0, ActivationEnergy.MOS, 328.0)
        assert t_high > t_low

    def test_arrhenius_relationship(self):
        """Verify Arrhenius equation implementation."""
        t_j = 85.0
        ea = ActivationEnergy.MOS  # 3480
        t_ref = 328.0

        expected = math.exp(ea * ((1 / t_ref) - (1 / (273 + t_j))))
        result = pi_temperature(t_j, ea, t_ref)
        assert abs(result - expected) < 1e-10


class TestPiAlpha:
    """Test thermal expansion mismatch factor."""

    def test_same_cte(self):
        """No mismatch when CTEs are equal."""
        result = pi_alpha(16.0, 16.0)
        assert result == 0.0

    def test_fr4_epoxy_mismatch(self):
        """FR4 (16 ppm/°C) vs epoxy package (21.5 ppm/°C)."""
        result = pi_alpha(16.0, 21.5)
        expected = 0.06 * (abs(16.0 - 21.5) ** 1.68)
        assert abs(result - expected) < 1e-10

    def test_symmetry(self):
        """π_α should be same regardless of order."""
        result1 = pi_alpha(16.0, 21.5)
        result2 = pi_alpha(21.5, 16.0)
        assert abs(result1 - result2) < 1e-10


class TestLambdaEOS:
    """Test electrical overstress calculations."""

    def test_not_interface(self):
        """Non-interface components have zero EOS."""
        result = lambda_eos("Not Interface", is_interface=False)
        assert result == 0.0

    def test_avionics_interface(self):
        """Avionics interface should return correct value."""
        result = lambda_eos("Avionics", is_interface=True)
        expected = (
            INTERFACE_EOS_VALUES["Avionics"]["pi_i"]
            * INTERFACE_EOS_VALUES["Avionics"]["l_eos"]
        )
        assert result == expected

    def test_all_interface_types(self):
        """All interface types should return valid values."""
        for itype in INTERFACE_EOS_VALUES.keys():
            result = lambda_eos(itype, is_interface=True)
            assert result >= 0

    def test_unknown_interface_type(self):
        """Unknown type should default to 0."""
        result = lambda_eos("Unknown Type", is_interface=True)
        assert result == 0.0


class TestLambdaIntegratedCircuit:
    """Test IC failure rate calculations."""

    def test_basic_calculation(self):
        """Basic IC calculation should return valid structure."""
        result = lambda_integrated_circuit(
            ic_type="MOS_DIGITAL", transistor_count=10000, t_junction=85.0
        )

        assert "lambda_die" in result
        assert "lambda_package" in result
        assert "lambda_eos" in result
        assert "lambda_total" in result
        assert "fit_total" in result

        # Total should be sum of contributions
        total = result["lambda_die"] + result["lambda_package"] + result["lambda_eos"]
        assert abs(result["lambda_total"] - total) < 1e-15

    def test_interface_increases_lambda(self):
        """Interface components should have higher λ."""
        result_no_if = lambda_integrated_circuit(
            ic_type="MOS_DIGITAL", transistor_count=10000, is_interface=False
        )

        result_with_if = lambda_integrated_circuit(
            ic_type="MOS_DIGITAL",
            transistor_count=10000,
            is_interface=True,
            interface_type="Avionics",
        )

        assert result_with_if["lambda_total"] > result_no_if["lambda_total"]

    def test_tau_on_reduces_lambda(self):
        """Lower duty cycle should reduce effective λ."""
        result_full = lambda_integrated_circuit(
            ic_type="MOS_DIGITAL", transistor_count=10000, tau_on=1.0
        )

        result_half = lambda_integrated_circuit(
            ic_type="MOS_DIGITAL", transistor_count=10000, tau_on=0.5
        )

        # Die and package contributions scale with tau_on
        assert result_half["lambda_die"] < result_full["lambda_die"]

    def test_transistor_count_effect(self):
        """More transistors should increase λ_die."""
        result_small = lambda_integrated_circuit(
            ic_type="MOS_DIGITAL", transistor_count=1000
        )

        result_large = lambda_integrated_circuit(
            ic_type="MOS_DIGITAL", transistor_count=1000000
        )

        assert result_large["lambda_die"] > result_small["lambda_die"]


class TestLambdaCapacitor:
    """Test capacitor failure rate calculations."""

    def test_ceramic_capacitor(self):
        """Ceramic capacitor calculation."""
        result = lambda_capacitor(
            capacitor_type="Ceramic Class II (X7R/X5R)",
            t_ambient=25.0,
            n_cycles=5256,
            delta_t=3.0,
        )

        assert result["lambda_total"] > 0
        assert result["fit_total"] > 0

    def test_temperature_effect(self):
        """Higher temperature should increase λ."""
        result_cold = lambda_capacitor(t_ambient=25.0)
        result_hot = lambda_capacitor(t_ambient=85.0)

        assert result_hot["lambda_total"] > result_cold["lambda_total"]

    def test_tau_on_effect(self):
        """tau_on should scale λ."""
        result_full = lambda_capacitor(tau_on=1.0)
        result_half = lambda_capacitor(tau_on=0.5)

        assert result_half["lambda_base"] < result_full["lambda_base"]


class TestLambdaResistor:
    """Test resistor failure rate calculations."""

    def test_smd_resistor(self):
        """SMD chip resistor calculation."""
        result = lambda_resistor(
            resistor_type="SMD Chip Resistor",
            t_ambient=25.0,
            operating_power=0.01,
            rated_power=0.125,
        )

        assert result["lambda_total"] > 0
        assert "t_resistor" in result  # Should calculate internal temperature

    def test_power_derating(self):
        """Higher power ratio should increase λ."""
        result_low = lambda_resistor(operating_power=0.01, rated_power=0.125)
        result_high = lambda_resistor(operating_power=0.1, rated_power=0.125)

        assert result_high["lambda_total"] > result_low["lambda_total"]


class TestSystemReliability:
    """Test system-level reliability calculations."""

    def test_reliability_from_lambda(self):
        """R(t) = exp(-λt)"""
        lam = 1e-6
        t = 1000
        expected = math.exp(-lam * t)
        result = reliability_from_lambda(lam, t)
        assert abs(result - expected) < 1e-10

    def test_lambda_from_reliability(self):
        """λ = -ln(R)/t"""
        r = 0.99
        t = 1000
        expected = -math.log(r) / t
        result = lambda_from_reliability(r, t)
        assert abs(result - expected) < 1e-10

    def test_roundtrip_conversion(self):
        """λ → R → λ should give same value."""
        lam_original = 1e-7
        t = 43800  # 5 years

        r = reliability_from_lambda(lam_original, t)
        lam_recovered = lambda_from_reliability(r, t)

        assert abs(lam_recovered - lam_original) < 1e-15

    def test_mttf(self):
        """MTTF = 1/λ"""
        lam = 1e-6
        expected = 1e6
        result = mttf_from_lambda(lam)
        assert abs(result - expected) < 1e-6

    def test_series_reliability(self):
        """R_series = R1 × R2 × ... × Rn"""
        r_list = [0.99, 0.98, 0.97]
        expected = 0.99 * 0.98 * 0.97
        result = r_series(r_list)
        assert abs(result - expected) < 1e-10

    def test_parallel_reliability(self):
        """R_parallel = 1 - (1-R1)(1-R2)...(1-Rn)"""
        r_list = [0.9, 0.9]
        expected = 1 - (0.1 * 0.1)  # 0.99
        result = r_parallel(r_list)
        assert abs(result - expected) < 1e-10

    def test_k_of_n_equals_series_when_k_equals_n(self):
        """K-of-N with k=n is equivalent to series."""
        r_list = [0.95, 0.95, 0.95]
        series_r = r_series(r_list)
        kn_r = r_k_of_n(r_list, k=3)
        assert abs(series_r - kn_r) < 1e-10

    def test_k_of_n_equals_parallel_when_k_equals_1(self):
        """K-of-N with k=1 is equivalent to parallel."""
        r_list = [0.9, 0.9, 0.9]
        parallel_r = r_parallel(r_list)
        kn_r = r_k_of_n(r_list, k=1)
        assert abs(parallel_r - kn_r) < 1e-10

    def test_2_of_3_redundancy(self):
        """2-of-3 system calculation."""
        r = 0.9
        r_list = [r, r, r]
        # P(at least 2 work) = P(all 3) + P(exactly 2)
        # = r³ + 3×r²×(1-r)
        expected = r**3 + 3 * r**2 * (1 - r)
        result = r_k_of_n(r_list, k=2)
        assert abs(result - expected) < 1e-10

    def test_lambda_series(self):
        """λ_series = λ1 + λ2 + ... + λn"""
        lam_list = [1e-7, 2e-7, 3e-7]
        expected = 6e-7
        result = lambda_series(lam_list)
        assert abs(result - expected) < 1e-15


class TestValidation:
    """Test input validation functions."""

    def test_validate_temperature_valid(self):
        """Valid temperatures should pass."""
        assert validate_temperature(25.0) == 25.0
        assert validate_temperature(-40.0) == -40.0
        assert validate_temperature(150.0) == 150.0

    def test_validate_temperature_invalid(self):
        """Invalid temperatures should raise or clamp."""
        with pytest.raises((ValueError, TypeError)):
            validate_temperature(-274)  # Below absolute zero

    def test_validate_positive(self):
        """Positive validation."""
        assert validate_positive(1.0) == 1.0
        assert validate_positive(0.001) == 0.001

        with pytest.raises(ValueError):
            validate_positive(-1.0)

    def test_validate_ratio(self):
        """Ratio validation (0-1)."""
        assert validate_ratio(0.5) == 0.5
        assert validate_ratio(0.0) == 0.0
        assert validate_ratio(1.0) == 1.0

        with pytest.raises(ValueError):
            validate_ratio(1.5)

        with pytest.raises(ValueError):
            validate_ratio(-0.1)


class TestCalculateComponentLambda:
    """Test the universal component lambda dispatcher."""

    def test_ic_dispatch(self):
        """IC calculation via dispatcher."""
        result = calculate_component_lambda(
            "Integrated Circuit",
            {
                "ic_type": "Microcontroller/DSP",
                "transistor_count": 100000,
                "t_junction": 85.0,
                "package": "QFP-64 (10x10mm)",
            },
        )

        assert "lambda_total" in result
        assert result["lambda_total"] > 0

    def test_resistor_dispatch(self):
        """Resistor calculation via dispatcher."""
        result = calculate_component_lambda(
            "Resistor",
            {
                "resistor_type": "SMD Chip Resistor",
                "t_ambient": 25.0,
                "operating_power": 0.01,
                "rated_power": 0.125,
            },
        )

        assert "lambda_total" in result
        assert result["lambda_total"] > 0

    def test_capacitor_dispatch(self):
        """Capacitor calculation via dispatcher."""
        result = calculate_component_lambda(
            "Capacitor",
            {"capacitor_type": "Ceramic Class II (X7R/X5R)", "t_ambient": 25.0},
        )

        assert "lambda_total" in result
        assert result["lambda_total"] > 0


class TestLookupTables:
    """Test that lookup tables are properly defined."""

    def test_interface_eos_values(self):
        """All interface types should have required fields."""
        required_fields = ["pi_i", "l_eos"]

        for itype, values in INTERFACE_EOS_VALUES.items():
            for field in required_fields:
                assert field in values, f"Missing {field} in {itype}"
            assert values["pi_i"] >= 0
            assert values["l_eos"] >= 0

    def test_thermal_expansion_substrate(self):
        """All substrates should have positive CTE."""
        for substrate, cte in THERMAL_EXPANSION_SUBSTRATE.items():
            assert cte > 0, f"Invalid CTE for {substrate}"

    def test_ic_type_choices(self):
        """IC type choices should map to valid die types."""
        from reliability_math import IC_DIE_TABLE

        for choice, die_type in IC_TYPE_CHOICES.items():
            assert die_type in IC_DIE_TABLE, f"{choice} maps to invalid {die_type}"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
