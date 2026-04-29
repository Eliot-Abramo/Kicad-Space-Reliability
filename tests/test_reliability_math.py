import math

import pytest
import reliability_math as rm


class ValidationTests:
    def test_validate_ratio_clamps_to_range(self):
        assert rm.validate_ratio(0.5) == 0.5
        assert rm.validate_ratio(-0.1) == 0.0
        assert rm.validate_ratio(1.5) == 1.0
        assert rm.validate_ratio("0.3") == 0.3

    def test_validate_ratio_raises_on_non_numeric(self):
        with pytest.raises(TypeError):
            rm.validate_ratio("not-a-number")

    def test_validate_positive_clamps_negative(self):
        assert rm.validate_positive(10.0) == 10.0
        assert rm.validate_positive(-5.0) == 0.0

    def test_validate_temperature_clamps_below_absolute_zero(self):
        assert rm.validate_temperature(rm.DEFAULT_T_AMBIENT_C) == rm.DEFAULT_T_AMBIENT_C
        assert rm.validate_temperature(-300.0) == rm.ABSOLUTE_ZERO_C

    def test_validate_temperature_fallback_on_bad_input(self):
        assert rm.validate_temperature(None) == rm.DEFAULT_T_AMBIENT_C
        assert rm.validate_temperature("bad") == rm.DEFAULT_T_AMBIENT_C


class LambdaConversionTests:
    def test_fit_to_lambda_uses_named_constant(self):
        assert rm.fit_to_lambda(1000) == pytest.approx(1000 * rm.LAMBDA_PER_FIT)
        assert rm.fit_to_lambda(1) == pytest.approx(rm.LAMBDA_PER_FIT)

    def test_lambda_to_fit_uses_named_constant(self):
        assert rm.lambda_to_fit(1e-6) == int(1e-6 * rm.FIT_PER_LAMBDA)
        assert rm.lambda_to_fit(0.0) == 0.0

    def test_roundtrip_fit_lambda(self):
        for fit in [0, 1, 100, 1000, 1_000_000]:
            assert rm.lambda_to_fit(rm.fit_to_lambda(fit)) == pytest.approx(fit)

    def test_format_lambda_shows_fit(self):
        assert rm.format_lambda(1e-9) == "1.00 FIT"
        assert rm.format_lambda(1e-9, as_fit=False) == "1.00e-09 /h"

    def test_format_reliability(self):
        assert rm.format_reliability(0.999) == "0.9990"
        assert rm.format_reliability(0.5) == "0.500"


class ReliabilityMathTests:
    def test_reliability_from_lambda(self):
        lam = 1e-6
        hours = 1000
        expected = math.exp(-lam * hours)
        assert rm.reliability_from_lambda(lam, hours) == pytest.approx(expected)

    def test_reliability_from_lambda_zero_hours(self):
        assert rm.reliability_from_lambda(1e-6, 0) == 1.0

    def test_reliability_from_lambda_zero_lambda(self):
        assert rm.reliability_from_lambda(0.0, 1000) == 1.0

    def test_lambda_from_reliability(self):
        r = 0.9
        hours = 1000
        expected = -math.log(r) / hours
        assert rm.lambda_from_reliability(r, hours) == pytest.approx(expected)

    def test_mttf_from_lambda(self):
        assert rm.mttf_from_lambda(1e-6) == pytest.approx(1e6)
        assert rm.mttf_from_lambda(2e-6) == pytest.approx(5e5)


class SystemTopologyTests:
    def test_r_series_single(self):
        r = rm.r_series([0.99])
        assert r == pytest.approx(0.99)

    def test_r_series_multiple(self):
        r = rm.r_series([0.99, 0.98, 0.97])
        assert r == pytest.approx(0.99 * 0.98 * 0.97)

    def test_r_parallel_single(self):
        r = rm.r_parallel([0.99])
        assert r == pytest.approx(0.99)

    def test_r_parallel_dual(self):
        r = rm.r_parallel([0.9, 0.9])
        assert r == pytest.approx(1 - (1 - 0.9) ** 2)

    def test_r_parallel_triple(self):
        r = rm.r_parallel([0.9, 0.8, 0.7])
        expected = 1 - (1 - 0.9) * (1 - 0.8) * (1 - 0.7)
        assert r == pytest.approx(expected)

    def test_r_k_of_n_correct_signature(self):
        r = rm.r_k_of_n([0.99, 0.99], k=2)
        assert r == pytest.approx(0.99**2)

    def test_r_k_of_n_redundant(self):
        r = rm.r_k_of_n([0.9, 0.9, 0.9], k=2)
        p_fail = 0.1
        expected = (0.9**3) + 3 * (0.9**2) * p_fail
        assert r == pytest.approx(expected)

    def test_lambda_series(self):
        lam = rm.lambda_series([1e-6, 2e-6, 3e-6])
        assert lam == pytest.approx(6e-6)

    def test_lambda_series_single(self):
        lam = rm.lambda_series([5e-7])
        assert lam == pytest.approx(5e-7)

    def test_connection_type_default(self):
        ct = rm.ConnectionType.SERIES
        assert ct.value == "series"

    def test_connection_type_explicit(self):
        ct = rm.ConnectionType("parallel")
        assert ct.value == "parallel"

    def test_connection_type_equals_string(self):
        ct = rm.ConnectionType("parallel")
        assert ct == "parallel"
        assert ct == rm.ConnectionType("parallel")
        assert ct != "series"

    def test_connection_type_equality_with_other(self):
        ct1 = rm.ConnectionType("k_of_n")
        ct2 = rm.ConnectionType("k_of_n")
        ct3 = rm.ConnectionType("series")
        assert ct1 == ct2
        assert ct1 != ct3

    def test_connection_type_hash(self):
        ct = rm.ConnectionType("parallel")
        assert hash(ct) == hash("parallel")

    def test_connection_type_str(self):
        assert str(rm.ConnectionType("parallel")) == "parallel"


class ComponentTypeTests:
    def test_get_component_types_returns_expected_families(self):
        types = rm.get_component_types()
        assert "Integrated Circuit" in types
        assert "Resistor" in types
        assert "Capacitor" in types
        assert "Diode" in types
        assert "Connector" in types

    def test_get_field_definitions_requires_component_type(self):
        fields = rm.get_field_definitions("Resistor")
        assert len(fields) > 0
        assert "t_ambient" in fields

    def test_get_field_definitions_ic_has_specific_fields(self):
        fields = rm.get_field_definitions("Integrated Circuit")
        assert "t_junction" in fields
        assert "package" in fields
        assert "ic_type" in fields


class LambdaFunctionDispatchTests:
    def test_calculate_lambda_returns_float(self):
        result = rm.calculate_lambda("Resistor", {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0})
        assert isinstance(result, float)
        assert result > 0

    def test_calculate_lambda_integrated_circuit(self):
        result = rm.calculate_lambda(
            "Integrated Circuit",
            {
                "t_junction": 50.0,
                "n_cycles": 365,
                "delta_t": 10.0,
                "t_ambient": 25.0,
                "a": 0.5,
                "package": "DIP",
                "n_pins": 16,
            },
        )
        assert isinstance(result, float)
        assert result > 0

    def test_calculate_lambda_capacitor(self):
        result = rm.calculate_lambda(
            "Capacitor",
            {
                "t_ambient": 40.0,
                "v_applied": 10.0,
                "v_rated": 25.0,
                "type": "ceramic",
            },
        )
        assert isinstance(result, float)
        assert result > 0

    def test_calculate_lambda_connector(self):
        result = rm.calculate_lambda(
            "Connector",
            {
                "t_ambient": 35.0,
                "n_pins": 10,
                "n_cycles": 100,
            },
        )
        assert isinstance(result, float)
        assert result > 0

    def test_calculate_component_lambda_returns_dict(self):
        result = rm.calculate_component_lambda("Resistor", {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0})
        assert "lambda_total" in result
        assert result["lambda_total"] > 0

    def test_lambda_is_positive_for_all_component_types(self):
        generic_params = {"t_ambient": 25.0, "t_junction": 50.0}
        for ctype in rm.get_component_types():
            result = rm.calculate_lambda(ctype, generic_params)
            assert isinstance(result, float)
            assert result >= 0


class CriticalityTests:
    def test_analyze_component_criticality_returns_rows(self):
        rows = rm.analyze_component_criticality(
            "Resistor",
            {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0},
            mission_hours=1000.0,
            perturbation=0.1,
        )
        assert len(rows) > 0
        for row in rows:
            assert "field" in row
            assert "sensitivity" in row
            assert "impact_percent" in row

    def test_criticality_temperature_elasticity_sign(self):
        rows = rm.analyze_component_criticality(
            "Integrated Circuit",
            {
                "t_junction": 50.0,
                "n_cycles": 365,
                "delta_t": 10.0,
                "t_ambient": 25.0,
                "a": 0.5,
                "package": "DIP",
                "n_pins": 16,
            },
            mission_hours=1000.0,
            perturbation=0.1,
        )
        for row in rows:
            if "temperature" in row["field"].lower() or "junction" in row["field"].lower():
                assert row["sensitivity"] > 0, f"Expected positive sensitivity for {row['field']}"
