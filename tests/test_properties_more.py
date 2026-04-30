"""Property-based tests for classification, sensitivity, Monte Carlo, component swap."""

from hypothesis import assume, given
from hypothesis import strategies as st

import classification
import component_swap
import monte_carlo
import reliability_math as rm
import sensitivity_analysis

# Shared strategies
lambdas = st.floats(min_value=1e-12, max_value=1e-6, allow_nan=False, allow_infinity=False)
reliabilities = st.floats(min_value=0.5, max_value=0.999999, allow_nan=False, allow_infinity=False)
component_names = st.text(min_size=1, max_size=10, alphabet="RUCLJTP_0123456789")
param_keys = st.sampled_from(["t_ambient", "t_junction", "v_applied", "v_rated", "stress", "drive", "temp", "load"])
param_values = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)


def _exp_reliability(lam, hours):
    import math
    return math.exp(-lam * hours)


class PropertyClassificationTests:
    @given(component_names)
    def test_classification_does_not_crash_on_any_name(self, ref):
        result = classification.classify_component_info(ref, "", {})
        assert isinstance(result.component_type, str)
        assert isinstance(result.confidence, str)
        assert isinstance(result.review_required, bool)
        assert len(result.component_type) > 0

    def test_same_input_gives_same_result(self):
        r1 = classification.classify_component_info("R1", "10k", {"Footprint": "Resistor"})
        r2 = classification.classify_component_info("R1", "10k", {"Footprint": "Resistor"})
        assert r1 == r2

    @given(component_names.filter(lambda n: not n[0].isalpha() or n[0].isalpha()))
    def test_explicit_field_overrides_heuristic(self, ref):
        assume(len(ref) > 0)
        result = classification.classify_component_info(
            ref, "", {"Reliability_Class": "Resistor"},
        )
        assert result.component_type == "Resistor"
        assert result.source == "explicit field"

    def test_empty_ref_returns_misc(self):
        result = classification.classify_component_info("", "", {})
        assert result.component_type == "Miscellaneous"

    def test_known_u_prefix_is_ic(self):
        result = classification.classify_component_info("U1", "", {})
        assert result.component_type == "Integrated Circuit"

    def test_r_prefix_is_resistor(self):
        result = classification.classify_component_info("R1", "", {})
        assert result.component_type == "Resistor"


class PropertySensitivityAnalysisTests:
    @given(
        st.lists(param_keys, min_size=1, max_size=5),
        st.lists(param_values, min_size=1, max_size=5),
    )
    def test_tornado_entries_have_positive_swing(self, keys, vals):
        assume(len(keys) == len(vals))
        params = dict(zip(keys, vals))
        sheet_data = {
            "/main": {
                "lambda": 1e-9,
                "components": [
                    {"ref": "DUT", "class": "Resistor", "lambda": 1e-9, "params": params},
                ],
            }
        }

        def fake_calc(_ctype, p):
            return {"lambda_total": sum(float(v) for v in p.values()) * 1e-9}

        from unittest import mock
        with mock.patch.object(
            sensitivity_analysis,
            "_import_math",
            return_value=(fake_calc, _exp_reliability),
        ):
            result = sensitivity_analysis.tornado_analysis(
                sheet_data, mission_hours=1000.0,
            )
        for entry in result.entries:
            assert entry.swing >= 0

    def test_tornado_empty_sheets_returns_empty(self):
        from unittest import mock
        with mock.patch.object(
            sensitivity_analysis,
            "_import_math",
            return_value=(lambda _c, _p: {"lambda_total": 0}, _exp_reliability),
        ):
            result = sensitivity_analysis.tornado_analysis({}, mission_hours=1000.0)
        assert len(result.entries) == 0


class PropertyComponentSwapTests:
    def test_swap_same_params_gives_zero_delta(self):
        from unittest import mock
        params = {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0}
        with mock.patch.object(
            rm,
            "calculate_component_lambda",
            return_value={"lambda_total": 1e-9},
        ):
            result = component_swap.quick_swap_comparison(
                component_type="Resistor",
                base_params=params,
                parameter="v_applied",
                new_value=5.0,
            )
        assert result["delta_fit"] == 0.0

    def test_swap_reducing_stress_reduces_fit(self):
        from unittest import mock

        def fake_lambda(_ct, params):
            stress = float(params.get("v_applied", 1.0))
            return {"lambda_total": stress * 1e-9}

        with mock.patch.object(
            rm,
            "calculate_component_lambda",
            side_effect=fake_lambda,
        ):
            result = component_swap.quick_swap_comparison(
                component_type="Resistor",
                base_params={"t_ambient": 25.0, "v_applied": 10.0, "v_rated": 10.0},
                parameter="v_applied",
                new_value=5.0,
            )
        assert result["delta_fit"] < 0
        assert result["improvement"] is True


class PropertyMonteCarloTests:
    def test_empty_components_raises_error(self):
        import pytest
        from unittest import mock
        with pytest.raises(ValueError):
            with mock.patch.object(
                monte_carlo,
                "_import_reliability_math",
                return_value=(lambda _c, _p: {"lambda_total": 0}, _exp_reliability),
            ):
                monte_carlo.run_uncertainty_analysis(
                    [], [], mission_hours=10.0, n_simulations=10,
                    confidence_level=0.90, seed=1,
                )
