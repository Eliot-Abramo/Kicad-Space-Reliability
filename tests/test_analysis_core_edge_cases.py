from unittest import mock

import budget_allocation
import classification
import monte_carlo
import numpy as np
import pytest
import sensitivity_analysis

from tests.helpers import (
    exp_reliability,
    make_component,
    make_component_input,
    make_parameter_spec,
    make_sheet_data,
)


class MonteCarloEdgeCaseTests:
    def test_empty_components_raises_value_error(self):
        with pytest.raises(ValueError), mock.patch.object(
            monte_carlo,
            "_import_reliability_math",
            return_value=(lambda _c, _p: {"lambda_total": 0}, exp_reliability),
        ):
            monte_carlo.run_uncertainty_analysis(
                [],
                [],
                mission_hours=10.0,
                n_simulations=10,
                confidence_level=0.90,
                seed=1,
            )

    def test_single_component_single_spec(self):
        components = [
            make_component_input("R1", "Resistor", {"t_ambient": 25.0}, lam=1e-9),
        ]
        specs = [
            make_parameter_spec("t_ambient", {"R1": 25.0}, delta_low=5.0, delta_high=5.0),
        ]

        def fake_calc(_ctype, params):
            return {"lambda_total": float(params["t_ambient"]) * 1e-9}

        def fake_sample(_rng, *_a, **_kw):
            return np.full(10, 25.0)

        with (
            mock.patch.object(
                monte_carlo,
                "_import_reliability_math",
                return_value=(fake_calc, exp_reliability),
            ),
            mock.patch.object(monte_carlo, "_sample_parameter", side_effect=fake_sample),
        ):
            result = monte_carlo.run_uncertainty_analysis(
                components,
                specs,
                mission_hours=100.0,
                n_simulations=10,
                confidence_level=0.90,
                seed=42,
            )

        assert len(result.lambda_samples) == 10
        assert len(result.reliability_samples) == 10
        for lam in result.lambda_samples:
            assert lam == pytest.approx(25e-9)
        assert "mean_reliability" in dir(result)

    def test_large_n_simulations_does_not_crash(self):
        components = [
            make_component_input("R1", "Resistor", {"t_ambient": 25.0}, lam=1e-9),
        ]
        specs = [
            make_parameter_spec("t_ambient", {"R1": 25.0}, delta_low=1.0, delta_high=1.0),
        ]

        def fake_calc(_ctype, params):
            return {"lambda_total": float(params["t_ambient"]) * 1e-9}

        def fake_sample(_rng, *_a, **_kw):
            return np.full(1000, 25.0)

        with (
            mock.patch.object(
                monte_carlo,
                "_import_reliability_math",
                return_value=(fake_calc, exp_reliability),
            ),
            mock.patch.object(monte_carlo, "_sample_parameter", side_effect=fake_sample),
        ):
            result = monte_carlo.run_uncertainty_analysis(
                components,
                specs,
                mission_hours=10.0,
                n_simulations=1000,
                confidence_level=0.90,
                seed=1,
            )
        assert result.n_simulations == 1000

    def test_short_mission_returns_high_reliability(self):
        components = [
            make_component_input("R1", "Resistor", {"t_ambient": 25.0}, lam=1e-9),
        ]
        specs = [
            make_parameter_spec("t_ambient", {"R1": 25.0}, delta_low=5.0, delta_high=5.0),
        ]

        def fake_calc(_ctype, params):
            return {"lambda_total": float(params["t_ambient"]) * 1e-9}

        def fake_sample(_rng, *_a, **_kw):
            return np.array([25.0, 26.0, 24.0, 25.0, 25.0, 26.0, 24.0, 25.0, 25.0, 26.0])

        with (
            mock.patch.object(
                monte_carlo,
                "_import_reliability_math",
                return_value=(fake_calc, exp_reliability),
            ),
            mock.patch.object(monte_carlo, "_sample_parameter", side_effect=fake_sample),
        ):
            result = monte_carlo.run_uncertainty_analysis(
                components,
                specs,
                mission_hours=1e-9,
                n_simulations=10,
                confidence_level=0.90,
                seed=1,
            )
            assert abs(result.mean_reliability - 1.0) < 1e-10

    def test_mismatched_refs_in_specs_handles_gracefully(self):
        components = [
            make_component_input("R1", "Resistor", {"x": 1.0}, lam=1e-9),
        ]
        specs = [
            make_parameter_spec("y", {"R2": 100.0}, delta_low=5.0, delta_high=5.0),
        ]

        def fake_calc(_ctype, params):
            return {"lambda_total": float(params.get("x", 0)) * 1e-9}

        def fake_sample(_rng, *_a, **_kw):
            return np.zeros(10)

        with (
            mock.patch.object(
                monte_carlo,
                "_import_reliability_math",
                return_value=(fake_calc, exp_reliability),
            ),
            mock.patch.object(monte_carlo, "_sample_parameter", side_effect=fake_sample),
        ):
            result = monte_carlo.run_uncertainty_analysis(
                components,
                specs,
                mission_hours=10.0,
                n_simulations=10,
                confidence_level=0.90,
                seed=1,
            )
        assert result.n_simulations == 10


class TornadoEdgeCaseTests:
    def test_single_component_tornado(self):
        def fake_calc(_ctype, params):
            return {"lambda_total": float(params.get("stress", 0)) * 1e-9}

        data = make_sheet_data(
            {
                "/main": {
                    "components": [make_component("R1", "Resistor", 3e-9, {"stress": 3.0})],
                },
            }
        )
        perturbations = [
            sensitivity_analysis.TornadoPerturbation("stress", 1.0, 1.0, "u"),
        ]

        with mock.patch.object(
            sensitivity_analysis,
            "_import_math",
            return_value=(fake_calc, exp_reliability),
        ):
            result = sensitivity_analysis.tornado_analysis(data, mission_hours=1000.0, perturbations=perturbations)
        assert len(result.entries) == 1

    def test_tornado_empty_sheets_returns_empty_entries(self):
        def fake_calc(_ctype, params):  # noqa: ARG001
            return {"lambda_total": 1e-9}

        with mock.patch.object(
            sensitivity_analysis,
            "_import_math",
            return_value=(fake_calc, exp_reliability),
        ):
            result = sensitivity_analysis.tornado_analysis({}, mission_hours=1000.0)
        assert len(result.entries) == 0


class BudgetEdgeCaseTests:
    def test_budget_allocation_with_no_components(self):
        result = budget_allocation.allocate_budget(
            {},
            mission_hours=1000.0,
            target_reliability=0.999,
            strategy="equal",
            margin_percent=10.0,
        )
        assert "actual_fit" in dir(result)

    def test_budget_allocation_perfect_reliability_target(self):
        data = make_sheet_data(
            {
                "/main": {
                    "components": [make_component("R1", "Resistor", 1e-12)],
                },
            }
        )
        result = budget_allocation.allocate_budget(
            data,
            mission_hours=1000.0,
            target_reliability=0.999999999,
            strategy="proportional",
            margin_percent=0.0,
        )
        assert result.fit_gap_to_close >= 0
        assert result.effective_budget_fit > 0


class ClassificationEdgeCaseTests:
    def test_classification_with_only_tp_testpoint_returns_misc(self):
        result = classification.classify_component_info("TP1", "", {})
        assert result.component_type == "Miscellaneous"
        assert result.review_required

    def test_classification_with_footprint_hint(self):
        result = classification.classify_component_info("J1", "", {"Footprint": "USB_A_Plug"})
        assert result.component_type == "Connector"

    def test_classification_with_lowercase_reliability_class(self):
        result = classification.classify_component_info("U1", "", {"Reliability_Class": "integrated circuit"})
        assert result.component_type == "Integrated Circuit"
