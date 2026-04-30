"""Integration tests that exercise the full analysis pipeline end-to-end."""

from datetime import datetime

import budget_allocation
import classification
import growth_tracking
import pytest
import reliability_math as rm


class ClassificationToLambdaIntegrationTests:
    def test_resistor_classify_then_calculate(self):
        result = classification.classify_component_info(
            "R1",
            "10k",
            {"Footprint": "Resistor_SMD", "Reliability_Class": "Resistor"},
        )
        assert result.component_type == "Resistor"
        lam = rm.calculate_lambda("Resistor", {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0})
        assert lam > 0
        # A resistor at 25C with 50% voltage stress should be well under 1 FIT
        assert lam < 1e-9

    def test_ic_classify_then_calculate(self):
        result = classification.classify_component_info(
            "U1",
            "STM32F4",
            {"Footprint": "LQFP-48", "Reliability_Class": "Integrated Circuit"},
        )
        assert result.component_type == "Integrated Circuit"
        lam = rm.calculate_lambda(
            "Integrated Circuit",
            {
                "t_junction": 50.0,
                "n_cycles": 365,
                "delta_t": 10.0,
                "t_ambient": 25.0,
                "a": 0.5,
                "package": "QFP-48 (7x7mm)",
                "ic_type": "MOS Digital (Micro/DSP)",
                "n_pins": 48,
            },
        )
        assert lam > 0
        # A 48-pin MOS digital IC at 50C junction temp should be under 100 FIT
        assert lam < 1e-7

    def test_different_component_types_differ_by_more_than_rounding(self):
        lam_resistor = rm.calculate_lambda("Resistor", {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0})
        lam_capacitor = rm.calculate_lambda("Capacitor", {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0})
        # Different IEC formulas should give meaningfully different results
        ratio = lam_resistor / lam_capacitor if lam_capacitor > 0 else 0
        assert ratio < 0.5


class LambdaToBudgetIntegrationTests:
    def test_sheet_data_flows_to_budget(self):
        sheet_data = {
            "/power": {
                "lambda": 50e-9,
                "components": [
                    {"ref": "U1", "class": "Integrated Circuit", "lambda": 30e-9, "params": {}},
                    {"ref": "C1", "class": "Capacitor", "lambda": 20e-9, "params": {}},
                ],
            },
        }
        result = budget_allocation.allocate_budget(
            sheet_data,
            mission_hours=1000.0,
            target_reliability=0.999,
            strategy="proportional",
            margin_percent=10.0,
        )
        assert result.actual_fit == pytest.approx(50.0)
        assert result.target_fit > 0
        assert len(result.sheet_budgets) == 1

    def test_budget_with_different_strategies_yield_different_allocations(self):
        sheet_data = {
            "/a": {"lambda": 30e-9, "components": [{"ref": "R1", "class": "Resistor", "lambda": 30e-9, "params": {}}]},
            "/b": {"lambda": 10e-9, "components": [{"ref": "R2", "class": "Resistor", "lambda": 10e-9, "params": {}}]},
        }
        result_equal = budget_allocation.allocate_budget(
            sheet_data,
            mission_hours=1000.0,
            target_reliability=0.9999,
            strategy="equal",
            margin_percent=5.0,
        )
        result_prop = budget_allocation.allocate_budget(
            sheet_data,
            mission_hours=1000.0,
            target_reliability=0.9999,
            strategy="proportional",
            margin_percent=5.0,
        )
        assert len(result_equal.sheet_budgets) == 2
        # Equal strategy gives same budget to both sheets
        budgets_equal = [s.budget_fit for s in result_equal.sheet_budgets]
        assert budgets_equal[0] == pytest.approx(budgets_equal[1])
        # Proportional strategy gives more budget to the higher-FIT sheet
        budgets_prop = [s.budget_fit for s in result_prop.sheet_budgets]
        assert budgets_prop[0] > budgets_prop[1]


class PipelineToSnapshotIntegrationTests:
    def test_sheet_data_to_snapshot(self):
        sheet_data = {
            "/main": {
                "lambda": 50e-9,
                "components": [
                    {"ref": "R1", "class": "Resistor", "lambda": 30e-9, "params": {}},
                    {"ref": "C1", "class": "Capacitor", "lambda": 20e-9, "params": {}},
                ],
            },
        }
        snap = growth_tracking.create_snapshot(
            sheet_data=sheet_data,
            system_lambda=50e-9,
            mission_hours=1000.0,
            version_label="integration-test",
            notes="Pipeline integration test",
        )
        assert snap.n_components == 2
        assert snap.n_sheets == 1
        assert snap.system_lambda == pytest.approx(50e-9)
        assert snap.system_fit == pytest.approx(50.0)
        datetime.fromisoformat(snap.timestamp)  # raises if invalid

    def test_snapshot_roundtrip_via_json(self):
        sheet_data = {
            "/main": {
                "lambda": 1e-6,
                "components": [{"ref": "R1", "class": "Resistor", "lambda": 1e-6, "params": {}}],
            },
        }
        snap = growth_tracking.create_snapshot(
            sheet_data=sheet_data,
            system_lambda=1e-6,
            mission_hours=8760.0,
            version_label="v1",
        )
        d = snap.to_dict()
        restored = growth_tracking.ReliabilitySnapshot.from_dict(d)
        for attr in ("version_label", "n_components", "system_lambda", "system_fit", "mission_hours"):
            assert getattr(snap, attr) == getattr(restored, attr)


class LambdaConsistencyTests:
    def test_reliability_is_monotonic_decreasing_with_lambda(self):
        hrs = 1000
        rs = [rm.reliability_from_lambda(10**p, hrs) for p in range(-8, -4)]
        for i in range(len(rs) - 1):
            assert rs[i] > rs[i + 1]

    def test_fit_conversion_via_budget_allocation(self):
        """FIT values computed by budget_allocation match reliability_math."""
        sheet_data = {
            "/main": {
                "lambda": 100e-9,
                "components": [{"ref": "R1", "class": "Resistor", "lambda": 100e-9, "params": {}}],
            },
        }
        result = budget_allocation.allocate_budget(
            sheet_data,
            mission_hours=1000.0,
            target_reliability=0.99,
            strategy="proportional",
            margin_percent=0.0,
        )
        expected_fit = rm.lambda_to_fit(100e-9)
        assert result.actual_fit == pytest.approx(expected_fit)
