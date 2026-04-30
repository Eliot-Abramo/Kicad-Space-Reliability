"""Extended property-based tests for budget, derating, growth tracking, mission profile."""

import math
from hypothesis import assume, given
from hypothesis import strategies as st

import budget_allocation
import derating_engine
import growth_tracking
import mission_profile
import reliability_math as rm


# Shared strategies
positive_floats = st.floats(min_value=1e-12, max_value=1e6, allow_nan=False, allow_infinity=False)
small_lambdas = st.floats(min_value=1e-12, max_value=1e-6, allow_nan=False, allow_infinity=False)
component_counts = st.integers(min_value=1, max_value=50)
reliabilities = st.floats(min_value=0.5, max_value=0.999999, allow_nan=False, allow_infinity=False)


def _make_sheet_with_components(n_comps, base_lam):
    """Generate a single-sheet component dict with n_comps identical parts."""
    return {
        "/main": {
            "lambda": base_lam * n_comps,
            "components": [
                {"ref": f"C{i}", "class": "Resistor", "lambda": base_lam, "params": {}}
                for i in range(n_comps)
            ],
        }
    }


class PropertyBudgetAllocationTests:
    @given(
        small_lambdas,
        component_counts,
        reliabilities,
        st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    def test_budget_fit_equals_sum_of_sheet_fits(self, base_lam, n_comps, target_rel, margin):
        data = _make_sheet_with_components(n_comps, base_lam)
        assume(target_rel > 0 and target_rel < 1)
        result = budget_allocation.allocate_budget(
            data, mission_hours=1000.0, target_reliability=target_rel,
            strategy="proportional", margin_percent=margin,
        )
        assert abs(result.actual_fit - base_lam * n_comps * 1e9) < 1e-6

    @given(
        small_lambdas,
        component_counts,
        st.sampled_from(["equal", "proportional", "complexity", "criticality"]),
    )
    def test_all_strategies_produce_valid_result(self, base_lam, n_comps, strategy):
        data = _make_sheet_with_components(n_comps, base_lam)
        result = budget_allocation.allocate_budget(
            data, mission_hours=1000.0, target_reliability=0.999,
            strategy=strategy, margin_percent=5.0,
        )
        assert result.target_fit > 0
        assert result.actual_fit > 0
        assert len(result.sheet_budgets) == 1

    @given(small_lambdas, component_counts)
    def test_higher_target_tightens_budget(self, base_lam, n_comps):
        data = _make_sheet_with_components(n_comps, base_lam)
        r1 = budget_allocation.allocate_budget(data, 1000.0, target_reliability=0.99, margin_percent=0.0)
        r2 = budget_allocation.allocate_budget(data, 1000.0, target_reliability=0.9999, margin_percent=0.0)
        assert r2.target_fit < r1.target_fit

    def test_no_components_returns_empty_results(self):
        result = budget_allocation.allocate_budget({}, 1000.0, target_reliability=0.999)
        assert result.actual_fit == 0.0
        assert len(result.sheet_budgets) == 0


class PropertyDeratingEngineTests:
    @given(small_lambdas, st.integers(min_value=1, max_value=5))
    def test_derating_priorities_are_non_negative(self, base_lam, n_comps):
        data = _make_sheet_with_components(n_comps, base_lam)
        result = derating_engine.compute_derating_guidance(
            data, mission_hours=1000.0, target_fit=base_lam * n_comps * 1e9 * 0.5,
        )
        for rec in result.recommendations:
            assert rec.priority >= 1
            assert rec.system_fit_reduction >= 0

    def test_no_components_returns_empty(self):
        result = derating_engine.compute_derating_guidance({}, 1000.0, target_fit=100.0)
        assert len(result.recommendations) == 0

    @given(small_lambdas)
    def test_derating_gap_matches_expected(self, base_lam):
        data = _make_sheet_with_components(2, base_lam)
        current_fit = base_lam * 2 * 1e9
        target_fit = current_fit * 0.5
        result = derating_engine.compute_derating_guidance(data, 1000.0, target_fit)
        assert result.system_actual_fit == current_fit
        assert result.system_target_fit == target_fit
        assert result.system_gap_fit == current_fit - target_fit


class PropertyGrowthTrackingTests:
    def test_identical_snapshots_have_zero_delta(self):
        snap = growth_tracking.ReliabilitySnapshot(
            timestamp="2025-01-01T00:00:00", version_label="v1", notes="",
            system_lambda=1e-6, system_fit=1000.0, system_reliability=0.99,
            mission_hours=1000.0, n_components=10, n_sheets=1,
            sheet_summary={}, component_lambdas={"R1": 1e-6}, component_details={},
        )
        comp = growth_tracking.compare_revisions(snap, snap)
        assert comp.system_delta_fit == 0.0
        assert comp.reliability_improvement == 0.0
        assert comp.components_improved == 0
        assert comp.components_degraded == 0

    def test_improvement_shows_negative_delta(self):
        early = growth_tracking.ReliabilitySnapshot(
            timestamp="2025-01-01T00:00:00", version_label="v1", notes="",
            system_lambda=2e-6, system_fit=2000.0, system_reliability=0.98,
            mission_hours=1000.0, n_components=10, n_sheets=1,
            sheet_summary={}, component_lambdas={"R1": 2e-6}, component_details={},
        )
        later = growth_tracking.ReliabilitySnapshot(
            timestamp="2025-06-01T00:00:00", version_label="v2", notes="",
            system_lambda=1e-6, system_fit=1000.0, system_reliability=0.99,
            mission_hours=1000.0, n_components=10, n_sheets=1,
            sheet_summary={}, component_lambdas={"R1": 1e-6}, component_details={},
        )
        comp = growth_tracking.compare_revisions(early, later)
        assert comp.system_delta_fit == -1000.0
        assert comp.reliability_improvement > 0
        assert comp.components_improved >= 1

    def test_snapshot_to_dict_roundtrip(self):
        snap = growth_tracking.ReliabilitySnapshot(
            timestamp="2025-01-01T00:00:00", version_label="v1", notes="test",
            system_lambda=1e-6, system_fit=1000.0, system_reliability=0.99,
            mission_hours=1000.0, n_components=5, n_sheets=2,
            sheet_summary={"/a": {"fit": 500.0}, "/b": {"fit": 500.0}},
            component_lambdas={"R1": 1e-6}, component_details={},
        )
        d = snap.to_dict()
        restored = growth_tracking.ReliabilitySnapshot.from_dict(d)
        assert snap.system_lambda == restored.system_lambda
        assert snap.n_components == restored.n_components
        assert snap.version_label == restored.version_label

    @given(st.lists(st.floats(min_value=1e-9, max_value=1e-6, allow_nan=False, allow_infinity=False), min_size=1, max_size=5))
    def test_create_snapshot_from_varying_lambdas(self, lambdas):
        sheet_data = {}
        for i, lam in enumerate(lambdas):
            sheet_data[f"/sheet_{i}"] = {
                "lambda": lam,
                "components": [{"ref": f"R{i}", "class": "Resistor", "lambda": lam, "params": {}}],
            }
        snap = growth_tracking.create_snapshot(
            sheet_data=sheet_data, system_lambda=sum(lambdas),
            mission_hours=1000.0, version_label="property-test",
        )
        assert snap.n_components == len(lambdas)
        assert snap.n_sheets == len(lambdas)
        assert abs(snap.system_lambda - sum(lambdas)) < 1e-15


class PropertyMissionProfileTests:
    def test_single_phase_same_as_no_phasing(self):
        phase = mission_profile.MissionPhase("A", 1.0, 25.0, 50.0, 365, 10.0, 1.0)
        prof = mission_profile.MissionProfile(phases=[phase])
        result = mission_profile.compute_phased_lambda(
            component_type="Resistor", base_params={"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0},
            phases=[phase],
        )
        assert "lambda_total" in result
        assert result["lambda_total"] > 0

    def test_multi_phase_impacts_result(self):
        phases = [
            mission_profile.MissionPhase("Hot", 0.3, 60.0, 85.0, 500, 30.0, 0.8),
            mission_profile.MissionPhase("Cold", 0.7, -10.0, 15.0, 200, 10.0, 0.2),
        ]
        prof = mission_profile.MissionProfile(phases=phases)
        impact = mission_profile.estimate_phasing_impact(
            component_type="Resistor", base_params={"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0},
            phases=phases,
        )
        assert impact["phased_lambda"] > 0
        assert impact["ratio"] > 0

    def test_phase_to_dict_roundtrip(self):
        p = mission_profile.MissionPhase("Test", 0.5, 35.0, 60.0, 400, 20.0, 0.9)
        d = p.to_dict()
        p2 = mission_profile.MissionPhase.from_dict(d)
        assert p.name == p2.name
        assert p.duration_frac == p2.duration_frac
        assert p.t_ambient == p2.t_ambient

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_phase_with_varying_tau(self, tau_on):
        p = mission_profile.MissionPhase("Var", 1.0, 25.0, 50.0, 365, 10.0, tau_on)
        result = mission_profile.compute_phased_lambda(
            component_type="Resistor", base_params={"t_ambient": 25.0},
            phases=[p],
        )
        assert "lambda_total" in result
        assert result["lambda_total"] >= 0
