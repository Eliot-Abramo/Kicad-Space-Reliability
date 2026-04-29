import pytest

import derating_engine as de


class DeratingRecommendationTests:
    def test_recommendation_to_dict(self):
        rec = de.DeratingRecommendation(
            reference="R1",
            component_type="Resistor",
            parameter="v_applied",
            current_value=10.0,
            required_value=5.0,
            change_absolute=-5.0,
            change_percent=-50.0,
            current_fit=100.0,
            target_fit=50.0,
            expected_fit=50.0,
            system_fit_reduction=50.0,
            system_fit_reduction_pct=5.0,
            feasibility="easy",
            actions=["Reduce voltage by 50%"],
            priority=1,
        )
        d = rec.to_dict()
        assert d["reference"] == "R1"
        assert d["feasibility"] == "easy"
        assert d["priority"] == 1
        assert d["actions"] == ["Reduce voltage by 50%"]


class DeratingGuidanceTests:
    def test_compute_returns_derating_result_with_recommendations(self):
        result = de.compute_derating_guidance(
            sheet_data={
                "/board": {
                    "lambda": 100e-9,
                    "components": [
                        {
                            "ref": "R1",
                            "class": "Resistor",
                            "lambda": 100e-9,
                            "params": {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0},
                        },
                    ],
                },
            },
            mission_hours=1000.0,
            target_fit=50.0,
        )
        assert isinstance(result, de.DeratingResult)
        assert isinstance(result.recommendations, list)
        assert result.system_actual_fit >= 0
        assert result.system_gap_fit >= 0

    def test_no_components_returns_empty_recommendations(self):
        result = de.compute_derating_guidance(
            sheet_data={"/board": {"lambda": 0.0, "components": []}},
            mission_hours=1000.0,
            target_fit=100.0,
        )
        assert len(result.recommendations) == 0

    def test_recommendation_has_expected_structure(self):
        result = de.compute_derating_guidance(
            sheet_data={
                "/board": {
                    "lambda": 100e-9,
                    "components": [
                        {
                            "ref": "R1",
                            "class": "Resistor",
                            "lambda": 100e-9,
                            "params": {"t_ambient": 25.0, "v_applied": 5.0, "v_rated": 10.0},
                        },
                    ],
                },
            },
            mission_hours=1000.0,
            target_fit=50.0,
        )
        if result.recommendations:
            rec = result.recommendations[0]
            assert rec.feasibility in {"easy", "moderate", "difficult", "infeasible"}
            assert rec.priority >= 1
            assert isinstance(rec.actions, list)
            assert isinstance(rec.to_dict(), dict)

    def test_result_contains_system_summary(self):
        result = de.compute_derating_guidance(
            sheet_data={
                "/board": {
                    "lambda": 100e-9,
                    "components": [],
                },
            },
            mission_hours=1000.0,
            target_fit=50.0,
        )
        assert result.system_actual_fit == 100.0
        assert result.system_target_fit == 50.0
        assert result.system_gap_fit == 50.0
