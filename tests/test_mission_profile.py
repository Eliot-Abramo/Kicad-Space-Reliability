import mission_profile as mp
import pytest


class MissionPhaseTests:
    def test_default_phase(self):
        p = mp.MissionPhase()
        assert p.name == "Nominal"
        assert p.duration_frac == 1.0
        assert p.t_ambient == 25.0
        assert p.t_junction is None
        assert p.n_cycles == 5256
        assert p.delta_t == 3.0
        assert p.tau_on == 1.0

    def test_custom_phase(self):
        p = mp.MissionPhase("Burn-in", 0.5, 60.0, 85.0, 100, 20.0, 0.8)
        assert p.name == "Burn-in"
        assert p.duration_frac == 0.5
        assert p.t_ambient == 60.0
        assert p.t_junction == 85.0
        assert p.n_cycles == 100
        assert p.delta_t == 20.0
        assert p.tau_on == 0.8

    def test_to_dict_roundtrip(self):
        p = mp.MissionPhase("Orbit", 0.7, 30.0, 55.0, 365, 15.0, 0.9)
        d = p.to_dict()
        p2 = mp.MissionPhase.from_dict(d)
        assert p.name == p2.name
        assert p.duration_frac == p2.duration_frac
        assert p.t_ambient == p2.t_ambient
        assert p.t_junction == p2.t_junction
        assert p.n_cycles == p2.n_cycles
        assert p.delta_t == p2.delta_t
        assert p.tau_on == p2.tau_on

    def test_from_dict_empty(self):
        p = mp.MissionPhase.from_dict({})
        assert p.name == "Nominal"
        assert p.duration_frac == 1.0

    def test_from_dict_partial(self):
        p = mp.MissionPhase.from_dict({"name": "Test", "duration_frac": 0.3})
        assert p.name == "Test"
        assert p.duration_frac == 0.3
        assert p.t_ambient == 25.0


class MissionProfileTests:
    def test_default_profile_is_single_phase(self):
        prof = mp.MissionProfile()
        assert prof.is_single_phase
        assert len(prof.phases) == 1
        assert prof.phases[0].name == "Nominal"

    def test_multi_phase_profile(self):
        phases = [
            mp.MissionPhase("Burn-in", 0.1, 60.0, 85.0, 100, 20.0, 0.8),
            mp.MissionPhase("Orbit", 0.9, 25.0, 50.0, 500, 10.0, 1.0),
        ]
        prof = mp.MissionProfile(phases=phases, mission_years=7.0)
        assert not prof.is_single_phase
        assert len(prof.phases) == 2

    def test_phases_sum_to_one(self):
        phases = [
            mp.MissionPhase("A", 0.3, 25.0),
            mp.MissionPhase("B", 0.5, 35.0),
            mp.MissionPhase("C", 0.2, 45.0),
        ]
        total = sum(p.duration_frac for p in phases)
        assert total == pytest.approx(1.0)

    def test_mission_templates_exist(self):
        assert "LEO Satellite" in mp.MISSION_TEMPLATES
        assert "GEO Satellite" in mp.MISSION_TEMPLATES

    def test_mission_template_has_phases(self):
        leo = mp.MISSION_TEMPLATES["LEO Satellite"]
        assert len(leo.phases) >= 1

    def test_to_dict_roundtrip(self):
        phases = [
            mp.MissionPhase("A", 0.3, 25.0, 45.0, 200, 8.0, 0.9),
            mp.MissionPhase("B", 0.7, 35.0, 55.0, 300, 12.0, 0.8),
        ]
        prof = mp.MissionProfile(phases=phases, mission_years=5.0)
        d = prof.to_dict()
        prof2 = mp.MissionProfile.from_dict(d)
        assert len(prof.phases) == len(prof2.phases)
        for a, b in zip(prof.phases, prof2.phases):
            assert a.name == b.name
            assert a.duration_frac == b.duration_frac

    def test_compute_phased_lambda(self):
        phases = [
            mp.MissionPhase("A", 1.0, 25.0, 50.0, 365, 10.0, 1.0),
        ]
        result = mp.compute_phased_lambda(
            component_type="Resistor",
            base_params={"t_ambient": 25.0},
            phases=phases,
        )
        assert "lambda_total" in result
        assert result["lambda_total"] > 0

    def test_estimate_phasing_impact(self):
        phases = [
            mp.MissionPhase("A", 0.5, 25.0, 50.0, 200, 8.0, 0.9),
            mp.MissionPhase("B", 0.5, 45.0, 70.0, 400, 20.0, 0.7),
        ]
        impact = mp.estimate_phasing_impact(
            component_type="Resistor",
            base_params={"t_ambient": 25.0},
            phases=phases,
        )
        assert "phased_lambda" in impact
        assert "ratio" in impact
