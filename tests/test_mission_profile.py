import unittest

import mission_profile as mp


class MissionPhaseTests(unittest.TestCase):
    def test_default_phase(self):
        p = mp.MissionPhase()
        self.assertEqual(p.name, "Nominal")
        self.assertEqual(p.duration_frac, 1.0)
        self.assertEqual(p.t_ambient, 25.0)
        self.assertIsNone(p.t_junction)
        self.assertEqual(p.n_cycles, 5256)
        self.assertEqual(p.delta_t, 3.0)
        self.assertEqual(p.tau_on, 1.0)

    def test_custom_phase(self):
        p = mp.MissionPhase("Burn-in", 0.5, 60.0, 85.0, 100, 20.0, 0.8)
        self.assertEqual(p.name, "Burn-in")
        self.assertEqual(p.duration_frac, 0.5)
        self.assertEqual(p.t_ambient, 60.0)
        self.assertEqual(p.t_junction, 85.0)
        self.assertEqual(p.n_cycles, 100)
        self.assertEqual(p.delta_t, 20.0)
        self.assertEqual(p.tau_on, 0.8)

    def test_to_dict_roundtrip(self):
        p = mp.MissionPhase("Orbit", 0.7, 30.0, 55.0, 365, 15.0, 0.9)
        d = p.to_dict()
        p2 = mp.MissionPhase.from_dict(d)
        self.assertEqual(p.name, p2.name)
        self.assertEqual(p.duration_frac, p2.duration_frac)
        self.assertEqual(p.t_ambient, p2.t_ambient)
        self.assertEqual(p.t_junction, p2.t_junction)
        self.assertEqual(p.n_cycles, p2.n_cycles)
        self.assertEqual(p.delta_t, p2.delta_t)
        self.assertEqual(p.tau_on, p2.tau_on)

    def test_from_dict_empty(self):
        p = mp.MissionPhase.from_dict({})
        self.assertEqual(p.name, "Nominal")
        self.assertEqual(p.duration_frac, 1.0)

    def test_from_dict_partial(self):
        p = mp.MissionPhase.from_dict({"name": "Test", "duration_frac": 0.3})
        self.assertEqual(p.name, "Test")
        self.assertEqual(p.duration_frac, 0.3)
        self.assertEqual(p.t_ambient, 25.0)


class MissionProfileTests(unittest.TestCase):
    def test_default_profile_is_single_phase(self):
        prof = mp.MissionProfile()
        self.assertTrue(prof.is_single_phase)
        self.assertEqual(len(prof.phases), 1)
        self.assertEqual(prof.phases[0].name, "Nominal")

    def test_multi_phase_profile(self):
        phases = [
            mp.MissionPhase("Burn-in", 0.1, 60.0, 85.0, 100, 20.0, 0.8),
            mp.MissionPhase("Orbit", 0.9, 25.0, 50.0, 500, 10.0, 1.0),
        ]
        prof = mp.MissionProfile(phases=phases, mission_years=7.0)
        self.assertFalse(prof.is_single_phase)
        self.assertEqual(len(prof.phases), 2)

    def test_phases_sum_to_one(self):
        phases = [
            mp.MissionPhase("A", 0.3, 25.0),
            mp.MissionPhase("B", 0.5, 35.0),
            mp.MissionPhase("C", 0.2, 45.0),
        ]
        total = sum(p.duration_frac for p in phases)
        self.assertAlmostEqual(total, 1.0)

    def test_mission_templates_exist(self):
        self.assertIn("LEO Satellite", mp.MISSION_TEMPLATES)
        self.assertIn("GEO Satellite", mp.MISSION_TEMPLATES)

    def test_mission_template_has_phases(self):
        leo = mp.MISSION_TEMPLATES["LEO Satellite"]
        self.assertTrue(len(leo.phases) >= 1)

    def test_to_dict_roundtrip(self):
        phases = [
            mp.MissionPhase("A", 0.3, 25.0, 45.0, 200, 8.0, 0.9),
            mp.MissionPhase("B", 0.7, 35.0, 55.0, 300, 12.0, 0.8),
        ]
        prof = mp.MissionProfile(phases=phases, mission_years=5.0)
        d = prof.to_dict()
        prof2 = mp.MissionProfile.from_dict(d)
        self.assertEqual(len(prof.phases), len(prof2.phases))
        for a, b in zip(prof.phases, prof2.phases):
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.duration_frac, b.duration_frac)

    def test_compute_phased_lambda(self):
        phases = [
            mp.MissionPhase("A", 1.0, 25.0, 50.0, 365, 10.0, 1.0),
        ]
        result = mp.compute_phased_lambda(
            component_type="Resistor",
            base_params={"t_ambient": 25.0},
            phases=phases,
        )
        self.assertIn("lambda_total", result)
        self.assertGreater(result["lambda_total"], 0)

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
        self.assertIn("phased_lambda", impact)
        self.assertIn("ratio", impact)


if __name__ == "__main__":
    unittest.main()
