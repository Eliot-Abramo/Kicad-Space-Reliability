import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import growth_tracking as gt


class ReliabilitySnapshotTests(unittest.TestCase):
    def test_snapshot_creation(self):
        snap = gt.ReliabilitySnapshot(
            timestamp="2025-01-01T00:00:00",
            version_label="v1",
            notes="Initial design",
            system_lambda=1e-6,
            system_fit=1000.0,
            system_reliability=0.99,
            mission_hours=1000.0,
            n_components=50,
            n_sheets=3,
            sheet_summary={
                "/power": {"lambda": 5e-7, "fit": 500.0, "r": 0.995, "n_components": 20},
                "/logic": {"lambda": 3e-7, "fit": 300.0, "r": 0.997, "n_components": 20},
                "/io": {"lambda": 2e-7, "fit": 200.0, "r": 0.998, "n_components": 10},
            },
            component_lambdas={"U1": 1e-7, "R1": 5e-8, "C1": 2e-8},
            component_details={},
        )
        self.assertEqual(snap.version_label, "v1")
        self.assertEqual(snap.n_components, 50)
        self.assertEqual(len(snap.sheet_summary), 3)

    def test_snapshot_to_dict_roundtrip(self):
        snap = gt.ReliabilitySnapshot(
            timestamp="2025-01-01T00:00:00",
            version_label="v1",
            notes="test",
            system_lambda=1e-6,
            system_fit=1000.0,
            system_reliability=0.99,
            mission_hours=1000.0,
            n_components=10,
            n_sheets=1,
            sheet_summary={"/main": {"lambda": 1e-6, "fit": 1000.0, "r": 0.99, "n_components": 10}},
            component_lambdas={"R1": 1e-6},
            component_details={},
        )
        d = snap.to_dict()
        snap2 = gt.ReliabilitySnapshot.from_dict(d)
        self.assertEqual(snap.version_label, snap2.version_label)
        self.assertEqual(snap.system_lambda, snap2.system_lambda)
        self.assertEqual(snap.n_components, snap2.n_components)


class SaveLoadTests(unittest.TestCase):
    def setUp(self):
        self.project_dir = Path(tempfile.mkdtemp())

    def test_save_and_load_snapshots(self):
        snap = gt.ReliabilitySnapshot(
            timestamp="2025-06-01T12:00:00",
            version_label="v2",
            notes="Improved design",
            system_lambda=5e-7,
            system_fit=500.0,
            system_reliability=0.995,
            mission_hours=1000.0,
            n_components=50,
            n_sheets=2,
            sheet_summary={},
            component_lambdas={},
            component_details={},
        )
        gt.save_snapshot(snap, project_path=str(self.project_dir))
        loaded = gt.load_snapshots(project_path=str(self.project_dir))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].version_label, "v2")
        self.assertAlmostEqual(loaded[0].system_lambda, 5e-7)

    def test_load_empty_directory(self):
        empty_dir = self.project_dir / "nonexistent"
        empty_dir.mkdir(parents=True, exist_ok=True)
        loaded = gt.load_snapshots(project_path=str(empty_dir))
        self.assertEqual(len(loaded), 0)

    def test_create_snapshot_produces_valid_object(self):
        sheet_data = {
            "/main": {
                "lambda": 1e-6,
                "components": [
                    {"ref": "R1", "class": "Resistor", "lambda": 1e-6, "params": {}},
                ],
            },
        }
        snap = gt.create_snapshot(
            sheet_data=sheet_data,
            system_lambda=1e-6,
            mission_hours=1000.0,
            version_label="test-snap",
            notes="Integration test",
        )
        self.assertEqual(snap.version_label, "test-snap")
        self.assertEqual(snap.n_sheets, 1)
        self.assertEqual(snap.n_components, 1)
        self.assertAlmostEqual(snap.system_lambda, 1e-6)
        datetime.fromisoformat(snap.timestamp)


class RevisionComparisonTests(unittest.TestCase):
    def _make_snap(self, ts, ver, lam, r=0.99, fit=1000.0):
        return gt.ReliabilitySnapshot(
            timestamp=ts,
            version_label=ver,
            notes="",
            system_lambda=lam,
            system_fit=fit,
            system_reliability=r,
            mission_hours=1000.0,
            n_components=10,
            n_sheets=1,
            sheet_summary={},
            component_lambdas={"R1": lam},
            component_details={},
        )

    def test_compare_revisions_identical(self):
        snap = self._make_snap("2025-01-01T00:00:00", "v1", 1e-6)
        comparison = gt.compare_revisions(snap, snap)
        self.assertAlmostEqual(comparison.system_delta_fit, 0.0)
        self.assertAlmostEqual(comparison.reliability_improvement, 0.0)

    def test_compare_revisions_improvement(self):
        earlier = self._make_snap("2025-01-01T00:00:00", "v1", 2e-6, 0.98, 2000.0)
        later = self._make_snap("2025-06-01T00:00:00", "v2", 1e-6, 0.99, 1000.0)
        comparison = gt.compare_revisions(earlier, later)
        self.assertAlmostEqual(comparison.system_delta_fit, -1000.0)
        self.assertGreater(comparison.reliability_improvement, 0)

    def test_build_growth_timeline(self):
        snaps = [
            self._make_snap("2025-01-01T00:00:00", "v1", 2e-6, 0.98, 2000.0),
            self._make_snap("2025-06-01T00:00:00", "v2", 1e-6, 0.99, 1000.0),
        ]
        snapshots_dir = Path(tempfile.mkdtemp())
        for s in snaps:
            gt.save_snapshot(s, project_path=str(snapshots_dir))

        timeline = gt.build_growth_timeline(project_path=str(snapshots_dir))
        self.assertIsInstance(timeline, gt.GrowthTimeline)
        self.assertEqual(len(timeline.snapshots), 2)


if __name__ == "__main__":
    unittest.main()
