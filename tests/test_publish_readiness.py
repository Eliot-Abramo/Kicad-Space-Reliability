import json
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class PublishReadinessTests(unittest.TestCase):
    def test_public_surfaces_do_not_advertise_unshipped_sobol_feature(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        metadata = json.loads((REPO_ROOT / "metadata.json").read_text(encoding="utf-8"))
        reliability_dialog = (REPO_ROOT / "plugins" / "reliability_dialog.py").read_text(encoding="utf-8")

        self.assertNotIn("Sobol sensitivity analysis", readme)
        self.assertNotIn("Sobol sensitivity analysis", metadata["description_full"])
        self.assertNotIn("Sobol sensitivity analysis", reliability_dialog)

    def test_release_version_is_consistent_in_public_files(self):
        expected = "3.3.0"
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        metadata = json.loads((REPO_ROOT / "metadata.json").read_text(encoding="utf-8"))
        package_init = (REPO_ROOT / "plugins" / "__init__.py").read_text(encoding="utf-8")
        reliability_math = (REPO_ROOT / "plugins" / "reliability_math.py").read_text(encoding="utf-8")
        report_generator = (REPO_ROOT / "plugins" / "report_generator.py").read_text(encoding="utf-8")

        self.assertIn(f"**Version:** {expected}", readme)
        self.assertEqual(metadata["versions"][0]["version"], expected)
        self.assertIn(f'__version__ = "{expected}"', package_init)
        self.assertIn(f'__version__ = "{expected}"', reliability_math)
        self.assertIn(f'PLUGIN_VERSION = "{expected}"', report_generator)

    def test_report_builder_wires_every_advertised_analysis_section(self):
        analysis_dialog = (REPO_ROOT / "plugins" / "analysis_dialog.py").read_text(encoding="utf-8")

        self.assertIn("monte_carlo=mc_dict", analysis_dialog)
        self.assertIn("tornado=tornado_dict", analysis_dialog)
        self.assertIn("design_margin=scenario_dict", analysis_dialog)
        self.assertIn("criticality=crit_list", analysis_dialog)

    def test_readme_points_to_methodology_doc(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        methodology = REPO_ROOT / "docs" / "METHODOLOGY.md"

        self.assertTrue(methodology.exists())
        self.assertIn("./docs/METHODOLOGY.md", readme)

    def test_windows_ui_policy_is_wired_into_source_files(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        theme_py = (REPO_ROOT / "plugins" / "ui" / "theme.py").read_text(encoding="utf-8")
        analysis_dialog = (REPO_ROOT / "plugins" / "analysis_dialog.py").read_text(encoding="utf-8")
        reliability_dialog = (REPO_ROOT / "plugins" / "reliability_dialog.py").read_text(encoding="utf-8")
        component_editor = (REPO_ROOT / "plugins" / "component_editor.py").read_text(encoding="utf-8")

        self.assertIn("Cross-platform UI validation", readme)
        self.assertIn('if IS_WINDOWS:\n        return "dark"', theme_py)
        self.assertIn("def apply_theme_recursively(", theme_py)
        self.assertIn("WINDOWS_FONT_POINT_DELTA = -3", theme_py)
        self.assertIn("SegmentedBook", analysis_dialog)
        self.assertIn("SegmentedBook", component_editor)
        self.assertIn("apply_theme_recursively(self, background=C.BG)", analysis_dialog)
        self.assertIn("apply_theme_recursively(self.nb, background=C.BG)", analysis_dialog)
        self.assertIn("apply_theme_recursively(self, background=Colors.BACKGROUND)", reliability_dialog)


if __name__ == "__main__":
    unittest.main()
