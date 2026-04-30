import json
import pathlib

_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[1] if _DIR.parent.name == "mutants" else _DIR.parents[0]


class PublishReadinessTests:
    def test_public_surfaces_do_not_advertise_unshipped_sobol_feature(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        metadata = json.loads((REPO_ROOT / "metadata.json").read_text(encoding="utf-8"))
        reliability_dialog = (REPO_ROOT / "plugins" / "reliability_dialog.py").read_text(encoding="utf-8")

        assert "Sobol sensitivity analysis" not in readme
        assert "Sobol sensitivity analysis" not in metadata["description_full"]
        assert "Sobol sensitivity analysis" not in reliability_dialog

    def test_release_version_is_consistent_in_public_files(self):
        expected = "3.3.0"
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        metadata = json.loads((REPO_ROOT / "metadata.json").read_text(encoding="utf-8"))
        package_init = (REPO_ROOT / "plugins" / "__init__.py").read_text(encoding="utf-8")
        reliability_math = (REPO_ROOT / "plugins" / "reliability_math.py").read_text(encoding="utf-8")
        report_generator = (REPO_ROOT / "plugins" / "report_generator.py").read_text(encoding="utf-8")

        assert f"<strong>Version:</strong> {expected}" in readme
        assert metadata["versions"][0]["version"] == expected
        assert f'__version__ = "{expected}"' in package_init
        assert f'__version__ = "{expected}"' in reliability_math
        assert f'PLUGIN_VERSION = "{expected}"' in report_generator

    def test_report_builder_wires_every_advertised_analysis_section(self):
        analysis_dialog = (REPO_ROOT / "plugins" / "analysis_dialog.py").read_text(encoding="utf-8")

        assert "monte_carlo=mc_dict" in analysis_dialog
        assert "tornado=tornado_dict" in analysis_dialog
        assert "design_margin=scenario_dict" in analysis_dialog
        assert "criticality=crit_list" in analysis_dialog

    def test_readme_points_to_methodology_doc(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        methodology = REPO_ROOT / "docs" / "METHODOLOGY.md"

        assert methodology.exists()
        assert "./docs/METHODOLOGY.md" in readme

    def test_windows_ui_policy_is_wired_into_source_files(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        theme_py = (REPO_ROOT / "plugins" / "ui" / "theme.py").read_text(encoding="utf-8")
        analysis_dialog = (REPO_ROOT / "plugins" / "analysis_dialog.py").read_text(encoding="utf-8")
        reliability_dialog = (REPO_ROOT / "plugins" / "reliability_dialog.py").read_text(encoding="utf-8")
        component_editor = (REPO_ROOT / "plugins" / "component_editor.py").read_text(encoding="utf-8")

        assert "Cross-platform UI validation" in readme
        assert 'if IS_WINDOWS:\n        return "light"' in theme_py
        assert "def apply_theme_recursively(" in theme_py
        assert "WINDOWS_FONT_POINT_DELTA = -2" in theme_py
        assert "SegmentedBook" in analysis_dialog
        assert "SegmentedBook" in component_editor
        assert "apply_theme_recursively(self, background=C.BG)" in analysis_dialog
        assert "apply_theme_recursively(self.nb, background=C.BG)" in analysis_dialog
        assert "apply_theme_recursively(self, background=Colors.BACKGROUND)" in reliability_dialog
