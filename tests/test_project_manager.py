"""Tests for the ProjectManager module — filesystem operations with temp dirs."""

import json
import tempfile
from pathlib import Path

import project_manager as pm


class ProjectManagerTests:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.manager = pm.ProjectManager(str(self.tmpdir))

    def test_creates_reliability_folder(self):
        self.manager.ensure_reliability_folder()
        assert (self.tmpdir / "Reliability").exists()
        assert (self.tmpdir / "Reliability").is_dir()

    def test_reliability_folder_exists_after_creation(self):
        assert not self.manager.reliability_folder_exists()
        self.manager.ensure_reliability_folder()
        assert self.manager.reliability_folder_exists()

    def test_get_data_path(self):
        self.manager.ensure_reliability_folder()
        path = self.manager.get_data_path()
        assert path.name == "reliability_data.json"
        assert path.parent == self.tmpdir / "Reliability"

    def test_get_reports_folder_creates_and_returns(self):
        reports = self.manager.get_reports_folder()
        assert reports.exists()
        assert reports.name == "Reports"

    def test_save_and_load_data(self):
        data = {"test": True, "value": 42}
        success = self.manager.save_data(data)
        assert success
        loaded = self.manager.load_data()
        assert loaded == data

    def test_load_data_nonexistent_returns_none(self):
        result = self.manager.load_data()
        assert result is None

    def test_save_data_creates_reliability_folder(self):
        data = {"key": "val"}
        self.manager.save_data(data)
        assert (self.tmpdir / "Reliability").exists()

    def test_default_data_structure(self):
        data = pm.ProjectManager.default_data()
        assert "components" in data
        assert "structure" in data
        assert "settings" in data
        assert data["settings"]["years"] == 5

    def test_get_folder_structure_info(self):
        self.manager.ensure_reliability_folder()
        info = self.manager.get_folder_structure_info()
        assert info["project_path"] == str(self.tmpdir)
        assert "Reliability" in info["reliability_folder"]
        assert info["logo_exists"] is False
        assert info["logo_file"] == "(none)"

    def test_logo_path_none_when_no_logo(self):
        logo = self.manager.get_logo_path()
        assert logo is None

    def test_logo_path_finds_logo(self):
        self.manager.ensure_reliability_folder()
        logo_file = self.tmpdir / "Reliability" / "logo.png"
        logo_file.write_text("fake-image-data")
        found = self.manager.get_logo_path()
        assert found == logo_file

    def test_logo_exists(self):
        assert not self.manager.logo_exists()
        self.manager.ensure_reliability_folder()
        (self.tmpdir / "Reliability" / "logo.svg").write_text("<svg></svg>")
        assert self.manager.logo_exists()

    def test_logo_mime_type_png(self):
        self.manager.ensure_reliability_folder()
        (self.tmpdir / "Reliability" / "logo.png").write_text("data")
        assert self.manager.get_logo_mime_type() == "image/png"

    def test_logo_mime_type_jpg(self):
        self.manager.ensure_reliability_folder()
        (self.tmpdir / "Reliability" / "logo.jpg").write_text("data")
        assert self.manager.get_logo_mime_type() == "image/jpeg"

    def test_logo_mime_type_none_when_no_logo(self):
        assert self.manager.get_logo_mime_type() is None

    def test_initialize_project_folder(self):
        tmp = Path(tempfile.mkdtemp())
        manager = pm.initialize_project_folder(str(tmp))
        assert isinstance(manager, pm.ProjectManager)
        assert (tmp / "Reliability").exists()
        assert (tmp / "Reliability" / "Reports").exists()

    def test_load_data_corrupted_json_returns_none(self):
        self.manager.ensure_reliability_folder()
        data_path = self.manager.get_data_path()
        data_path.write_text("this is not valid json")
        assert self.manager.load_data() is None

    def test_get_reliability_folder(self):
        self.manager.ensure_reliability_folder()
        assert self.manager.get_reliability_folder() == self.tmpdir / "Reliability"
