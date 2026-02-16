"""
Project Manager
===============
Manages the Reliability/ folder structure for KiCad projects.
Handles configuration, component data, block setup, and reports.

Author:  Eliot Abramo
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


class ProjectManager:
    """Manages project-specific Reliability folder and files."""

    RELIABILITY_FOLDER = "Reliability"
    DATA_FILENAME = "reliability_data.json"
    LOGO_EXTENSIONS = [".png", ".jpg", ".jpeg", ".svg", ".bmp", ".gif"]

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.reliability_dir = self.project_path / self.RELIABILITY_FOLDER

    def reliability_folder_exists(self) -> bool:
        """Check if Reliability/ folder exists."""
        return self.reliability_dir.exists() and self.reliability_dir.is_dir()

    def ensure_reliability_folder(self) -> Path:
        self.reliability_dir.mkdir(parents=True, exist_ok=True)
        return self.reliability_dir

    def get_reliability_folder(self) -> Path:
        return self.reliability_dir

    def get_data_path(self) -> Path:
        return self.reliability_dir / self.DATA_FILENAME
    
    def get_logo_path(self) -> Optional[Path]:
        """Get path to logo file, trying multiple extensions."""
        for ext in self.LOGO_EXTENSIONS:
            path = self.reliability_dir / f"logo{ext}"
            if path.exists() and path.is_file():
                return path
        return None
    
    def logo_exists(self) -> bool:
        """Check if any logo file exists in Reliability folder."""
        return self.get_logo_path() is not None
    
    def get_logo_mime_type(self) -> Optional[str]:
        """Get MIME type for the logo file."""
        logo = self.get_logo_path()
        if not logo:
            return None
        ext = logo.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
        }
        return mime_map.get(ext, "image/png")
    
    def get_reports_folder(self) -> Path:
        reports_dir = self.reliability_dir / "Reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir
    
    def get_available_logo_path(self) -> Optional[Path]:
        return self.get_logo_path()
    
    def get_folder_structure_info(self) -> Dict[str, str]:
        self.ensure_reliability_folder()
        reports_folder = self.get_reports_folder()
        logo = self.get_logo_path()
        return {
            "project_path": str(self.project_path),
            "reliability_folder": str(self.reliability_dir),
            "data_file": str(self.get_data_path()),
            "logo_file": str(logo) if logo else "(none)",
            "reports_folder": str(reports_folder),
            "logo_exists": self.logo_exists(),
        }

    def load_data(self) -> Optional[Dict[str, Any]]:
        """Load reliability_data.json. Returns None if file doesn't exist."""
        path = self.get_data_path()
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save_data(self, data: Dict[str, Any]) -> bool:
        """Save reliability_data.json. Creates Reliability/ if needed."""
        try:
            self.ensure_reliability_folder()
            path = self.get_data_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except OSError:
            return False

    @staticmethod
    def default_data() -> Dict[str, Any]:
        """Default reliability data: blank canvas, default settings."""
        return {
            "components": {},
            "structure": {"blocks": {}, "root": None, "mission_hours": 43800},
            "settings": {"years": 5, "cycles": 5256, "dt": 3.0, "tau_on": 1.0},
            "mission_profile": None,
        }


def initialize_project_folder(project_path: str) -> ProjectManager:
    manager = ProjectManager(project_path)
    manager.ensure_reliability_folder()
    manager.get_reports_folder()
    return manager
