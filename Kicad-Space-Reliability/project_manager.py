"""
Project Manager
===============
Manages the Reliability/ folder structure for KiCad projects.
Handles configuration, reports, and assets storage.

Author:  Eliot Abramo
"""

import os
from pathlib import Path
from typing import Optional, Dict


class ProjectManager:
    """Manages project-specific Reliability folder and files."""
    
    RELIABILITY_FOLDER = "Reliability"
    CONFIG_FILENAME = "reliability_config.json"
    LOGO_FILENAME = "logo.png"
    
    def __init__(self, project_path: str):
        """
        Initialize project manager.
        
        Args:
            project_path: Path to KiCad project directory (parent of .kicad_pro)
        """
        self.project_path = Path(project_path)
        self.reliability_dir = self.project_path / self.RELIABILITY_FOLDER
    
    def ensure_reliability_folder(self) -> Path:
        """
        Ensure Reliability folder exists in project.
        Creates it if doesn't exist.
        
        Returns:
            Path to Reliability folder
        """
        self.reliability_dir.mkdir(parents=True, exist_ok=True)
        return self.reliability_dir
    
    def get_reliability_folder(self) -> Path:
        """
        Get path to Reliability folder.
        Does not create it if it doesn't exist.
        
        Returns:
            Path to Reliability folder
        """
        return self.reliability_dir
    
    def get_config_path(self) -> Path:
        """Get path to configuration file in Reliability folder."""
        return self.reliability_dir / self.CONFIG_FILENAME
    
    def get_logo_path(self) -> Path:
        """Get path to logo.png file in Reliability folder."""
        return self.reliability_dir / self.LOGO_FILENAME
    
    def logo_exists(self) -> bool:
        """Check if logo.png exists in Reliability folder."""
        logo_path = self.get_logo_path()
        return logo_path.exists() and logo_path.is_file()
    
    def get_reports_folder(self) -> Path:
        """
        Get path to Reports subfolder in Reliability folder.
        Creates it if doesn't exist.
        
        Returns:
            Path to Reports folder
        """
        reports_dir = self.reliability_dir / "Reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir
    
    def get_available_logo_path(self) -> Optional[Path]:
        """
        Get path to logo if it exists, None otherwise.
        
        Returns:
            Path to logo.png or None
        """
        if self.logo_exists():
            return self.get_logo_path()
        return None
    
    def get_folder_structure_info(self) -> Dict[str, str]:
        """
        Get information about the project's Reliability folder structure.
        
        Returns:
            Dictionary with folder paths and information
        """
        self.ensure_reliability_folder()
        reports_folder = self.get_reports_folder()
        
        return {
            "project_path": str(self.project_path),
            "reliability_folder": str(self.reliability_dir),
            "config_file": str(self.get_config_path()),
            "logo_file": str(self.get_logo_path()),
            "reports_folder": str(reports_folder),
            "logo_exists": self.logo_exists(),
        }


def initialize_project_folder(project_path: str) -> ProjectManager:
    """
    Initialize Reliability folder structure for a project.
    
    Args:
        project_path: Path to KiCad project
        
    Returns:
        ProjectManager instance
    """
    manager = ProjectManager(project_path)
    manager.ensure_reliability_folder()
    manager.get_reports_folder()
    return manager
