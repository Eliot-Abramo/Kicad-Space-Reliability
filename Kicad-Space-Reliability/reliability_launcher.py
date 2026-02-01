#!/usr/bin/env python3
"""
Reliability Calculator - Main Launcher

This is the primary entry point for the Reliability Calculator.
It can be launched:
1. From KiCad via the BOM generator (bom_reliability.py)
2. Directly by double-clicking this file
3. From command line: python reliability_launcher.py [project_path]

On first launch without a project, it shows a welcome dialog to browse for a project.
"""

import sys
import os

# Ensure the plugin directory is in the path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

import wx
import json
from pathlib import Path


class ProjectSelector(wx.Dialog):
    """
    Welcome dialog for selecting a KiCad project.
    Shows recent projects and allows browsing for new ones.
    """

    def __init__(self, parent=None):
        super().__init__(
            parent,
            title="Reliability Calculator",
            size=(500, 400),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        self.selected_project = None
        self.recent_projects = self._load_recent_projects()

        self._create_ui()
        self.Centre()

    def _create_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        header = wx.StaticText(panel, label="⚡ Reliability Calculator")
        header_font = header.GetFont()
        header_font.SetPointSize(16)
        header_font.SetWeight(wx.FONTWEIGHT_BOLD)
        header.SetFont(header_font)
        main_sizer.Add(header, 0, wx.ALL | wx.ALIGN_CENTER, 15)

        # Description
        desc = wx.StaticText(
            panel,
            label="Calculate component and system reliability for your KiCad project.\n"
            "Select a project to get started.",
            style=wx.ALIGN_CENTER,
        )
        desc.SetForegroundColour(wx.Colour(80, 80, 80))
        main_sizer.Add(desc, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        main_sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 10)

        # Recent projects
        if self.recent_projects:
            recent_label = wx.StaticText(panel, label="Recent Projects:")
            recent_label.SetFont(recent_label.GetFont().Bold())
            main_sizer.Add(recent_label, 0, wx.LEFT | wx.TOP, 15)

            self.recent_list = wx.ListBox(
                panel, choices=self.recent_projects, style=wx.LB_SINGLE
            )
            self.recent_list.Bind(wx.EVT_LISTBOX_DCLICK, self.on_recent_dclick)
            main_sizer.Add(self.recent_list, 1, wx.EXPAND | wx.ALL, 10)

            open_recent_btn = wx.Button(panel, label="Open Selected")
            open_recent_btn.Bind(wx.EVT_BUTTON, self.on_open_recent)
            main_sizer.Add(open_recent_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        main_sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # Browse button
        browse_sizer = wx.BoxSizer(wx.HORIZONTAL)

        browse_btn = wx.Button(panel, label="📁 Browse for Project...", size=(200, 40))
        browse_btn.SetFont(browse_btn.GetFont().Bold())
        browse_btn.Bind(wx.EVT_BUTTON, self.on_browse)
        browse_sizer.Add(browse_btn, 0, wx.ALL, 10)

        main_sizer.Add(browse_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        # Cancel button
        cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        main_sizer.Add(cancel_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM, 15)

        panel.SetSizer(main_sizer)

    def _get_config_path(self) -> Path:
        """Get the path for storing configuration."""
        if sys.platform == "win32":
            config_dir = Path(os.environ.get("APPDATA", "")) / "kicad_reliability"
        else:
            config_dir = Path.home() / ".config" / "kicad_reliability"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "recent_projects.json"

    def _load_recent_projects(self) -> list:
        """Load list of recent projects."""
        try:
            config_path = self._get_config_path()
            if config_path.exists():
                with open(config_path, "r") as f:
                    data = json.load(f)
                    # Filter to only existing projects
                    return [p for p in data.get("recent", []) if Path(p).exists()]
        except Exception:
            pass
        return []

    def _save_recent_project(self, project_path: str):
        """Add a project to the recent list."""
        try:
            recent = self._load_recent_projects()
            # Remove if already exists
            if project_path in recent:
                recent.remove(project_path)
            # Add to front
            recent.insert(0, project_path)
            # Keep only last 10
            recent = recent[:10]

            config_path = self._get_config_path()
            with open(config_path, "w") as f:
                json.dump({"recent": recent}, f)
        except Exception:
            pass

    def on_browse(self, event):
        """Browse for a KiCad project."""
        dlg = wx.DirDialog(
            self,
            "Select KiCad Project Directory",
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
        )

        if dlg.ShowModal() == wx.ID_OK:
            project_path = dlg.GetPath()

            # Verify it's a KiCad project
            pro_files = list(Path(project_path).glob("*.kicad_pro"))
            sch_files = list(Path(project_path).glob("*.kicad_sch"))

            if not pro_files and not sch_files:
                wx.MessageBox(
                    "The selected directory doesn't appear to contain a KiCad project.\n"
                    "Please select a directory with .kicad_pro or .kicad_sch files.",
                    "Not a KiCad Project",
                    wx.OK | wx.ICON_WARNING,
                )
                dlg.Destroy()
                return

            self.selected_project = project_path
            self._save_recent_project(project_path)
            self.EndModal(wx.ID_OK)

        dlg.Destroy()

    def on_open_recent(self, event):
        """Open the selected recent project."""
        selection = self.recent_list.GetSelection()
        if selection != wx.NOT_FOUND:
            self.selected_project = self.recent_projects[selection]
            self.EndModal(wx.ID_OK)

    def on_recent_dclick(self, event):
        """Handle double-click on recent project."""
        self.on_open_recent(event)


def main():
    """Main entry point."""
    app = wx.App()

    # Check if a project path was provided
    project_path = None

    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1])
        if arg_path.exists():
            if arg_path.is_dir():
                project_path = str(arg_path)
            else:
                # It's a file, use parent directory
                project_path = str(arg_path.parent)

    # If no project provided, show selector
    if not project_path:
        selector = ProjectSelector()
        if selector.ShowModal() == wx.ID_OK:
            project_path = selector.selected_project
        selector.Destroy()

        if not project_path:
            return 0

    # Import and launch the main dialog
    from reliability_dialog import ReliabilityMainDialog

    dlg = ReliabilityMainDialog(None, project_path)
    dlg.ShowModal()
    dlg.Destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
