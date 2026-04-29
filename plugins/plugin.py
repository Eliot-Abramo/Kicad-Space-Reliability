"""
KiCad Action Plugin for Reliability Calculation

Author:  Eliot Abramo
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pcbnew
import wx


def get_kicad_project_path():
    """Get current project path from KiCad."""
    try:
        board = pcbnew.GetBoard()
        if board:
            board_file = board.GetFileName()
            if board_file:
                board_path = Path(board_file)
                pro_file = board_path.with_suffix(".kicad_pro")
                if pro_file.exists():
                    return str(pro_file.parent)
                return str(board_path.parent)
    except Exception:  # noqa: BLE001
        pass
    return None


def get_kicad_parent_window():
    """Best-effort parent selection across KiCad 8/9/10 on Linux and Windows."""
    try:
        active = wx.GetActiveWindow()
        if active and active.IsShown():
            return active
    except Exception:  # noqa: BLE001
        pass

    try:
        app = wx.GetApp()
        top = app.GetTopWindow() if app else None
        if top and top.IsShown():
            return top
    except Exception:  # noqa: BLE001
        pass

    try:
        tops = [w for w in wx.GetTopLevelWindows() if w and w.IsShown()]
        if not tops:
            return None

        def score(window):
            title = ""
            with contextlib.suppress(Exception):
                title = window.GetTitle().lower()
            points = 0
            if "pcb editor" in title or "pcbnew" in title:
                points += 5
            if "kicad" in title:
                points += 3
            if "editor" in title:
                points += 1
            return points

        tops.sort(key=score, reverse=True)
        return tops[0]
    except Exception:  # noqa: BLE001
        return None


class ReliabilityPlugin(pcbnew.ActionPlugin):
    """KiCad Action Plugin for reliability analysis."""

    def defaults(self):
        self.name = "Reliability Calculator"
        self.category = "Analysis"
        self.description = "IEC TR 62380 reliability analysis with block diagram editor"
        self.show_toolbar_button = True
        plugin_dir = Path(__file__).resolve().parent
        icon_path = plugin_dir / "icon.png"
        if icon_path.exists():
            self.icon_file_name = icon_path
            self.dark_icon_file_name = icon_path

    def Run(self):  # noqa: N802
        """Main entry point."""
        parent = get_kicad_parent_window()

        project_path = get_kicad_project_path()

        if not project_path:
            wx.MessageBox("Could not determine KiCad project path.", "Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            from .reliability_dialog import ReliabilityMainDialog

            dlg = ReliabilityMainDialog(parent, project_path)
            dlg.ShowModal()
            dlg.Destroy()
        except Exception as e:  # noqa: BLE001
            wx.MessageBox(f"Error launching Reliability Calculator:\n\n{e!s}", "Plugin Error", wx.OK | wx.ICON_ERROR)
            import traceback

            traceback.print_exc()


def run_standalone(project_path=None):
    """Run standalone for testing."""
    wx.App()
    from .reliability_dialog import ReliabilityMainDialog

    dlg = ReliabilityMainDialog(None, project_path)
    dlg.ShowModal()
    dlg.Destroy()


if __name__ == "__main__":
    run_standalone()
