"""
KiCad Action Plugin for Reliability Calculation
"""

import os
import pcbnew
import wx
from pathlib import Path


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
    except Exception:
        pass
    return None


class ReliabilityPlugin(pcbnew.ActionPlugin):
    """KiCad Action Plugin for reliability analysis."""

    def defaults(self):
        self.name = "Reliability Calculator"
        self.category = "Analysis"
        self.description = "IEC TR 62380 reliability analysis with block diagram editor"
        self.show_toolbar_button = True
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            self.icon_file_name = icon_path

    def Run(self):
        """Main entry point."""
        parent = None
        try:
            tops = [w for w in wx.GetTopLevelWindows() if w and w.IsShown()]
            parent = tops[0] if tops else None
        except Exception:
            pass

        project_path = get_kicad_project_path()

        try:
            from .reliability_dialog import ReliabilityMainDialog

            dlg = ReliabilityMainDialog(parent, project_path)
            dlg.ShowModal()
            dlg.Destroy()
        except Exception as e:
            wx.MessageBox(
                f"Error launching Reliability Calculator:\n\n{str(e)}",
                "Plugin Error",
                wx.OK | wx.ICON_ERROR,
            )
            import traceback

            traceback.print_exc()


def run_standalone(project_path=None):
    """Run standalone for testing."""
    app = wx.App()
    from .reliability_dialog import ReliabilityMainDialog

    dlg = ReliabilityMainDialog(None, project_path)
    dlg.ShowModal()
    dlg.Destroy()


if __name__ == "__main__":
    run_standalone()
