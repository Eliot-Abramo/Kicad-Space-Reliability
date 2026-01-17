"""
Main Reliability Calculator Dialog
===================================

The primary user interface for the KiCad Reliability Calculator plugin.
This module implements the main dialog window that provides:

- Visual block diagram editor for defining system topology
- Component reliability field editing
- System reliability calculations based on IEC TR 62380
- Export capabilities for reports (HTML, Markdown, CSV)

Architecture
------------
This dialog follows a clean separation of concerns:

- **reliability_math.py**: All mathematical calculations (IEC TR 62380 formulas)
- **component_editor.py**: Component field editing dialogs
- **schematic_parser.py**: KiCad schematic file parsing
- **block_editor.py**: Visual drag-and-drop block diagram editor

This module focuses purely on UI layout and event handling, delegating
all calculations and data processing to the appropriate modules.

Design System
-------------
The UI follows a consistent design system defined at the top of this module:

- **Colors**: Professional color palette with semantic colors
- **Spacing**: Consistent spacing values (XS=4, SM=8, MD=12, LG=16, XL=24)
- **Fonts**: Standardized font configurations

Usage
-----
Standalone testing::

    python -m reliability_dialog

As a KiCad plugin::

    from reliability_dialog import ReliabilityMainDialog
    dlg = ReliabilityMainDialog(parent, project_path)
    dlg.ShowModal()
    dlg.Destroy()

Author: Eliot Abramo
License: MIT
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wx
import wx.lib.scrolledpanel as scrolled

from .block_editor import BlockEditor, Block
from .reliability_math import (
    calculate_component_lambda,
    reliability_from_lambda,
    lambda_from_reliability,
    r_series,
    r_parallel,
    r_k_of_n,
    calculate_lambda,
)
from .component_editor import (
    ComponentEditorDialog,
    BatchComponentEditorDialog,
    ComponentData,
    classify_component,
    QuickReferenceDialog,
)
from .schematic_parser import SchematicParser, create_test_data


# =============================================================================
# Design System (mirrors dialogs.py)
# =============================================================================

class Colors:
    """Professional color palette (same values as dialogs.py)."""

    BACKGROUND = wx.Colour(245, 246, 247)
    PANEL_BG = wx.Colour(255, 255, 255)
    HEADER_BG = wx.Colour(38, 50, 56)
    HEADER_FG = wx.Colour(255, 255, 255)
    ACCENT = wx.Colour(30, 136, 229)
    BORDER = wx.Colour(218, 220, 224)
    TEXT_PRIMARY = wx.Colour(32, 33, 36)
    TEXT_SECONDARY = wx.Colour(95, 99, 104)
    SUCCESS = wx.Colour(67, 160, 71)
    WARNING = wx.Colour(251, 140, 0)
    ERROR = wx.Colour(229, 57, 53)
    INFO_BG = wx.Colour(227, 242, 253)
    SELECTED = wx.Colour(232, 240, 254)


class Spacing:
    """Consistent spacing values."""

    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 24


class Fonts:
    """Font configurations."""

    @staticmethod
    def header():
        return wx.Font(15, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)

    @staticmethod
    def title():
        return wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)

    @staticmethod
    def body():
        return wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

    @staticmethod
    def small():
        return wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

    @staticmethod
    def mono():
        return wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)


class BaseDialog(wx.Dialog):
    """Base dialog with common functionality (escape-to-close, min size, background)."""

    def __init__(self, parent, title: str, size: Tuple[int, int], **kwargs):
        min_w, min_h = kwargs.pop("min_size", (500, 380))
        size = (max(size[0], min_w), max(size[1], min_h))

        style = kwargs.pop("style", wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        super().__init__(parent, title=title, size=size, style=style, **kwargs)

        self.SetMinSize((min_w, min_h))
        self.SetBackgroundColour(Colors.BACKGROUND)

        self.CentreOnScreen() if parent is None else self.CentreOnParent()
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char)

    def _on_char(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.EndModal(wx.ID_CANCEL)
        else:
            event.Skip()


class IconButton(wx.Button):
    """Button with Unicode icon prefix (cheap but effective)."""

    ICONS = {
        "open": "↗",
        "save": "💾",
        "load": "📁",
        "folder": "📂",
        "edit": "✎",
        "export": "⇩",
        "calc": "↻",
        "help": "ⓘ",
    }

    def __init__(self, parent, label: str, icon: Optional[str] = None, **kwargs):
        if icon and icon in self.ICONS:
            label = f"{self.ICONS[icon]} {label}"
        super().__init__(parent, label=label, **kwargs)


class InfoBanner(wx.Panel):
    """Information banner with icon."""

    def __init__(self, parent, message: str, style: str = "info"):
        super().__init__(parent)

        colors = {
            "info": (Colors.INFO_BG, Colors.ACCENT, "ⓘ"),
            "warning": (wx.Colour(255, 243, 224), Colors.WARNING, "⚠"),
            "success": (wx.Colour(232, 245, 233), Colors.SUCCESS, "✓"),
        }
        bg, fg, icon = colors.get(style, colors["info"])

        self.SetBackgroundColour(bg)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        icon_text = wx.StaticText(self, label=icon)
        icon_text.SetForegroundColour(fg)
        icon_text.SetFont(Fonts.title())
        sizer.Add(icon_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.MD)

        msg_text = wx.StaticText(self, label=message)
        msg_text.SetForegroundColour(Colors.TEXT_PRIMARY)
        msg_text.SetFont(Fonts.body())
        msg_text.Wrap(700)
        sizer.Add(msg_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.MD)

        self.SetSizer(sizer)


class SectionHeader(wx.Panel):
    """Section header with title and subtitle."""

    def __init__(self, parent, title: str, subtitle: Optional[str] = None):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)

        sizer = wx.BoxSizer(wx.VERTICAL)

        title_text = wx.StaticText(self, label=title)
        title_text.SetFont(Fonts.title())
        title_text.SetForegroundColour(Colors.TEXT_PRIMARY)
        sizer.Add(title_text, 0, wx.BOTTOM, Spacing.XS)

        if subtitle:
            sub_text = wx.StaticText(self, label=subtitle)
            sub_text.SetFont(Fonts.small())
            sub_text.SetForegroundColour(Colors.TEXT_SECONDARY)
            sizer.Add(sub_text, 0)

        self.SetSizer(sizer)


class StatusIndicator(wx.Panel):
    """Simple status strip."""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BACKGROUND)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.icon = wx.StaticText(self, label="●")
        self.icon.SetForegroundColour(Colors.SUCCESS)
        sizer.Add(self.icon, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.XS)

        self.text = wx.StaticText(self, label="Ready")
        self.text.SetFont(Fonts.small())
        self.text.SetForegroundColour(Colors.TEXT_SECONDARY)
        sizer.Add(self.text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.XS)

        self.SetSizer(sizer)

    def set_status(self, message: str, status: str = "ok"):
        colors = {
            "ok": Colors.SUCCESS,
            "warning": Colors.WARNING,
            "error": Colors.ERROR,
            "working": Colors.ACCENT,
        }
        self.icon.SetForegroundColour(colors.get(status, Colors.SUCCESS))
        self.text.SetLabel(message)
        self.Refresh()


# =============================================================================
# Small helper widgets / panels
# =============================================================================

class SheetPanel(wx.Panel):
    """Panel listing available schematic sheets."""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)

        self.sheets: List[str] = []
        self.on_add = None
        self.on_edit = None

        main = wx.BoxSizer(wx.VERTICAL)

        header = SectionHeader(self, "Schematic Sheets", "Select sheets to add as blocks")
        main.Add(header, 0, wx.ALL | wx.EXPAND, Spacing.LG)

        self.list = wx.ListBox(self, style=wx.LB_EXTENDED)
        self.list.SetFont(Fonts.body())
        self.list.Bind(wx.EVT_LISTBOX_DCLICK, self._on_dclick)
        main.Add(self.list, 1, wx.LEFT | wx.RIGHT | wx.EXPAND, Spacing.LG)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_add = IconButton(self, "Add Selected", icon="calc", size=(-1, 32))
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        btn_row.Add(self.btn_add, 1, wx.RIGHT, Spacing.SM)

        self.btn_all = IconButton(self, "Add All", icon="calc", size=(-1, 32))
        self.btn_all.Bind(wx.EVT_BUTTON, self._on_add_all)
        btn_row.Add(self.btn_all, 1)

        main.Add(btn_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, Spacing.LG)

        self.btn_edit = IconButton(self, "Edit Components…", icon="edit", size=(-1, 34))
        self.btn_edit.Bind(wx.EVT_BUTTON, self._on_edit)
        self.btn_edit.SetToolTip("Edit reliability fields for components in the selected sheet(s)")
        main.Add(self.btn_edit, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, Spacing.LG)

        self.SetSizer(main)

    def set_sheets(self, sheets: List[str]):
        self.sheets = sheets
        self.list.Set(sheets)

    def _on_add(self, event):
        selections = self.list.GetSelections()
        if self.on_add:
            for i in selections:
                self.on_add(self.sheets[i])

    def _on_add_all(self, event):
        if self.on_add:
            for s in self.sheets:
                self.on_add(s)

    def _on_dclick(self, event):
        self._on_add(event)

    def _on_edit(self, event):
        if self.on_edit:
            selections = self.list.GetSelections()
            if selections:
                self.on_edit([self.sheets[i] for i in selections])
            else:
                wx.MessageBox("Please select a sheet first.", "No Selection", wx.OK | wx.ICON_INFORMATION)


class ComponentPanel(scrolled.ScrolledPanel):
    """Panel showing component details with edit capability."""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)

        self.current_sheet: Optional[str] = None
        self.on_component_edit = None

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        header_row = wx.BoxSizer(wx.HORIZONTAL)
        self.header = wx.StaticText(self, label="Components")
        self.header.SetFont(Fonts.title())
        self.header.SetForegroundColour(Colors.TEXT_PRIMARY)
        header_row.Add(self.header, 1, wx.ALIGN_CENTER_VERTICAL)

        self.btn_edit = IconButton(self, "Edit", icon="edit", size=(90, 30))
        self.btn_edit.Bind(wx.EVT_BUTTON, self._on_edit)
        self.btn_edit.SetToolTip("Edit selected component's reliability fields")
        header_row.Add(self.btn_edit, 0, wx.ALIGN_CENTER_VERTICAL)

        self.sizer.Add(header_row, 0, wx.ALL | wx.EXPAND, Spacing.LG)

        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE)
        self.list.SetFont(Fonts.body())
        self.list.InsertColumn(0, "Ref", width=55)
        self.list.InsertColumn(1, "Value", width=95)
        self.list.InsertColumn(2, "Type", width=120)
        self.list.InsertColumn(3, "λ (FIT)", width=85)
        self.list.InsertColumn(4, "R", width=75)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_dclick)
        self.sizer.Add(self.list, 1, wx.LEFT | wx.RIGHT | wx.EXPAND, Spacing.LG)

        self.summary = wx.StaticText(self, label="")
        self.summary.SetFont(Fonts.small())
        self.summary.SetForegroundColour(Colors.TEXT_SECONDARY)
        self.sizer.Add(self.summary, 0, wx.ALL, Spacing.LG)

        self.SetSizer(self.sizer)
        self.SetupScrolling(scroll_x=False)

    def set_data(self, sheet: str, components: List[Dict], total_lam: float, r: float):
        self.current_sheet = sheet

        label = sheet.rstrip("/").split("/")[-1] or "Root"
        self.header.SetLabel(f"Components — {label}")

        self.list.DeleteAllItems()
        for i, c in enumerate(components):
            idx = self.list.InsertItem(i, c.get("ref", "?"))
            self.list.SetItem(idx, 1, (c.get("value", "") or "")[:24])
            self.list.SetItem(idx, 2, (c.get("class", "") or "")[:24])
            lam = float(c.get("lambda", 0) or 0)
            fit = lam * 1e9
            self.list.SetItem(idx, 3, f"{fit:.2f}")
            self.list.SetItem(idx, 4, f"{float(c.get('r', 1) or 1):.6f}")

        fit_total = total_lam * 1e9
        self.summary.SetLabel(f"Sheet total: λ = {fit_total:.2f} FIT   •   R = {r:.6f}")
        self.Layout()

    def _on_edit(self, event):
        idx = self.list.GetFirstSelected()
        if idx >= 0 and self.on_component_edit:
            ref = self.list.GetItemText(idx, 0)
            self.on_component_edit(self.current_sheet, ref)
        else:
            wx.MessageBox("Please select a component first.", "No Selection", wx.OK | wx.ICON_INFORMATION)

    def _on_dclick(self, event):
        if self.on_component_edit:
            ref = self.list.GetItemText(event.GetIndex(), 0)
            self.on_component_edit(self.current_sheet, ref)


class SettingsPanel(wx.Panel):
    """Mission profile settings."""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)

        self.on_change = None

        main = wx.BoxSizer(wx.VERTICAL)

        header = SectionHeader(self, "Mission Profile", "Used for converting λ ↔ R (hours, cycles, ΔT)")
        main.Add(header, 0, wx.ALL | wx.EXPAND, Spacing.LG)

        form = wx.FlexGridSizer(3, 3, Spacing.SM, Spacing.MD)
        form.AddGrowableCol(1, 1)

        form.Add(self._label("Mission duration"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.years = wx.SpinCtrl(self, min=1, max=30, initial=5, size=(90, -1))
        self.years.SetFont(Fonts.body())
        self.years.Bind(wx.EVT_SPINCTRL, self._on_change)
        form.Add(self.years, 0)
        form.Add(self._unit("years"), 0, wx.ALIGN_CENTER_VERTICAL)

        form.Add(self._label("Thermal cycles / year"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.cycles = wx.SpinCtrl(self, min=100, max=20000, initial=5256, size=(90, -1))
        self.cycles.SetFont(Fonts.body())
        self.cycles.Bind(wx.EVT_SPINCTRL, self._on_change)
        self.cycles.SetToolTip("Annual thermal cycles (example: 5256 ≈ LEO satellite)")
        form.Add(self.cycles, 0)
        form.Add(self._unit("cycles"), 0, wx.ALIGN_CENTER_VERTICAL)

        form.Add(self._label("ΔT per cycle"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.dt = wx.SpinCtrlDouble(self, min=0.5, max=30, initial=3.0, inc=0.5, size=(90, -1))
        self.dt.SetFont(Fonts.body())
        self.dt.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        self.dt.SetToolTip("Temperature swing per thermal cycle (°C)")
        form.Add(self.dt, 0)
        form.Add(self._unit("°C"), 0, wx.ALIGN_CENTER_VERTICAL)

        main.Add(form, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, Spacing.LG)

        help_btn = IconButton(self, "IEC 62380 Quick Reference", icon="help", size=(-1, 34))
        help_btn.Bind(wx.EVT_BUTTON, self._on_help)
        main.Add(help_btn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, Spacing.LG)

        self.SetSizer(main)

    def _label(self, text: str) -> wx.StaticText:
        lbl = wx.StaticText(self, label=text)
        lbl.SetFont(Fonts.body())
        lbl.SetForegroundColour(Colors.TEXT_SECONDARY)
        return lbl

    def _unit(self, text: str) -> wx.StaticText:
        lbl = wx.StaticText(self, label=text)
        lbl.SetFont(Fonts.body())
        lbl.SetForegroundColour(Colors.TEXT_SECONDARY)
        return lbl

    def get_hours(self) -> float:
        return float(self.years.GetValue()) * 365.0 * 24.0

    def get_cycles(self) -> int:
        return int(self.cycles.GetValue())

    def get_dt(self) -> float:
        return float(self.dt.GetValue())

    def _on_change(self, event):
        if self.on_change:
            self.on_change()

    def _on_help(self, event):
        dlg = QuickReferenceDialog(self)
        dlg.ShowModal()
        dlg.Destroy()


# =============================================================================
# Main dialog
# =============================================================================

class ReliabilityMainDialog(BaseDialog):
    """Main reliability calculator dialog."""

    def __init__(self, parent, project_path: str = None):
        # Get screen size for better default sizing
        display = wx.Display(0)
        screen_rect = display.GetClientArea()
        # Use 90% of screen size for better visibility
        default_w = min(1700, int(screen_rect.Width * 0.90))
        default_h = min(1050, int(screen_rect.Height * 0.90))

        super().__init__(
            parent,
            title="Reliability Calculator (IEC TR 62380)",
            size=(default_w, default_h),
            min_size=(1200, 850),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX,
        )

        self.project_path = project_path
        self.parser: Optional[SchematicParser] = None
        self.sheet_data: Dict[str, Dict] = {}
        self.component_edits: Dict[str, Dict[str, Dict]] = {}

        self._create_ui()
        self._bind_events()

        if project_path:
            self._load_project(project_path)
        else:
            self._load_test_data()

    # ---------------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------------

    def _create_ui(self):
        root = wx.BoxSizer(wx.VERTICAL)

        # ─────────────────────────────────────────────────────────────────
        # Header bar
        # ─────────────────────────────────────────────────────────────────
        header = wx.Panel(self)
        header.SetBackgroundColour(Colors.HEADER_BG)
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)

        title_sizer = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(header, label="Reliability Calculator")
        title.SetFont(Fonts.header())
        title.SetForegroundColour(Colors.HEADER_FG)
        title_sizer.Add(title, 0, wx.BOTTOM, Spacing.XS)

        subtitle = wx.StaticText(header, label="IEC TR 62380 • Block diagram + per-sheet component model")
        subtitle.SetFont(Fonts.body())
        subtitle.SetForegroundColour(wx.Colour(176, 190, 197))
        title_sizer.Add(subtitle, 0)

        header_sizer.Add(title_sizer, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.LG)

        self.project_badge = wx.StaticText(header, label="(no project)")
        self.project_badge.SetFont(Fonts.body())
        self.project_badge.SetForegroundColour(wx.Colour(176, 190, 197))
        header_sizer.Add(self.project_badge, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, Spacing.LG)

        header.SetSizer(header_sizer)
        root.Add(header, 0, wx.EXPAND)

        # ─────────────────────────────────────────────────────────────────
        # Content
        # ─────────────────────────────────────────────────────────────────
        content = wx.Panel(self)
        content.SetBackgroundColour(Colors.PANEL_BG)
        content_sizer = wx.BoxSizer(wx.VERTICAL)

        info = InfoBanner(
            content,
            "Tip: Add one or more sheets on the left. On the right, drag blocks to organize the system. "
            "Right‑click a selection to group as Series / Parallel / K‑of‑N.",
            style="info",
        )
        content_sizer.Add(info, 0, wx.ALL | wx.EXPAND, Spacing.LG)

        toolbar = self._create_toolbar(content)
        content_sizer.Add(toolbar, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, Spacing.LG)

        splitter = wx.SplitterWindow(content, style=wx.SP_LIVE_UPDATE)
        splitter.SetMinimumPaneSize(240)

        # Left side: sheets + settings
        left = wx.Panel(splitter)
        left.SetBackgroundColour(Colors.PANEL_BG)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        self.sheet_panel = SheetPanel(left)
        left_sizer.Add(self.sheet_panel, 1, wx.EXPAND | wx.BOTTOM, Spacing.LG)

        self.settings_panel = SettingsPanel(left)
        left_sizer.Add(self.settings_panel, 0, wx.EXPAND)

        left.SetSizer(left_sizer)

        # Right side: editor + bottom split (components/results)
        right = wx.SplitterWindow(splitter, style=wx.SP_LIVE_UPDATE)
        right.SetMinimumPaneSize(320)

        # Editor "card"
        editor_panel = wx.Panel(right)
        editor_panel.SetBackgroundColour(Colors.PANEL_BG)
        editor_sizer = wx.BoxSizer(wx.VERTICAL)

        editor_header = SectionHeader(
            editor_panel,
            "System Block Diagram",
            "Drag to pan. Shift+drag to select & group. Scroll to zoom. F to fit all.",
        )
        editor_sizer.Add(editor_header, 0, wx.ALL | wx.EXPAND, Spacing.LG)

        self.editor = BlockEditor(editor_panel)
        editor_sizer.Add(self.editor, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, Spacing.LG)

        editor_panel.SetSizer(editor_sizer)

        bottom = wx.SplitterWindow(right, style=wx.SP_LIVE_UPDATE)
        bottom.SetMinimumPaneSize(250)

        self.comp_panel = ComponentPanel(bottom)

        results_panel = wx.Panel(bottom)
        results_panel.SetBackgroundColour(Colors.PANEL_BG)
        results_sizer = wx.BoxSizer(wx.VERTICAL)

        results_header = SectionHeader(results_panel, "System Results", "Summary + per-block details")
        results_sizer.Add(results_header, 0, wx.ALL | wx.EXPAND, Spacing.LG)

        self.results = wx.TextCtrl(results_panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.results.SetFont(Fonts.mono())
        self.results.SetBackgroundColour(wx.Colour(250, 250, 250))
        results_sizer.Add(self.results, 1, wx.LEFT | wx.RIGHT | wx.EXPAND, Spacing.LG)

        btn_calc = IconButton(results_panel, "Recalculate", icon="calc", size=(-1, 36))
        btn_calc.SetToolTip("Recompute system reliability from the current block structure and mission profile")
        btn_calc.Bind(wx.EVT_BUTTON, self._on_calculate)
        results_sizer.Add(btn_calc, 0, wx.ALL | wx.EXPAND, Spacing.LG)

        results_panel.SetSizer(results_sizer)

        bottom.SplitVertically(self.comp_panel, results_panel, 470)
        right.SplitHorizontally(editor_panel, bottom, 420)
        splitter.SplitVertically(left, right, 320)

        content_sizer.Add(splitter, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, Spacing.LG)

        content.SetSizer(content_sizer)
        root.Add(content, 1, wx.EXPAND)

        # ─────────────────────────────────────────────────────────────────
        # Footer
        # ─────────────────────────────────────────────────────────────────
        footer = wx.Panel(self)
        footer.SetBackgroundColour(Colors.BACKGROUND)
        footer_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.status = StatusIndicator(footer)
        footer_sizer.Add(self.status, 1, wx.ALL | wx.EXPAND, Spacing.XS)

        btn_close = wx.Button(footer, label="Close", size=(90, 34))
        btn_close.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CANCEL))
        footer_sizer.Add(btn_close, 0, wx.ALL, Spacing.SM)

        footer.SetSizer(footer_sizer)
        root.Add(footer, 0, wx.EXPAND)

        self.SetSizer(root)

    def _create_toolbar(self, parent: wx.Window) -> wx.Panel:
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(Colors.PANEL_BG)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        lbl = wx.StaticText(panel, label="Project")
        lbl.SetFont(Fonts.body())
        lbl.SetForegroundColour(Colors.TEXT_SECONDARY)
        sizer.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, Spacing.SM)

        self.txt_project = wx.TextCtrl(panel, value="(none)", style=wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.txt_project.SetFont(Fonts.body())
        sizer.Add(self.txt_project, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, Spacing.LG)

        btn_open = IconButton(panel, "Open…", icon="folder", size=(110, 34))
        btn_open.Bind(wx.EVT_BUTTON, self._on_open)
        sizer.Add(btn_open, 0, wx.RIGHT, Spacing.SM)

        btn_save = IconButton(panel, "Save Config", icon="save", size=(130, 34))
        btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        sizer.Add(btn_save, 0, wx.RIGHT, Spacing.SM)

        btn_load_cfg = IconButton(panel, "Load Config", icon="load", size=(130, 34))
        btn_load_cfg.Bind(wx.EVT_BUTTON, self._on_load_config)
        sizer.Add(btn_load_cfg, 0, wx.RIGHT, Spacing.LG)

        btn_batch = IconButton(panel, "Batch Edit", icon="edit", size=(120, 34))
        btn_batch.Bind(wx.EVT_BUTTON, self._on_batch_edit)
        btn_batch.SetToolTip("Edit reliability fields for all components in the project")
        sizer.Add(btn_batch, 0, wx.RIGHT, Spacing.SM)

        btn_export = IconButton(panel, "Export", icon="export", size=(100, 34))
        btn_export.Bind(wx.EVT_BUTTON, self._on_export)
        sizer.Add(btn_export, 0)

        panel.SetSizer(sizer)
        return panel

    def _bind_events(self):
        self.editor.on_block_activate = self._on_block_activate
        self.sheet_panel.on_add = self._add_sheet
        self.sheet_panel.on_edit = self._edit_sheet_components
        self.editor.on_selection_change = self._on_block_select
        self.editor.on_structure_change = self._on_calculate
        self.settings_panel.on_change = self._recalculate_all
        self.comp_panel.on_component_edit = self._edit_single_component

    # ---------------------------------------------------------------------
    # Data loading / parsing
    # ---------------------------------------------------------------------

    def _load_project(self, path: str):
        self.project_path = path
        self.txt_project.SetValue(path)
        self.project_badge.SetLabel(Path(path).name if path else "(no project)")

        self.parser = SchematicParser(path)
        if self.parser.parse():
            sheets = self.parser.get_sheet_paths()
            self.sheet_panel.set_sheets(sheets)
            self._calculate_sheets()
            self.status.set_status(f"Loaded {len(sheets)} sheet(s) from {path}", "ok")
        else:
            wx.MessageBox(
                f"Could not parse schematics in:\n{path}",
                "Parse Error",
                wx.OK | wx.ICON_WARNING,
            )
            self.status.set_status("Parse failed", "warning")

    def _load_test_data(self):
        sheets = [
            "/Project Architecture/",
            "/Project Architecture/Power/",
            "/Project Architecture/Power/Protection Satellite 24V/",
            "/Project Architecture/Power/Battery Charger/",
            "/Project Architecture/Power/LDO_3v3_sat/",
            "/Project Architecture/Power/System On Logic/",
            "/Project Architecture/Control/MCU_A/",
            "/Project Architecture/Trigger IDD/",
        ]

        self.parser = create_test_data(sheets)
        self.sheet_panel.set_sheets(sheets)
        self.txt_project.SetValue("Test Data")
        self.project_badge.SetLabel("Test Data")
        self._calculate_sheets()
        self.status.set_status("Loaded test data", "ok")

    # ---------------------------------------------------------------------
    # Core computation glue
    # ---------------------------------------------------------------------

    def _calculate_sheets(self):
        if not self.parser:
            return

        hours = self.settings_panel.get_hours()
        cycles = self.settings_panel.get_cycles()
        dt = self.settings_panel.get_dt()

        for path in self.parser.get_sheet_paths():
            components = self.parser.get_sheet_components(path)

            comp_data = []
            total_lam = 0.0

            for c in components:
                edited = self.component_edits.get(path, {}).get(c.reference, {})

                if edited:
                    comp_type = edited.get("_component_type", "Resistor")
                    params = edited.copy()
                    params["n_cycles"] = cycles
                    params["delta_t"] = dt
                    result = calculate_component_lambda(comp_type, params)
                    lam = float(result.get("lambda_total", 0) or 0)
                    cls_name = comp_type
                else:
                    cls = c.get_field("Reliability_Class", c.get_field("Class", ""))
                    if not cls:
                        cls = classify_component(c.reference, c.value, {})

                    params = {
                        "n_cycles": cycles,
                        "delta_t": dt,
                        "t_ambient": c.get_float("T_Ambient", 25),
                        "t_junction": c.get_float("T_Junction", 85),
                        "operating_power": c.get_float("Operating_Power", 0.01),
                        "rated_power": c.get_float("Rated_Power", 0.125),
                    }

                    lam = float(calculate_lambda(cls or "Resistor", params) or 0)
                    cls_name = cls or "Unknown"

                r = reliability_from_lambda(lam, hours)
                total_lam += lam

                comp_data.append(
                    {
                        "ref": c.reference,
                        "value": c.value,
                        "class": cls_name,
                        "lambda": lam,
                        "r": r,
                    }
                )

            sheet_r = reliability_from_lambda(total_lam, hours)

            self.sheet_data[path] = {
                "components": comp_data,
                "lambda": total_lam,
                "r": sheet_r,
            }

    def _recalculate_all(self):
        self.status.set_status("Recalculating…", "working")
        self._calculate_sheets()

        for _, b in self.editor.blocks.items():
            if not b.is_group:
                data = self.sheet_data.get(b.name, {})
                b.reliability = float(data.get("r", 1.0) or 1.0)
                b.lambda_val = float(data.get("lambda", 0.0) or 0.0)

        self._on_calculate(None)

    def _calculate_system(self) -> Tuple[float, float]:
        hours = self.settings_panel.get_hours()

        def calc(block_id: str) -> float:
            b = self.editor.blocks.get(block_id)
            if not b:
                return 1.0

            if b.is_group:
                child_rs = [calc(cid) for cid in b.children]

                if b.connection_type == "series":
                    r = r_series(child_rs)
                elif b.connection_type == "parallel":
                    r = r_parallel(child_rs)
                else:
                    r = r_k_of_n(child_rs, b.k_value)

                b.reliability = float(r)
                b.lambda_val = float(lambda_from_reliability(r, hours))
                return b.reliability

            data = self.sheet_data.get(b.name, {})
            b.reliability = float(data.get("r", 1.0) or 1.0)
            b.lambda_val = float(data.get("lambda", 0.0) or 0.0)
            return b.reliability

        if not self.editor.root_id:
            return 1.0, 0.0

        sys_r = calc(self.editor.root_id)
        sys_lam = float(lambda_from_reliability(sys_r, hours))

        self.editor.Refresh()
        return sys_r, sys_lam

    # ---------------------------------------------------------------------
    # UI events
    # ---------------------------------------------------------------------

    def _add_sheet(self, path: str):
        for b in self.editor.blocks.values():
            if b.name == path:
                return

        label = path.rstrip("/").split("/")[-1] or "Root"
        block = self.editor.add_block(f"sheet_{len(self.editor.blocks)}", path, label)

        data = self.sheet_data.get(path, {})
        block.reliability = float(data.get("r", 1.0) or 1.0)
        block.lambda_val = float(data.get("lambda", 0.0) or 0.0)

        self.editor.Refresh()
        self.status.set_status(f"Added sheet: {label}", "ok")

    def _on_block_select(self, block_id: Optional[str]):
        if block_id:
            b = self.editor.blocks.get(block_id)
            if b and not b.is_group:
                data = self.sheet_data.get(b.name, {})
                self.comp_panel.set_data(
                    b.name,
                    data.get("components", []),
                    float(data.get("lambda", 0) or 0),
                    float(data.get("r", 1) or 1),
                )

    def _on_block_activate(self, block_id: str, sheet_path: str):
        """Handle double-click on a sheet block - open component editor."""
        components = self.parser.get_sheet_components(sheet_path) if self.parser else []

        if not components:
            wx.MessageBox(f"No components found in {sheet_path}", "Info", wx.ICON_INFORMATION)
            return

        # Convert to ComponentData for the batch editor
        comp_data_list: List[ComponentData] = []
        for comp in components:
            comp_type = classify_component(comp.reference, comp.value, comp.fields)
            comp_data_list.append(
                ComponentData(
                    reference=comp.reference,
                    value=comp.value,
                    component_type=comp_type,
                    fields=dict(comp.fields),
                )
            )

        dlg = BatchComponentEditorDialog(self, comp_data_list, sheet_path)
        if dlg.ShowModal() == wx.ID_OK:
            # Update fields back to parser components
            for cd in dlg.components:
                for comp in components:
                    if comp.reference == cd.reference:
                        comp.fields.update(cd.fields)
                        break

            # Recalculate and refresh
            self._recalculate_sheet(sheet_path)
            self.status.set_status(f"Updated {len(components)} component(s) in {sheet_path}", "ok")

        dlg.Destroy()

    def _on_calculate(self, event):
        self.status.set_status("Calculating…", "working")

        sys_r, sys_lam = self._calculate_system()
        hours = self.settings_panel.get_hours()
        years = hours / (365 * 24)
        sys_fit = sys_lam * 1e9

        lines = [
            "═" * 52,
            "            SYSTEM RELIABILITY ANALYSIS",
            "═" * 52,
            "",
            f"  Mission Duration: {years:.2f} years ({hours:.0f} h)",
            "",
            f"  ► System Reliability:  R = {sys_r:.6f}",
            f"  ► Failure Rate:        λ = {sys_fit:.2f} FIT",
            f"                           = {sys_lam:.2e} /h",
        ]

        if sys_lam > 0:
            mttf = 1 / sys_lam
            lines.append(f"  ► MTTF:                {mttf:.2e} hours")
            lines.append(f"                         ({mttf/(365*24):.2f} years)")

        lines.extend(
            [
                "",
                "═" * 52,
                "                 BLOCK DETAILS",
                "═" * 52,
            ]
        )

        for bid, b in sorted(self.editor.blocks.items()):
            if bid.startswith("__"):
                continue

            if b.is_group:
                lines.append(f"\n  [{b.label}] ({len(b.children)} blocks)")
                lines.append(f"    R = {float(getattr(b, 'reliability', 1.0) or 1.0):.6f}")
            else:
                fit = float(getattr(b, "lambda_val", 0.0) or 0.0) * 1e9
                lines.append(f"\n  {b.label}")
                lines.append(
                    f"    λ = {fit:.2f} FIT, R = {float(getattr(b, 'reliability', 1.0) or 1.0):.6f}"
                )

        self.results.SetValue("\n".join(lines))
        self.status.set_status(f"System R = {sys_r:.6f} ({years:.1f}y mission)", "ok")

    def _edit_single_component(self, sheet_path: str, ref: str):
        if not self.parser or not sheet_path:
            return

        components = self.parser.get_sheet_components(sheet_path)
        comp = None
        for c in components:
            if c.reference == ref:
                comp = c
                break
        if not comp:
            return

        edited = self.component_edits.get(sheet_path, {}).get(ref, {})

        if edited:
            comp_type = edited.get("_component_type", "Resistor")
            fields = dict(edited)
        else:
            comp_type = classify_component(comp.reference, comp.value, {})
            fields = {}

        comp_data = ComponentData(
            reference=ref,
            value=comp.value,
            component_type=comp_type,
            fields=fields,
        )

        dlg = ComponentEditorDialog(self, comp_data, self.settings_panel.get_hours())
        if dlg.ShowModal() == wx.ID_OK:
            result = dlg.get_result()
            if result:
                if sheet_path not in self.component_edits:
                    self.component_edits[sheet_path] = {}
                self.component_edits[sheet_path][ref] = result

                self._recalculate_all()

                data = self.sheet_data.get(sheet_path, {})
                self.comp_panel.set_data(
                    sheet_path,
                    data.get("components", []),
                    float(data.get("lambda", 0) or 0),
                    float(data.get("r", 1) or 1),
                )
                self.status.set_status(f"Updated {ref}", "ok")
        dlg.Destroy()

    def _edit_sheet_components(self, sheets: List[str]):
        if not self.parser:
            return

        all_components: List[ComponentData] = []
        for sheet in sheets:
            components = self.parser.get_sheet_components(sheet)
            for c in components:
                edited = self.component_edits.get(sheet, {}).get(c.reference, {})

                if edited:
                    comp_type = edited.get("_component_type", "Resistor")
                    fields = dict(edited)
                else:
                    comp_type = classify_component(c.reference, c.value, {})
                    fields = {}

                all_components.append(
                    ComponentData(
                        reference=c.reference,
                        value=c.value,
                        component_type=comp_type,
                        fields=fields,
                    )
                )

        if not all_components:
            wx.MessageBox("No components found.", "No Components", wx.OK | wx.ICON_INFORMATION)
            return

        dlg = BatchComponentEditorDialog(self, all_components, self.settings_panel.get_hours())
        if dlg.ShowModal() == wx.ID_OK:
            results = dlg.get_results()

            for sheet in sheets:
                components = self.parser.get_sheet_components(sheet)
                for c in components:
                    if c.reference in results:
                        if sheet not in self.component_edits:
                            self.component_edits[sheet] = {}
                        self.component_edits[sheet][c.reference] = results[c.reference]

            self._recalculate_all()
            self.status.set_status(f"Batch updated {len(results)} component(s)", "ok")
        dlg.Destroy()

    def _on_batch_edit(self, event):
        if not self.parser:
            wx.MessageBox("No project loaded.", "No Project", wx.OK | wx.ICON_INFORMATION)
            return
        sheets = self.parser.get_sheet_paths()
        self._edit_sheet_components(sheets)

    def _on_open(self, event):
        dlg = wx.DirDialog(
            self,
            "Select KiCad Project Directory",
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.editor.clear()
            self.sheet_data.clear()
            self.component_edits.clear()
            self._load_project(dlg.GetPath())
        dlg.Destroy()

    def _on_save(self, event):
        default_dir = self.project_path or os.getcwd()
        dlg = wx.FileDialog(
            self,
            "Save Configuration",
            defaultDir=default_dir,
            defaultFile="reliability_config.json",
            wildcard="JSON (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            config = {
                "project": self.project_path,
                "structure": self.editor.get_structure(),
                "settings": {
                    "years": self.settings_panel.years.GetValue(),
                    "cycles": self.settings_panel.cycles.GetValue(),
                    "dt": self.settings_panel.dt.GetValue(),
                },
                "component_edits": self.component_edits,
            }
            with open(dlg.GetPath(), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            self.status.set_status(f"Saved config: {dlg.GetPath()}", "ok")
        dlg.Destroy()

    def _on_load_config(self, event):
        dlg = wx.FileDialog(
            self,
            "Load Configuration",
            wildcard="JSON (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dlg.ShowModal() == wx.ID_OK:
            try:
                with open(dlg.GetPath(), "r", encoding="utf-8") as f:
                    config = json.load(f)

                settings = config.get("settings", {})
                self.settings_panel.years.SetValue(settings.get("years", 5))
                self.settings_panel.cycles.SetValue(settings.get("cycles", 5256))
                self.settings_panel.dt.SetValue(settings.get("dt", 3.0))
                self.component_edits = config.get("component_edits", {})
                self.editor.load_structure(config.get("structure", {}))
                self._recalculate_all()
                self.status.set_status(f"Loaded config: {dlg.GetPath()}", "ok")
            except Exception as e:
                wx.MessageBox(f"Error: {e}", "Load Error", wx.OK | wx.ICON_ERROR)
                self.status.set_status("Load failed", "error")
        dlg.Destroy()

    def _on_export(self, event):
        dlg = wx.FileDialog(
            self,
            "Export Report",
            wildcard="HTML (*.html)|*.html|Markdown (*.md)|*.md|CSV (*.csv)|*.csv",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            idx = dlg.GetFilterIndex()

            sys_r, sys_lam = self._calculate_system()
            hours = self.settings_panel.get_hours()

            if idx == 0:
                content = self._generate_html(sys_r, sys_lam, hours)
            elif idx == 1:
                content = self._generate_md(sys_r, sys_lam, hours)
            else:
                content = self._generate_csv()

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.status.set_status(f"Exported report: {path}", "ok")
        dlg.Destroy()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _recalculate_sheet(self, sheet_path: str):
        """Recalculate a single sheet and refresh any block using it."""
        if not self.parser:
            return

        hours = self.settings_panel.get_hours()
        cycles = self.settings_panel.get_cycles()
        dt = self.settings_panel.get_dt()

        components = self.parser.get_sheet_components(sheet_path)

        comp_data = []
        total_lam = 0.0

        for c in components:
            edited = self.component_edits.get(sheet_path, {}).get(c.reference, {})

            if edited:
                comp_type = edited.get("_component_type", "Resistor")
                params = edited.copy()
                params["n_cycles"] = cycles
                params["delta_t"] = dt
                result = calculate_component_lambda(comp_type, params)
                lam = float(result.get("lambda_total", 0) or 0)
                cls_name = comp_type
            else:
                cls = c.get_field("Reliability_Class", c.get_field("Class", ""))
                if not cls:
                    cls = classify_component(c.reference, c.value, {})

                params = {
                    "n_cycles": cycles,
                    "delta_t": dt,
                    "t_ambient": c.get_float("T_Ambient", 25),
                    "t_junction": c.get_float("T_Junction", 85),
                    "operating_power": c.get_float("Operating_Power", 0.01),
                    "rated_power": c.get_float("Rated_Power", 0.125),
                }

                lam = float(calculate_lambda(cls or "Resistor", params) or 0)
                cls_name = cls or "Unknown"

            r = reliability_from_lambda(lam, hours)
            total_lam += lam

            comp_data.append(
                {
                    "ref": c.reference,
                    "value": c.value,
                    "class": cls_name,
                    "lambda": lam,
                    "r": r,
                }
            )

        sheet_r = reliability_from_lambda(total_lam, hours)
        self.sheet_data[sheet_path] = {"components": comp_data, "lambda": total_lam, "r": sheet_r}

        for _, b in self.editor.blocks.items():
            if not b.is_group and b.name == sheet_path:
                b.reliability = float(sheet_r)
                b.lambda_val = float(total_lam)

        # If that sheet is currently displayed, refresh the component panel.
        sel = getattr(self.editor, "selected_id", None)
        if sel and sel in self.editor.blocks:
            b = self.editor.blocks[sel]
            if not b.is_group and b.name == sheet_path:
                self.comp_panel.set_data(sheet_path, comp_data, total_lam, sheet_r)

        self.editor.Refresh()

    def _generate_html(self, sys_r: float, sys_lam: float, hours: float) -> str:
        years = hours / (365 * 24)
        sys_fit = sys_lam * 1e9
        html = f"""<!DOCTYPE html>
<html><head><title>Reliability Report - IEC TR 62380</title>
<style>
body {{ font-family: Arial; margin: 20px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; }}
th {{ background: #f5f5f5; }}
.summary {{ background: #e8f4e8; padding: 15px; border-radius: 5px; margin: 15px 0; }}
</style></head><body>
<h1>Reliability Analysis Report</h1>
<p><i>Based on IEC TR 62380</i></p>
<div class="summary">
<h2>System Summary</h2>
<p><b>Mission:</b> {years:.2f} years</p>
<p><b>Reliability:</b> R = {sys_r:.6f}</p>
<p><b>Failure Rate:</b> λ = {sys_fit:.2f} FIT</p>
</div>
<h2>Sheet Analysis</h2>
"""
        for path, data in sorted(self.sheet_data.items()):
            fit = float(data["lambda"]) * 1e9
            html += f"""<h3>{path}</h3>
<p>R = {float(data["r"]):.6f}, λ = {fit:.2f} FIT</p>
<table><tr><th>Ref</th><th>Value</th><th>Type</th><th>λ (FIT)</th><th>R</th></tr>
"""
            for c in data["components"]:
                c_fit = float(c["lambda"]) * 1e9
                html += f'<tr><td>{c["ref"]}</td><td>{c["value"]}</td><td>{c["class"]}</td>'
                html += f"<td>{c_fit:.2f}</td><td>{float(c['r']):.6f}</td></tr>\n"
            html += "</table>\n"
        html += "</body></html>"
        return html

    def _generate_md(self, sys_r: float, sys_lam: float, hours: float) -> str:
        years = hours / (365 * 24)
        sys_fit = sys_lam * 1e9
        md = f"""# Reliability Analysis Report

*Based on IEC TR 62380*

## System Summary

- **Mission:** {years:.2f} years
- **Reliability:** R = {sys_r:.6f}
- **Failure Rate:** λ = {sys_fit:.2f} FIT

## Sheet Analysis

"""
        for path, data in sorted(self.sheet_data.items()):
            fit = float(data["lambda"]) * 1e9
            md += f"""### {path}

R = {float(data["r"]):.6f}, λ = {fit:.2f} FIT

| Ref | Value | Type | λ (FIT) | R |
|-----|-------|------|---------|---|
"""
            for c in data["components"]:
                c_fit = float(c["lambda"]) * 1e9
                md += f'| {c["ref"]} | {c["value"]} | {c["class"]} | {c_fit:.2f} | {float(c["r"]):.6f} |\n'
            md += "\n"
        return md

    def _generate_csv(self) -> str:
        lines = ["Sheet,Reference,Value,Type,Lambda_FIT,Reliability"]
        for path, data in sorted(self.sheet_data.items()):
            for c in data["components"]:
                c_fit = float(c["lambda"]) * 1e9
                lines.append(
                    f'"{path}","{c["ref"]}","{c["value"]}","{c["class"]}",{c_fit:.2f},{float(c["r"]):.6f}'
                )
        return "\n".join(lines)


if __name__ == "__main__":
    app = wx.App()
    dlg = ReliabilityMainDialog(None)
    dlg.ShowModal()
    dlg.Destroy()
