"""
Main Reliability Calculator Dialog
===================================
Primary UI integrating all IEC TR 62380 features with block diagram editor.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wx
import wx.lib.scrolledpanel as scrolled

from .block_editor import BlockEditor, Block
from .reliability_math import (
    calculate_component_lambda, reliability_from_lambda, lambda_from_reliability,
    r_series, r_parallel, r_k_of_n, calculate_lambda, get_component_types
)
from .component_editor import (
    ComponentEditorDialog, BatchComponentEditorDialog, ComponentData,
    classify_component, QuickReferenceDialog
)
from .schematic_parser import SchematicParser, create_test_data
from .project_manager import ProjectManager
from .report_generator import ReportGenerator, ReportData


# Design system colors
class Colors:
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


class SheetPanel(wx.Panel):
    """Panel listing schematic sheets."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.sheets = []
        self.on_add = None
        self.on_edit = None
        
        main = wx.BoxSizer(wx.VERTICAL)
        
        lbl = wx.StaticText(self, label="Schematic Sheets")
        lbl.SetFont(lbl.GetFont().Bold())
        main.Add(lbl, 0, wx.ALL, 10)
        
        self.list = wx.ListBox(self, style=wx.LB_EXTENDED)
        self.list.Bind(wx.EVT_LISTBOX_DCLICK, self._on_dclick)
        main.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_add = wx.Button(self, label="Add Selected")
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        btn_row.Add(self.btn_add, 1, wx.RIGHT, 5)
        self.btn_all = wx.Button(self, label="Add All")
        self.btn_all.Bind(wx.EVT_BUTTON, self._on_add_all)
        btn_row.Add(self.btn_all, 1)
        main.Add(btn_row, 0, wx.EXPAND | wx.ALL, 10)
        
        self.btn_edit = wx.Button(self, label="Edit Components...")
        self.btn_edit.Bind(wx.EVT_BUTTON, self._on_edit)
        main.Add(self.btn_edit, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        self.SetSizer(main)
    
    def set_sheets(self, sheets: List[str]):
        self.sheets = sheets
        self.list.Set(sheets)
    
    def _on_add(self, event):
        for i in self.list.GetSelections():
            if self.on_add: self.on_add(self.sheets[i])
    
    def _on_add_all(self, event):
        for s in self.sheets:
            if self.on_add: self.on_add(s)
    
    def _on_dclick(self, event):
        self._on_add(event)
    
    def _on_edit(self, event):
        selections = self.list.GetSelections()
        if selections and self.on_edit:
            self.on_edit([self.sheets[i] for i in selections])
        elif not selections:
            wx.MessageBox("Select a sheet first.", "No Selection", wx.OK | wx.ICON_INFORMATION)


class SettingsPanel(wx.Panel):
    """Mission profile settings with tau_on."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.on_change = None
        
        main = wx.BoxSizer(wx.VERTICAL)
        
        lbl = wx.StaticText(self, label="Mission Profile")
        lbl.SetFont(lbl.GetFont().Bold())
        main.Add(lbl, 0, wx.ALL, 10)
        
        form = wx.FlexGridSizer(4, 3, 8, 10)
        form.AddGrowableCol(1, 1)
        
        form.Add(wx.StaticText(self, label="Duration"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.years = wx.SpinCtrl(self, min=1, max=30, initial=5, size=(80, -1))
        self.years.Bind(wx.EVT_SPINCTRL, self._on_change)
        form.Add(self.years, 0)
        form.Add(wx.StaticText(self, label="years"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        form.Add(wx.StaticText(self, label="Thermal cycles"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.cycles = wx.SpinCtrl(self, min=100, max=20000, initial=5256, size=(80, -1))
        self.cycles.Bind(wx.EVT_SPINCTRL, self._on_change)
        self.cycles.SetToolTip("Annual thermal cycles (5256 â‰ˆ LEO satellite)")
        form.Add(self.cycles, 0)
        form.Add(wx.StaticText(self, label="/year"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        form.Add(wx.StaticText(self, label="Î”T per cycle"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.dt = wx.SpinCtrlDouble(self, min=0.5, max=50, initial=3.0, inc=0.5, size=(80, -1))
        self.dt.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        form.Add(self.dt, 0)
        form.Add(wx.StaticText(self, label="Â°C"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        form.Add(wx.StaticText(self, label="Default Ï„_on"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tau_on = wx.SpinCtrlDouble(self, min=0.01, max=1.0, initial=1.0, inc=0.05, size=(80, -1))
        self.tau_on.SetDigits(2)
        self.tau_on.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        self.tau_on.SetToolTip("Working time ratio (1.0 = continuous, 0.5 = 50% duty)")
        form.Add(self.tau_on, 0)
        form.Add(wx.StaticText(self, label="(0-1)"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        main.Add(form, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        
        help_btn = wx.Button(self, label="IEC 62380 Reference...")
        help_btn.Bind(wx.EVT_BUTTON, self._on_help)
        main.Add(help_btn, 0, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(main)
    
    def get_hours(self) -> float:
        return float(self.years.GetValue()) * 365.0 * 24.0
    
    def get_cycles(self) -> int:
        return int(self.cycles.GetValue())
    
    def get_dt(self) -> float:
        return float(self.dt.GetValue())
    
    def get_tau_on(self) -> float:
        return float(self.tau_on.GetValue())
    
    def _on_change(self, event):
        if self.on_change: self.on_change()
    
    def _on_help(self, event):
        dlg = QuickReferenceDialog(self)
        dlg.ShowModal()
        dlg.Destroy()


class ComponentPanel(scrolled.ScrolledPanel):
    """Component details panel."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.current_sheet = None
        self.on_component_edit = None
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        header = wx.BoxSizer(wx.HORIZONTAL)
        self.header = wx.StaticText(self, label="Components")
        self.header.SetFont(self.header.GetFont().Bold())
        header.Add(self.header, 1, wx.ALIGN_CENTER_VERTICAL)
        self.btn_edit = wx.Button(self, label="Edit", size=(70, -1))
        self.btn_edit.Bind(wx.EVT_BUTTON, self._on_edit)
        header.Add(self.btn_edit, 0)
        sizer.Add(header, 0, wx.EXPAND | wx.ALL, 10)
        
        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE)
        self.list.InsertColumn(0, "Ref", width=50)
        self.list.InsertColumn(1, "Value", width=80)
        self.list.InsertColumn(2, "Type", width=110)
        self.list.InsertColumn(3, "Î» (FIT)", width=70)
        self.list.InsertColumn(4, "R", width=70)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_dclick)
        sizer.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        
        self.summary = wx.StaticText(self, label="")
        self.summary.SetForegroundColour(Colors.TEXT_SECONDARY)
        sizer.Add(self.summary, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        self.SetupScrolling(scroll_x=False)
    
    def set_data(self, sheet: str, components: List[Dict], total_lam: float, r: float):
        self.current_sheet = sheet
        label = sheet.rstrip("/").split("/")[-1] or "Root"
        self.header.SetLabel(f"Components â€” {label}")
        
        self.list.DeleteAllItems()
        for i, c in enumerate(components):
            idx = self.list.InsertItem(i, c.get("ref", "?"))
            self.list.SetItem(idx, 1, (c.get("value", "") or "")[:20])
            self.list.SetItem(idx, 2, (c.get("class", "") or "")[:20])
            lam = float(c.get("lambda", 0) or 0)
            self.list.SetItem(idx, 3, f"{lam*1e9:.1f}")
            self.list.SetItem(idx, 4, f"{float(c.get('r', 1) or 1):.5f}")
        
        self.summary.SetLabel(f"Sheet: Î» = {total_lam*1e9:.1f} FIT â€¢ R = {r:.5f}")
        self.Layout()
    
    def _on_edit(self, event):
        idx = self.list.GetFirstSelected()
        if idx >= 0 and self.on_component_edit:
            ref = self.list.GetItemText(idx, 0)
            self.on_component_edit(self.current_sheet, ref)
    
    def _on_dclick(self, event):
        if self.on_component_edit:
            ref = self.list.GetItemText(event.GetIndex(), 0)
            self.on_component_edit(self.current_sheet, ref)


class ReliabilityMainDialog(wx.Dialog):
    """Main reliability calculator dialog."""
    
    def __init__(self, parent, project_path: str = None):
        display = wx.Display(0)
        rect = display.GetClientArea()
        w = min(1600, int(rect.Width * 0.9))
        h = min(1000, int(rect.Height * 0.9))
        
        super().__init__(parent, title="Reliability Calculator (IEC TR 62380)",
                        size=(w, h), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        self.SetMinSize((1100, 750))
        self.SetBackgroundColour(Colors.BACKGROUND)
        
        self.project_path = project_path
        self.project_manager: Optional[ProjectManager] = None
        self.parser: Optional[SchematicParser] = None
        self.sheet_data: Dict[str, Dict] = {}
        self.component_edits: Dict[str, Dict[str, Dict]] = {}
        
        # Initialize project manager if project path is provided
        if project_path:
            self.project_manager = ProjectManager(project_path)
            self.project_manager.ensure_reliability_folder()
        
        self._create_ui()
        self._bind_events()
        
        if project_path:
            self._load_project(project_path)
        else:
            self._load_test_data()
        
        self.Centre()
    
    def _create_ui(self):
        root = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        header = wx.Panel(self)
        header.SetBackgroundColour(Colors.HEADER_BG)
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        title = wx.StaticText(header, label="âš¡ Reliability Calculator")
        title.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        title.SetForegroundColour(Colors.HEADER_FG)
        header_sizer.Add(title, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)
        self.project_badge = wx.StaticText(header, label="(no project)")
        self.project_badge.SetForegroundColour(wx.Colour(176, 190, 197))
        header_sizer.Add(self.project_badge, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)
        header.SetSizer(header_sizer)
        root.Add(header, 0, wx.EXPAND)
        
        # Toolbar
        toolbar = self._create_toolbar()
        root.Add(toolbar, 0, wx.EXPAND | wx.ALL, 8)
        
        # Main content
        splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        splitter.SetMinimumPaneSize(220)
        
        # Left panel
        left = wx.Panel(splitter)
        left.SetBackgroundColour(Colors.PANEL_BG)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sheet_panel = SheetPanel(left)
        left_sizer.Add(self.sheet_panel, 1, wx.EXPAND | wx.BOTTOM, 8)
        self.settings_panel = SettingsPanel(left)
        left_sizer.Add(self.settings_panel, 0, wx.EXPAND)
        left.SetSizer(left_sizer)
        
        # Right panel
        right = wx.SplitterWindow(splitter, style=wx.SP_LIVE_UPDATE)
        right.SetMinimumPaneSize(280)
        
        # Editor panel
        editor_panel = wx.Panel(right)
        editor_panel.SetBackgroundColour(Colors.PANEL_BG)
        editor_sizer = wx.BoxSizer(wx.VERTICAL)
        editor_lbl = wx.StaticText(editor_panel, label="System Block Diagram")
        editor_lbl.SetFont(editor_lbl.GetFont().Bold())
        editor_sizer.Add(editor_lbl, 0, wx.ALL, 10)
        self.editor = BlockEditor(editor_panel)
        editor_sizer.Add(self.editor, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        editor_panel.SetSizer(editor_sizer)
        
        # Bottom panel
        bottom = wx.SplitterWindow(right, style=wx.SP_LIVE_UPDATE)
        bottom.SetMinimumPaneSize(220)
        
        self.comp_panel = ComponentPanel(bottom)
        
        results_panel = wx.Panel(bottom)
        results_panel.SetBackgroundColour(Colors.PANEL_BG)
        results_sizer = wx.BoxSizer(wx.VERTICAL)
        results_lbl = wx.StaticText(results_panel, label="System Results")
        results_lbl.SetFont(results_lbl.GetFont().Bold())
        results_sizer.Add(results_lbl, 0, wx.ALL, 10)
        self.results = wx.TextCtrl(results_panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.results.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.results.SetBackgroundColour(wx.Colour(250, 250, 250))
        results_sizer.Add(self.results, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        btn_calc = wx.Button(results_panel, label="Recalculate")
        btn_calc.Bind(wx.EVT_BUTTON, self._on_calculate)
        results_sizer.Add(btn_calc, 0, wx.EXPAND | wx.ALL, 10)
        results_panel.SetSizer(results_sizer)
        
        bottom.SplitVertically(self.comp_panel, results_panel, 420)
        right.SplitHorizontally(editor_panel, bottom, 380)
        splitter.SplitVertically(left, right, 280)
        
        root.Add(splitter, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        
        # Footer
        footer = wx.BoxSizer(wx.HORIZONTAL)
        self.status = wx.StaticText(self, label="Ready")
        self.status.SetForegroundColour(Colors.TEXT_SECONDARY)
        footer.Add(self.status, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        close_btn = wx.Button(self, label="Close", size=(80, -1))
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CANCEL))
        footer.Add(close_btn, 0, wx.ALL, 8)
        root.Add(footer, 0, wx.EXPAND)
        
        self.SetSizer(root)
    
    def _create_toolbar(self) -> wx.Panel:
        panel = wx.Panel(self)
        panel.SetBackgroundColour(Colors.PANEL_BG)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        sizer.Add(wx.StaticText(panel, label="Project:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.txt_project = wx.TextCtrl(panel, value="(none)", style=wx.TE_READONLY)
        sizer.Add(self.txt_project, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        
        btn_open = wx.Button(panel, label="Open...", size=(80, -1))
        btn_open.Bind(wx.EVT_BUTTON, self._on_open)
        sizer.Add(btn_open, 0, wx.RIGHT, 5)
        
        btn_save = wx.Button(panel, label="Save Config", size=(90, -1))
        btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        sizer.Add(btn_save, 0, wx.RIGHT, 5)
        
        btn_load = wx.Button(panel, label="Load Config", size=(90, -1))
        btn_load.Bind(wx.EVT_BUTTON, self._on_load_config)
        sizer.Add(btn_load, 0, wx.RIGHT, 15)
        
        btn_mc = wx.Button(panel, label="Analysis Suite", size=(110, -1))
        btn_mc.SetToolTip("Monte Carlo uncertainty and Sobol sensitivity analysis")
        btn_mc.Bind(wx.EVT_BUTTON, self._on_monte_carlo)
        sizer.Add(btn_mc, 0, wx.RIGHT, 5)
        
        btn_export = wx.Button(panel, label="Export Report", size=(100, -1))
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
    
    def _load_project(self, path: str):
        self.project_path = path
        self.txt_project.SetValue(path)
        self.project_badge.SetLabel(Path(path).name if path else "(no project)")
        
        self.parser = SchematicParser(path)
        if self.parser.parse():
            sheets = self.parser.get_sheet_paths()
            self.sheet_panel.set_sheets(sheets)
            self._calculate_sheets()
            self.status.SetLabel(f"Loaded {len(sheets)} sheet(s)")
        else:
            wx.MessageBox(f"Could not parse schematics in:\n{path}", "Parse Error", wx.OK | wx.ICON_WARNING)
    
    def _load_test_data(self):
        sheets = [
            "/Project/", "/Project/Power/", "/Project/Power/LDO_3v3/",
            "/Project/Power/Buck_5V/", "/Project/MCU/", "/Project/Sensors/"
        ]
        self.parser = create_test_data(sheets)
        self.sheet_panel.set_sheets(sheets)
        self.txt_project.SetValue("Test Data")
        self.project_badge.SetLabel("Test Data")
        self._calculate_sheets()
        self.status.SetLabel("Loaded test data")
    
    def _calculate_sheets(self):
        if not self.parser: return
        hours = self.settings_panel.get_hours()
        cycles = self.settings_panel.get_cycles()
        dt = self.settings_panel.get_dt()
        tau_on = self.settings_panel.get_tau_on()
        
        for path in self.parser.get_sheet_paths():
            components = self.parser.get_sheet_components(path)
            comp_data = []
            total_lam = 0.0
            
            for c in components:
                edited = self.component_edits.get(path, {}).get(c.reference, {})
                if edited:
                    ct = edited.get("_component_type", "Resistor")
                    params = edited.copy()
                    params.setdefault("n_cycles", cycles)
                    params.setdefault("delta_t", dt)
                    params.setdefault("tau_on", tau_on)
                    result = calculate_component_lambda(ct, params)
                    lam = float(result.get("lambda_total", 0) or 0)
                    cls_name = ct
                else:
                    cls = c.get_field("Reliability_Class", c.get_field("Class", ""))
                    if not cls: cls = classify_component(c.reference, c.value, {})
                    params = {
                        "n_cycles": cycles, "delta_t": dt, "tau_on": tau_on,
                        "t_ambient": c.get_float("T_Ambient", 25),
                        "t_junction": c.get_float("T_Junction", 85),
                        "operating_power": c.get_float("Operating_Power", 0.01),
                        "rated_power": c.get_float("Rated_Power", 0.125),
                    }
                    lam = float(calculate_lambda(cls or "Resistor", params) or 0)
                    cls_name = cls or "Unknown"
                
                r = reliability_from_lambda(lam, hours)
                total_lam += lam
                # Store params for Monte Carlo uncertainty analysis
                comp_data.append({
                    "ref": c.reference, "value": c.value, "class": cls_name, 
                    "lambda": lam, "r": r, "params": params
                })
            
            sheet_r = reliability_from_lambda(total_lam, hours)
            self.sheet_data[path] = {"components": comp_data, "lambda": total_lam, "r": sheet_r}
    
    def _recalculate_all(self):
        self.status.SetLabel("Recalculating...")
        self._calculate_sheets()
        for _, b in self.editor.blocks.items():
            if not b.is_group:
                data = self.sheet_data.get(b.name, {})
                b.reliability = float(data.get("r", 1.0) or 1.0)
                b.lambda_val = float(data.get("lambda", 0.0) or 0.0)
        self._on_calculate(None)
    
    def _calculate_system(self) -> Tuple[float, float]:
        hours = self.settings_panel.get_hours()
        
        def calc(bid: str) -> float:
            b = self.editor.blocks.get(bid)
            if not b: return 1.0
            if b.is_group:
                child_rs = [calc(cid) for cid in b.children]
                if b.connection_type == "series": r = r_series(child_rs)
                elif b.connection_type == "parallel": r = r_parallel(child_rs)
                else: r = r_k_of_n(child_rs, b.k_value)
                b.reliability = float(r)
                b.lambda_val = float(lambda_from_reliability(r, hours))
                return b.reliability
            data = self.sheet_data.get(b.name, {})
            b.reliability = float(data.get("r", 1.0) or 1.0)
            b.lambda_val = float(data.get("lambda", 0.0) or 0.0)
            return b.reliability
        
        if not self.editor.root_id: return 1.0, 0.0
        sys_r = calc(self.editor.root_id)
        sys_lam = float(lambda_from_reliability(sys_r, hours))
        self.editor.Refresh()
        return sys_r, sys_lam
    
    def _add_sheet(self, path: str):
        for b in self.editor.blocks.values():
            if b.name == path: return
        label = path.rstrip("/").split("/")[-1] or "Root"
        block = self.editor.add_block(f"sheet_{len(self.editor.blocks)}", path, label)
        data = self.sheet_data.get(path, {})
        block.reliability = float(data.get("r", 1.0) or 1.0)
        block.lambda_val = float(data.get("lambda", 0.0) or 0.0)
        self.editor.Refresh()
    
    def _on_block_select(self, block_id: Optional[str]):
        if block_id:
            b = self.editor.blocks.get(block_id)
            if b and not b.is_group:
                data = self.sheet_data.get(b.name, {})
                self.comp_panel.set_data(b.name, data.get("components", []),
                                         float(data.get("lambda", 0) or 0), float(data.get("r", 1) or 1))
    
    def _on_block_activate(self, block_id: str, sheet_path: str):
        components = self.parser.get_sheet_components(sheet_path) if self.parser else []
        if not components:
            wx.MessageBox(f"No components in {sheet_path}", "Info", wx.OK | wx.ICON_INFORMATION)
            return
        comp_list = [ComponentData(c.reference, c.value, classify_component(c.reference, c.value, c.fields), dict(c.fields)) for c in components]
        dlg = BatchComponentEditorDialog(self, comp_list, self.settings_panel.get_hours())
        if dlg.ShowModal() == wx.ID_OK:
            if sheet_path not in self.component_edits: self.component_edits[sheet_path] = {}
            for ref, fields in dlg.get_results().items():
                self.component_edits[sheet_path][ref] = fields
            self._recalculate_all()
        dlg.Destroy()
    
    def _on_calculate(self, event):
        sys_r, sys_lam = self._calculate_system()
        hours = self.settings_panel.get_hours()
        years = hours / (365 * 24)
        
        lines = [
            "=" * 45, "       SYSTEM RELIABILITY ANALYSIS", "=" * 45, "",
            f"  Mission: {years:.1f} years ({hours:.0f} h)", "",
            f"  â–¶ Reliability:  R = {sys_r:.6f}",
            f"  â–¶ Failure Rate: Î» = {sys_lam*1e9:.2f} FIT"
        ]
        if sys_lam > 0:
            mttf = 1 / sys_lam
            lines.append(f"  â–¶ MTTF: {mttf/(365*24):.1f} years")
        
        lines.extend(["", "=" * 45, "            BLOCK DETAILS", "=" * 45])
        for bid, b in sorted(self.editor.blocks.items()):
            if bid.startswith("__"): continue
            if b.is_group:
                lines.append(f"\n  [{b.label}] ({len(b.children)} blocks)")
                lines.append(f"    R = {float(b.reliability or 1.0):.6f}")
            else:
                lines.append(f"\n  {b.label}")
                lines.append(f"    Î» = {float(b.lambda_val or 0)*1e9:.1f} FIT, R = {float(b.reliability or 1.0):.6f}")
        
        self.results.SetValue("\n".join(lines))
        self.status.SetLabel(f"System R = {sys_r:.6f} ({years:.1f}y)")
    
    def _edit_single_component(self, sheet_path: str, ref: str):
        if not self.parser or not sheet_path: return
        components = self.parser.get_sheet_components(sheet_path)
        comp = None
        for c in components:
            if c.reference == ref:
                comp = c
                break
        if not comp: return
        
        edited = self.component_edits.get(sheet_path, {}).get(ref, {})
        if edited:
            ct = edited.get("_component_type", "Resistor")
            fields = dict(edited)
        else:
            ct = classify_component(comp.reference, comp.value, {})
            fields = {}
        
        comp_data = ComponentData(ref, comp.value, ct, fields)
        dlg = ComponentEditorDialog(self, comp_data, self.settings_panel.get_hours())
        if dlg.ShowModal() == wx.ID_OK:
            result = dlg.get_result()
            if result:
                if sheet_path not in self.component_edits: self.component_edits[sheet_path] = {}
                self.component_edits[sheet_path][ref] = result
                self._recalculate_all()
                data = self.sheet_data.get(sheet_path, {})
                self.comp_panel.set_data(sheet_path, data.get("components", []),
                                         float(data.get("lambda", 0) or 0), float(data.get("r", 1) or 1))
        dlg.Destroy()
    
    def _edit_sheet_components(self, sheets: List[str]):
        if not self.parser: return
        all_comps = []
        for sheet in sheets:
            components = self.parser.get_sheet_components(sheet)
            for c in components:
                edited = self.component_edits.get(sheet, {}).get(c.reference, {})
                if edited:
                    ct = edited.get("_component_type", "Resistor")
                    fields = dict(edited)
                else:
                    ct = classify_component(c.reference, c.value, {})
                    fields = {}
                all_comps.append(ComponentData(c.reference, c.value, ct, fields))
        
        if not all_comps:
            wx.MessageBox("No components found.", "No Components", wx.OK | wx.ICON_INFORMATION)
            return
        
        dlg = BatchComponentEditorDialog(self, all_comps, self.settings_panel.get_hours())
        if dlg.ShowModal() == wx.ID_OK:
            results = dlg.get_results()
            for sheet in sheets:
                components = self.parser.get_sheet_components(sheet)
                for c in components:
                    if c.reference in results:
                        if sheet not in self.component_edits: self.component_edits[sheet] = {}
                        self.component_edits[sheet][c.reference] = results[c.reference]
            self._recalculate_all()
        dlg.Destroy()
    
    def _on_open(self, event):
        dlg = wx.DirDialog(self, "Select KiCad Project Directory", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            self.editor.clear()
            self.sheet_data.clear()
            self.component_edits.clear()
            self._load_project(dlg.GetPath())
        dlg.Destroy()
    
    def _on_save(self, event):
        # Check if project manager is available
        default_path = "reliability_config.json"
        if self.project_manager:
            default_path = str(self.project_manager.get_config_path())
            initial_dir = str(self.project_manager.get_reliability_folder())
        else:
            initial_dir = ""
        
        dlg = wx.FileDialog(
            self, "Save Configuration",
            defaultFile=Path(default_path).name,
            defaultDir=initial_dir,
            wildcard="JSON (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        )
        if dlg.ShowModal() == wx.ID_OK:
            config = {
                "project": self.project_path,
                "structure": self.editor.get_structure(),
                "settings": {
                    "years": self.settings_panel.years.GetValue(),
                    "cycles": self.settings_panel.cycles.GetValue(),
                    "dt": self.settings_panel.dt.GetValue(),
                    "tau_on": self.settings_panel.tau_on.GetValue(),
                },
                "component_edits": self.component_edits,
            }
            with open(dlg.GetPath(), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            self.status.SetLabel(f"Saved: {dlg.GetPath()}")
        dlg.Destroy()
    
    def _on_load_config(self, event):
        # Check if project manager is available  
        if self.project_manager:
            initial_dir = str(self.project_manager.get_reliability_folder())
        else:
            initial_dir = ""
        
        dlg = wx.FileDialog(
            self, "Load Configuration",
            defaultDir=initial_dir,
            wildcard="JSON (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        )
        if dlg.ShowModal() == wx.ID_OK:
            try:
                with open(dlg.GetPath(), "r", encoding="utf-8") as f:
                    config = json.load(f)
                settings = config.get("settings", {})
                self.settings_panel.years.SetValue(settings.get("years", 5))
                self.settings_panel.cycles.SetValue(settings.get("cycles", 5256))
                self.settings_panel.dt.SetValue(settings.get("dt", 3.0))
                self.settings_panel.tau_on.SetValue(settings.get("tau_on", 1.0))
                self.component_edits = config.get("component_edits", {})
                self.editor.load_structure(config.get("structure", {}))
                self._recalculate_all()
                self.status.SetLabel(f"Loaded: {dlg.GetPath()}")
            except Exception as e:
                wx.MessageBox(f"Error: {e}", "Load Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()
    
    def _on_monte_carlo(self, event):
        """Open comprehensive analysis dialog with Monte Carlo and Sobol sensitivity."""
        try:
            from .analysis_dialog import AnalysisDialog
            
            sys_r, sys_lam = self._calculate_system()
            
            if sys_lam <= 0:
                wx.MessageBox("System failure rate is zero. Add components first.", 
                             "No Data", wx.OK | wx.ICON_WARNING)
                return
            
            dlg = AnalysisDialog(
                self,
                system_lambda=sys_lam,
                mission_hours=self.settings_panel.get_hours(),
                sheet_data=self.sheet_data,
                project_path=self.project_path,
                logo_path=str(self.project_manager.get_available_logo_path()) if self.project_manager and self.project_manager.logo_exists() else None,
                n_cycles=int(self.settings_panel.cycles.GetValue()),
                delta_t=float(self.settings_panel.dt.GetValue()),
                title="System Reliability Analysis"
            )
            dlg.ShowModal()
            dlg.Destroy()
            
        except ImportError as e:
            # Fallback to simple Monte Carlo if analysis_dialog not available
            try:
                from .monte_carlo import quick_monte_carlo
                sys_r, sys_lam = self._calculate_system()
                result = quick_monte_carlo(sys_lam, self.settings_panel.get_hours(), 
                                          uncertainty_percent=25.0, n_simulations=5000)
                
                msg = f"Monte Carlo Analysis (5000 simulations)\n\n"
                msg += f"Mean R: {result.mean:.6f}\n"
                msg += f"Std Dev: {result.std:.6f}\n"
                msg += f"5th percentile: {result.percentile_5:.6f}\n"
                msg += f"95th percentile: {result.percentile_95:.6f}\n"
                msg += f"Converged: {'Yes' if result.converged else 'No'}"
                wx.MessageBox(msg, "Monte Carlo Results", wx.OK | wx.ICON_INFORMATION)
            except Exception as e2:
                wx.MessageBox(f"Error: {e2}", "Monte Carlo Error", wx.OK | wx.ICON_ERROR)
        except Exception as e:
            wx.MessageBox(f"Error: {e}", "Analysis Error", wx.OK | wx.ICON_ERROR)
    
    def _on_export(self, event):
        # Determine initial directory and filename
        if self.project_manager:
            reports_folder = self.project_manager.get_reports_folder()
            initial_dir = str(reports_folder)
            default_name = f"reliability_report_{wx.DateTime.Now().Format('%Y%m%d_%H%M%S')}"
        else:
            initial_dir = ""
            default_name = "reliability_report"
        
        dlg = wx.FileDialog(
            self, "Export Report",
            defaultFile=default_name + ".html",
            defaultDir=initial_dir,
            wildcard="HTML (*.html)|*.html|Markdown (*.md)|*.md|CSV (*.csv)|*.csv|JSON (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
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
            elif idx == 2:
                content = self._generate_csv()
            else:
                content = self._generate_json(sys_r, sys_lam, hours)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.status.SetLabel(f"Exported: {path}")
        dlg.Destroy()
    
    def _generate_html(self, sys_r: float, sys_lam: float, hours: float) -> str:
        years = hours / (365 * 24)
        
        # Get logo path if available
        logo_path = None
        if self.project_manager:
            logo_path = self.project_manager.get_available_logo_path()
        
        # Create report generator with logo support
        generator = ReportGenerator(logo_path=str(logo_path) if logo_path else None)
        
        # Build report data
        report_data = ReportData(
            project_name=Path(self.project_path).name if self.project_path else "Reliability Report",
            mission_hours=hours,
            mission_years=years,
            n_cycles=int(self.settings_panel.cycles.GetValue()),
            delta_t=float(self.settings_panel.dt.GetValue()),
            system_reliability=sys_r,
            system_lambda=sys_lam,
            system_mttf_hours=1/sys_lam if sys_lam > 0 else float('inf'),
            sheets=self.sheet_data,
            blocks=[],
        )
        
        return generator.generate_html(report_data)
    
    def _generate_md(self, sys_r: float, sys_lam: float, hours: float) -> str:
        years = hours / (365 * 24)
        md = f"""# Reliability Analysis Report

*IEC TR 62380 Analysis*

## System Summary

- **Mission:** {years:.1f} years ({hours:.0f} hours)
- **Reliability:** R = {sys_r:.6f}
- **Failure Rate:** Î» = {sys_lam*1e9:.2f} FIT

## Sheet Analysis

"""
        for path, data in sorted(self.sheet_data.items()):
            fit = float(data["lambda"]) * 1e9
            md += f"""### {path}

R = {float(data["r"]):.6f}, Î» = {fit:.2f} FIT

| Ref | Value | Type | Î» (FIT) | R |
|-----|-------|------|---------|---|
"""
            for c in data["components"][:20]:
                c_fit = float(c["lambda"]) * 1e9
                md += f'| {c["ref"]} | {c["value"]} | {c["class"]} | {c_fit:.2f} | {float(c["r"]):.6f} |\n'
            md += "\n"
        return md
    
    def _generate_csv(self) -> str:
        lines = ["Sheet,Reference,Value,Type,Lambda_FIT,Reliability"]
        for path, data in sorted(self.sheet_data.items()):
            for c in data["components"]:
                c_fit = float(c["lambda"]) * 1e9
                lines.append(f'"{path}","{c["ref"]}","{c["value"]}","{c["class"]}",{c_fit:.2f},{float(c["r"]):.6f}')
        return "\n".join(lines)
    
    def _generate_json(self, sys_r: float, sys_lam: float, hours: float) -> str:
        return json.dumps({
            "system": {"reliability": sys_r, "lambda_fit": sys_lam*1e9, "mission_hours": hours},
            "sheets": self.sheet_data
        }, indent=2)


if __name__ == "__main__":
    app = wx.App()
    dlg = ReliabilityMainDialog(None)
    dlg.ShowModal()
    dlg.Destroy()
