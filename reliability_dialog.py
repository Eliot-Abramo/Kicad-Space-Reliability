"""
Main Reliability Calculator Dialog

The primary UI for the reliability calculator.
"""

import os
import json
import wx
import wx.lib.scrolledpanel as scrolled
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from block_editor import BlockEditor, Block
from reliability_core import (
    ConnectionType, calculate_lambda, reliability, lambda_from_reliability,
    r_series, r_parallel, r_k_of_n, COMPONENT_CLASSES
)
from schematic_parser import SchematicParser, create_test_data


class SheetPanel(wx.Panel):
    """Panel listing available sheets."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.sheets = []
        self.on_add = None
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        header = wx.StaticText(self, label="📋 Schematic Sheets")
        header.SetFont(header.GetFont().Bold())
        sizer.Add(header, 0, wx.ALL, 5)
        
        self.list = wx.ListBox(self, style=wx.LB_EXTENDED)
        self.list.Bind(wx.EVT_LISTBOX_DCLICK, self._on_dclick)
        sizer.Add(self.list, 1, wx.EXPAND | wx.ALL, 5)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.btn_add = wx.Button(self, label="Add Selected")
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        btn_sizer.Add(self.btn_add, 1, wx.RIGHT, 3)
        
        self.btn_all = wx.Button(self, label="Add All")
        self.btn_all.Bind(wx.EVT_BUTTON, self._on_add_all)
        btn_sizer.Add(self.btn_all, 1)
        
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(sizer)
    
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


class ComponentPanel(scrolled.ScrolledPanel):
    """Panel showing component details."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.header = wx.StaticText(self, label="📦 Components")
        self.header.SetFont(self.header.GetFont().Bold())
        self.sizer.Add(self.header, 0, wx.ALL, 5)
        
        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.list.InsertColumn(0, "Ref", width=50)
        self.list.InsertColumn(1, "Value", width=70)
        self.list.InsertColumn(2, "Class", width=120)
        self.list.InsertColumn(3, "λ", width=70)
        self.list.InsertColumn(4, "R", width=60)
        self.sizer.Add(self.list, 1, wx.EXPAND | wx.ALL, 5)
        
        self.summary = wx.StaticText(self, label="")
        self.summary.SetFont(self.summary.GetFont().Bold())
        self.sizer.Add(self.summary, 0, wx.ALL, 5)
        
        self.SetSizer(self.sizer)
        self.SetupScrolling()
    
    def set_data(self, sheet: str, components: List[Dict], total_lam: float, r: float):
        self.header.SetLabel(f"📦 {sheet}")
        
        self.list.DeleteAllItems()
        for i, c in enumerate(components):
            idx = self.list.InsertItem(i, c.get("ref", "?"))
            self.list.SetItem(idx, 1, c.get("value", "")[:10])
            self.list.SetItem(idx, 2, c.get("class", "")[:15])
            self.list.SetItem(idx, 3, f"{c.get('lambda', 0):.1e}")
            self.list.SetItem(idx, 4, f"{c.get('r', 1):.4f}")
        
        self.summary.SetLabel(f"Total: λ = {total_lam:.2e}, R = {r:.6f}")
        self.Layout()


class SettingsPanel(wx.Panel):
    """Settings panel."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.on_change = None
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        header = wx.StaticText(self, label="⚙️ Settings")
        header.SetFont(header.GetFont().Bold())
        sizer.Add(header, 0, wx.ALL, 5)
        
        # Mission time
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(self, label="Mission:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.years = wx.SpinCtrl(self, min=1, max=30, initial=5, size=(60, -1))
        self.years.Bind(wx.EVT_SPINCTRL, self._on_change)
        row.Add(self.years, 0, wx.RIGHT, 3)
        row.Add(wx.StaticText(self, label="years"), 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(row, 0, wx.ALL, 5)
        
        # Cycles
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(self, label="Cycles/yr:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.cycles = wx.SpinCtrl(self, min=100, max=20000, initial=5256, size=(70, -1))
        self.cycles.Bind(wx.EVT_SPINCTRL, self._on_change)
        row.Add(self.cycles, 0)
        sizer.Add(row, 0, wx.ALL, 5)
        
        # Delta T
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(self, label="ΔT:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.dt = wx.SpinCtrlDouble(self, min=0.5, max=30, initial=3, inc=0.5, size=(60, -1))
        self.dt.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        row.Add(self.dt, 0, wx.RIGHT, 3)
        row.Add(wx.StaticText(self, label="°C"), 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(row, 0, wx.ALL, 5)
        
        self.SetSizer(sizer)
    
    def get_hours(self) -> float:
        return self.years.GetValue() * 365 * 24
    
    def get_cycles(self) -> int:
        return self.cycles.GetValue()
    
    def get_dt(self) -> float:
        return self.dt.GetValue()
    
    def _on_change(self, event):
        if self.on_change:
            self.on_change()


class ReliabilityMainDialog(wx.Dialog):
    """Main reliability calculator dialog."""
    
    def __init__(self, parent, project_path: str = None):
        super().__init__(
            parent,
            title="⚡ Reliability Calculator",
            size=(1300, 850),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX
        )
        
        self.project_path = project_path
        self.parser: Optional[SchematicParser] = None
        self.sheet_data: Dict[str, Dict] = {}
        
        self._create_ui()
        self._bind_events()
        
        # Load project
        if project_path:
            self._load_project(project_path)
        else:
            self._load_test_data()
    
    def _create_ui(self):
        main = wx.BoxSizer(wx.VERTICAL)
        
        # Toolbar
        toolbar = self._create_toolbar()
        main.Add(toolbar, 0, wx.EXPAND | wx.ALL, 5)
        
        # Main splitter
        splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        
        # Left panel
        left = wx.Panel(splitter)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.sheet_panel = SheetPanel(left)
        left_sizer.Add(self.sheet_panel, 2, wx.EXPAND)
        
        self.settings_panel = SettingsPanel(left)
        left_sizer.Add(self.settings_panel, 0, wx.EXPAND | wx.TOP, 5)
        
        left.SetSizer(left_sizer)
        
        # Right panel
        right = wx.SplitterWindow(splitter, style=wx.SP_LIVE_UPDATE)
        
        # Editor panel
        editor_panel = wx.Panel(right)
        editor_sizer = wx.BoxSizer(wx.VERTICAL)
        
        editor_header = wx.StaticText(editor_panel, label="🔗 System Block Diagram")
        editor_header.SetFont(editor_header.GetFont().Bold())
        editor_sizer.Add(editor_header, 0, wx.ALL, 5)
        
        hint = wx.StaticText(editor_panel, 
            label="Drag rectangle to select multiple blocks → Group as Series/Parallel/K-of-N")
        hint.SetForegroundColour(wx.Colour(100, 100, 100))
        editor_sizer.Add(hint, 0, wx.LEFT | wx.BOTTOM, 5)
        
        self.editor = BlockEditor(editor_panel)
        editor_sizer.Add(self.editor, 1, wx.EXPAND | wx.ALL, 5)
        
        editor_panel.SetSizer(editor_sizer)
        
        # Bottom panel
        bottom = wx.SplitterWindow(right, style=wx.SP_LIVE_UPDATE)
        
        self.comp_panel = ComponentPanel(bottom)
        
        # Results panel
        results_panel = wx.Panel(bottom)
        results_sizer = wx.BoxSizer(wx.VERTICAL)
        
        results_header = wx.StaticText(results_panel, label="📊 System Results")
        results_header.SetFont(results_header.GetFont().Bold())
        results_sizer.Add(results_header, 0, wx.ALL, 5)
        
        self.results = wx.TextCtrl(results_panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.results.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        results_sizer.Add(self.results, 1, wx.EXPAND | wx.ALL, 5)
        
        btn_calc = wx.Button(results_panel, label="🔄 Calculate System Reliability")
        btn_calc.SetFont(btn_calc.GetFont().Bold())
        btn_calc.Bind(wx.EVT_BUTTON, self._on_calculate)
        results_sizer.Add(btn_calc, 0, wx.EXPAND | wx.ALL, 5)
        
        results_panel.SetSizer(results_sizer)
        
        bottom.SplitVertically(self.comp_panel, results_panel, 450)
        right.SplitHorizontally(editor_panel, bottom, 380)
        splitter.SplitVertically(left, right, 260)
        
        main.Add(splitter, 1, wx.EXPAND | wx.ALL, 5)
        
        # Status
        self.status = wx.StaticText(self, label="Ready")
        main.Add(self.status, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main)
    
    def _create_toolbar(self) -> wx.Panel:
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        sizer.Add(wx.StaticText(panel, label="Project:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        
        self.txt_project = wx.TextCtrl(panel, value="(none)", style=wx.TE_READONLY)
        sizer.Add(self.txt_project, 1, wx.RIGHT, 10)
        
        btn_load = wx.Button(panel, label="📂 Open...")
        btn_load.Bind(wx.EVT_BUTTON, self._on_open)
        sizer.Add(btn_load, 0, wx.RIGHT, 5)
        
        btn_save = wx.Button(panel, label="💾 Save Config")
        btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        sizer.Add(btn_save, 0, wx.RIGHT, 5)
        
        btn_load_cfg = wx.Button(panel, label="📁 Load Config")
        btn_load_cfg.Bind(wx.EVT_BUTTON, self._on_load_config)
        sizer.Add(btn_load_cfg, 0, wx.RIGHT, 15)
        
        btn_export = wx.Button(panel, label="📄 Export Report")
        btn_export.Bind(wx.EVT_BUTTON, self._on_export)
        sizer.Add(btn_export, 0)
        
        panel.SetSizer(sizer)
        return panel
    
    def _bind_events(self):
        self.sheet_panel.on_add = self._add_sheet
        self.editor.on_selection_change = self._on_block_select
        self.editor.on_structure_change = self._on_calculate
        self.settings_panel.on_change = self._recalculate_all
    
    def _load_project(self, path: str):
        """Load a KiCad project."""
        self.project_path = path
        self.txt_project.SetValue(path)
        
        self.parser = SchematicParser(path)
        if self.parser.parse():
            sheets = self.parser.get_sheet_paths()
            self.sheet_panel.set_sheets(sheets)
            self._calculate_sheets()
            self.status.SetLabel(f"Loaded {len(sheets)} sheets from {path}")
        else:
            wx.MessageBox(f"Could not parse schematics in:\n{path}", 
                         "Parse Error", wx.OK | wx.ICON_WARNING)
    
    def _load_test_data(self):
        """Load test data."""
        sheets = [
            "/Project Architecture/",
            "/Project Architecture/Power/",
            "/Project Architecture/Power/Protection Satellite 24V/",
            "/Project Architecture/Power/Battery Charger/",
            "/Project Architecture/Power/LDO_3v3_sat/",
            "/Project Architecture/Power/Ideal Diode Satellite/",
            "/Project Architecture/Power/Protection Battery/",
            "/Project Architecture/Power/System On Logic/",
            "/Project Architecture/Power/System On Logic/On Arbitration/",
            "/Project Architecture/Power/System On Logic/Off Arbitration/",
            "/Project Architecture/Power/System On Logic/On Memory/",
            "/Project Architecture/Power/LDO_3v3_bat/",
            "/Project Architecture/Power/Ideal Diode Battery/",
            "/Project Architecture/Power/Deploy/",
            "/Project Architecture/Power/Deploy/Boost/",
            "/Project Architecture/Power/Deploy/Boost/TRIGGER_LOGIC_B1/",
            "/Project Architecture/Power/Deploy/Boost/TRIGGER_LOGIC_B2/",
            "/Project Architecture/Power/Deploy/Buck/",
            "/Project Architecture/Power/Deploy/Buck/TRIGGER_LOGIC_B3/",
            "/Project Architecture/Power/Unlatch Arbitration/",
            "/Project Architecture/Power/Passivate Arbitration/",
            "/Project Architecture/Power/Passivate Memory/",
            "/Project Architecture/Control/MCU_A/",
            "/Project Architecture/Trigger IDD/",
        ]
        
        self.parser = create_test_data(sheets)
        self.sheet_panel.set_sheets(sheets)
        self.txt_project.SetValue("Test Data (from your sheet names)")
        self._calculate_sheets()
        self.status.SetLabel("Loaded test data")
    
    def _calculate_sheets(self):
        """Calculate reliability for all sheets."""
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
                cls = c.get_field("Reliability_Class", c.get_field("Class", ""))
                
                params = {
                    "n_cycles": cycles,
                    "delta_t": dt,
                    "t_ambient": c.get_float("T_Ambient", c.get_float("Temperature_Ambient", 25)),
                    "t_junction": c.get_float("T_Junction", c.get_float("Temperature_Junction", 85)),
                    "operating_power": c.get_float("Operating_Power", 0.01),
                    "rated_power": c.get_float("Rated_Power", 0.125),
                    "package": c.get_field("Package", c.footprint.split(":")[-1] if c.footprint else ""),
                    "transistor_type": c.get_field("Transistor_Type", "MOS"),
                    "diode_type": c.get_field("Diode_Type", "signal"),
                    "ic_type": c.get_field("IC_Type", ""),
                    "construction_year": c.get_int("Construction_Year", 2020),
                    "inductor_type": c.get_field("Inductor_Type", "power"),
                    "power_loss": c.get_float("Power_Loss", 0.1),
                    "surface_area": c.get_float("Surface_Area", 100),
                }
                
                lam = calculate_lambda(cls or "Resistor", params)
                r = reliability(lam, hours)
                total_lam += lam
                
                comp_data.append({
                    "ref": c.reference,
                    "value": c.value,
                    "class": cls or "Unknown",
                    "lambda": lam,
                    "r": r,
                })
            
            sheet_r = reliability(total_lam, hours)
            
            self.sheet_data[path] = {
                "components": comp_data,
                "lambda": total_lam,
                "r": sheet_r,
            }
    
    def _recalculate_all(self):
        """Recalculate everything."""
        self._calculate_sheets()
        
        # Update block data
        for bid, b in self.editor.blocks.items():
            if not b.is_group:
                data = self.sheet_data.get(b.name, {})
                b.reliability = data.get("r", 1.0)
                b.lambda_val = data.get("lambda", 0.0)
        
        self._on_calculate(None)
    
    def _calculate_system(self) -> Tuple[float, float]:
        """Calculate system reliability."""
        hours = self.settings_panel.get_hours()
        
        def calc(block_id: str) -> float:
            b = self.editor.blocks.get(block_id)
            if not b:
                return 1.0
            
            if b.is_group:
                child_rs = [calc(cid) for cid in b.children]
                
                if b.connection_type == ConnectionType.SERIES:
                    r = r_series(child_rs)
                elif b.connection_type == ConnectionType.PARALLEL:
                    r = r_parallel(child_rs)
                else:
                    r = r_k_of_n(child_rs, b.k_value)
                
                b.reliability = r
                b.lambda_val = lambda_from_reliability(r, hours)
                return r
            else:
                data = self.sheet_data.get(b.name, {})
                b.reliability = data.get("r", 1.0)
                b.lambda_val = data.get("lambda", 0.0)
                return b.reliability
        
        if not self.editor.root_id:
            return 1.0, 0.0
        
        sys_r = calc(self.editor.root_id)
        sys_lam = lambda_from_reliability(sys_r, hours)
        
        self.editor.Refresh()
        return sys_r, sys_lam
    
    def _add_sheet(self, path: str):
        """Add sheet to diagram."""
        # Check if already added
        for b in self.editor.blocks.values():
            if b.name == path:
                return
        
        label = path.rstrip('/').split('/')[-1] or "Root"
        block = self.editor.add_block(f"sheet_{len(self.editor.blocks)}", path, label)
        
        data = self.sheet_data.get(path, {})
        block.reliability = data.get("r", 1.0)
        block.lambda_val = data.get("lambda", 0.0)
        
        self.editor.Refresh()
    
    def _on_block_select(self, block_id: Optional[str]):
        """Handle block selection."""
        if block_id:
            b = self.editor.blocks.get(block_id)
            if b and not b.is_group:
                data = self.sheet_data.get(b.name, {})
                self.comp_panel.set_data(
                    b.name,
                    data.get("components", []),
                    data.get("lambda", 0),
                    data.get("r", 1)
                )
    
    def _on_calculate(self, event):
        """Calculate and display results."""
        sys_r, sys_lam = self._calculate_system()
        hours = self.settings_panel.get_hours()
        years = hours / (365 * 24)
        
        lines = [
            "═" * 45,
            "       SYSTEM RELIABILITY ANALYSIS",
            "═" * 45,
            "",
            f"  Mission Duration: {years:.1f} years ({hours:.0f} h)",
            "",
            f"  ► System Reliability:  R = {sys_r:.6f}",
            f"  ► Failure Rate:        λ = {sys_lam:.2e} /h",
        ]
        
        if sys_lam > 0:
            mttf = 1 / sys_lam
            lines.append(f"  ► MTTF:                {mttf:.2e} hours")
            lines.append(f"                         ({mttf/(365*24):.1f} years)")
        
        lines.extend([
            "",
            "═" * 45,
            "       BLOCK DETAILS",
            "═" * 45,
        ])
        
        for bid, b in sorted(self.editor.blocks.items()):
            if bid.startswith("__"):
                continue
            
            if b.is_group:
                typ = b.label
                lines.append(f"\n  [{typ}] ({len(b.children)} blocks)")
                lines.append(f"    R = {b.reliability:.6f}")
            else:
                lines.append(f"\n  {b.label}")
                lines.append(f"    λ = {b.lambda_val:.2e}, R = {b.reliability:.6f}")
        
        self.results.SetValue('\n'.join(lines))
        self.status.SetLabel(f"System R = {sys_r:.6f} ({years:.0f}y mission)")
    
    def _on_open(self, event):
        """Open a project."""
        dlg = wx.DirDialog(self, "Select KiCad Project", 
                          style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        
        if dlg.ShowModal() == wx.ID_OK:
            self.editor.clear()
            self.sheet_data.clear()
            self._load_project(dlg.GetPath())
        
        dlg.Destroy()
    
    def _on_save(self, event):
        """Save configuration."""
        default_dir = self.project_path or os.getcwd()
        
        dlg = wx.FileDialog(self, "Save Configuration", defaultDir=default_dir,
                           defaultFile="reliability_config.json",
                           wildcard="JSON (*.json)|*.json",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        
        if dlg.ShowModal() == wx.ID_OK:
            config = {
                "project": self.project_path,
                "structure": self.editor.get_structure(),
                "settings": {
                    "years": self.settings_panel.years.GetValue(),
                    "cycles": self.settings_panel.cycles.GetValue(),
                    "dt": self.settings_panel.dt.GetValue(),
                }
            }
            
            with open(dlg.GetPath(), 'w') as f:
                json.dump(config, f, indent=2)
            
            self.status.SetLabel(f"Saved to {dlg.GetPath()}")
        
        dlg.Destroy()
    
    def _on_load_config(self, event):
        """Load configuration."""
        dlg = wx.FileDialog(self, "Load Configuration",
                           wildcard="JSON (*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        
        if dlg.ShowModal() == wx.ID_OK:
            try:
                with open(dlg.GetPath(), 'r') as f:
                    config = json.load(f)
                
                settings = config.get("settings", {})
                self.settings_panel.years.SetValue(settings.get("years", 5))
                self.settings_panel.cycles.SetValue(settings.get("cycles", 5256))
                self.settings_panel.dt.SetValue(settings.get("dt", 3.0))
                
                self.editor.load_structure(config.get("structure", {}))
                self._recalculate_all()
                
                self.status.SetLabel(f"Loaded from {dlg.GetPath()}")
                
            except Exception as e:
                wx.MessageBox(f"Error: {e}", "Load Error", wx.OK | wx.ICON_ERROR)
        
        dlg.Destroy()
    
    def _on_export(self, event):
        """Export report."""
        dlg = wx.FileDialog(self, "Export Report",
                           wildcard="HTML (*.html)|*.html|Markdown (*.md)|*.md|CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            idx = dlg.GetFilterIndex()
            
            sys_r, sys_lam = self._calculate_system()
            hours = self.settings_panel.get_hours()
            
            if idx == 0:  # HTML
                content = self._generate_html(sys_r, sys_lam, hours)
            elif idx == 1:  # Markdown
                content = self._generate_md(sys_r, sys_lam, hours)
            else:  # CSV
                content = self._generate_csv()
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.status.SetLabel(f"Exported to {path}")
        
        dlg.Destroy()
    
    def _generate_html(self, sys_r: float, sys_lam: float, hours: float) -> str:
        years = hours / (365*24)
        html = f'''<!DOCTYPE html>
<html><head><title>Reliability Report</title>
<style>
body {{ font-family: Arial; margin: 20px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; }}
th {{ background: #f5f5f5; }}
.summary {{ background: #e8f4e8; padding: 15px; border-radius: 5px; margin: 15px 0; }}
</style></head><body>
<h1>⚡ Reliability Analysis Report</h1>
<div class="summary">
<h2>System Summary</h2>
<p><b>Mission Duration:</b> {years:.1f} years</p>
<p><b>System Reliability:</b> R = {sys_r:.6f}</p>
<p><b>Failure Rate:</b> λ = {sys_lam:.2e} /hour</p>
</div>
<h2>Sheet Analysis</h2>
'''
        for path, data in sorted(self.sheet_data.items()):
            html += f'''<h3>{path}</h3>
<p>Sheet R = {data["r"]:.6f}, λ = {data["lambda"]:.2e}</p>
<table><tr><th>Ref</th><th>Value</th><th>Class</th><th>λ</th><th>R</th></tr>
'''
            for c in data["components"]:
                html += f'<tr><td>{c["ref"]}</td><td>{c["value"]}</td><td>{c["class"]}</td>'
                html += f'<td>{c["lambda"]:.2e}</td><td>{c["r"]:.4f}</td></tr>\n'
            html += '</table>\n'
        
        html += '</body></html>'
        return html
    
    def _generate_md(self, sys_r: float, sys_lam: float, hours: float) -> str:
        years = hours / (365*24)
        md = f'''# Reliability Analysis Report

## System Summary

- **Mission Duration:** {years:.1f} years
- **System Reliability:** R = {sys_r:.6f}
- **Failure Rate:** λ = {sys_lam:.2e} /hour

## Sheet Analysis

'''
        for path, data in sorted(self.sheet_data.items()):
            md += f'''### {path}

Sheet R = {data["r"]:.6f}, λ = {data["lambda"]:.2e}

| Ref | Value | Class | λ | R |
|-----|-------|-------|---|---|
'''
            for c in data["components"]:
                md += f'| {c["ref"]} | {c["value"]} | {c["class"]} | {c["lambda"]:.2e} | {c["r"]:.4f} |\n'
            md += '\n'
        
        return md
    
    def _generate_csv(self) -> str:
        lines = ["Sheet,Reference,Value,Class,Lambda,Reliability"]
        for path, data in sorted(self.sheet_data.items()):
            for c in data["components"]:
                lines.append(f'"{path}","{c["ref"]}","{c["value"]}","{c["class"]}",{c["lambda"]:.2e},{c["r"]:.6f}')
        return '\n'.join(lines)


if __name__ == "__main__":
    app = wx.App()
    dlg = ReliabilityMainDialog(None)
    dlg.ShowModal()
    dlg.Destroy()
