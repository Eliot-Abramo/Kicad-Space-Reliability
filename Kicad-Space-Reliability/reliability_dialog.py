"""
Main Reliability Calculator Dialog
===================================
Primary UI integrating all IEC TR 62380 features with block diagram editor.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wx

try:
    import pcbnew
    _HAS_PCBNEW = True
except ImportError:
    _HAS_PCBNEW = False

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
)
from .schematic_parser import SchematicParser, Component, Sheet, create_test_data
from .project_manager import ProjectManager
from .report_generator import ReportGenerator, ReportData
from .mission_profile import MissionProfile

try:
    from .ui.panels import Colors, SheetPanel, SettingsPanel, ComponentPanel
except ImportError:
    from ui.panels import Colors, SheetPanel, SettingsPanel, ComponentPanel


class ReliabilityMainDialog(wx.Dialog):
    """Main reliability calculator dialog."""

    def __init__(self, parent, project_path: str = None):
        display = wx.Display(0)
        rect = display.GetClientArea()
        w = min(1600, int(rect.Width * 0.9))
        h = min(1000, int(rect.Height * 0.9))

        super().__init__(
            parent,
            title="Reliability Calculator (IEC TR 62380)",
            size=(w, h),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX,
        )
        self.SetMinSize((1100, 750))
        self.SetBackgroundColour(Colors.BACKGROUND)

        self.project_path = project_path
        self.project_manager: Optional[ProjectManager] = None
        self.parser: Optional[SchematicParser] = None
        self.sheet_data: Dict[str, Dict] = {}
        self.component_edits: Dict[str, Dict[str, Dict]] = {}

        if project_path:
            self.project_manager = ProjectManager(project_path)

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
        title = wx.StaticText(header, label="[Z] Reliability Calculator")
        title.SetFont(
            wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
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
        self.results = wx.TextCtrl(
            results_panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE
        )
        self.results.SetFont(
            wx.Font(
                9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
            )
        )
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

        sizer.Add(
            wx.StaticText(panel, label="Project:"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            5,
        )
        self.txt_project = wx.TextCtrl(panel, value="(none)", style=wx.TE_READONLY)
        sizer.Add(self.txt_project, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        btn_open = wx.Button(panel, label="Open...", size=(80, -1))
        btn_open.Bind(wx.EVT_BUTTON, self._on_open)
        sizer.Add(btn_open, 0, wx.RIGHT, 5)

        btn_save = wx.Button(panel, label="Save", size=(70, -1))
        btn_save.SetToolTip("Save components and block diagram to Reliability/reliability_data.json")
        btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        sizer.Add(btn_save, 0, wx.RIGHT, 15)

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
        self.editor.on_structure_change = self._on_structure_change
        self.settings_panel.on_change = self._recalculate_all
        self.comp_panel.on_component_edit = self._edit_single_component

    def _on_structure_change(self):
        self._on_calculate(None)
        self._save_reliability_data()

    def _load_project(self, path: str):
        self.project_path = path
        self.txt_project.SetValue(path)
        self.project_badge.SetLabel(Path(path).name if path else "(no project)")

        # Project manager flow: check Reliability/ folder
        if not self.project_manager.reliability_folder_exists():
            wx.MessageBox(
                f"Reliability folder not found in project.\n\n"
                f"Created: {self.project_manager.reliability_dir}\n"
                f"Using default values (Miscellaneous, FIT 0) and blank canvas.",
                "Reliability Folder Created",
                wx.OK | wx.ICON_INFORMATION,
            )
            self.project_manager.ensure_reliability_folder()
            self._init_default_data()
        else:
            data = self.project_manager.load_data()
            if data:
                self._apply_loaded_data(data)
            else:
                self._init_default_data()

        # PRIMARY: parse schematic (.kicad_sch) files for components.
        # The schematic is the single source of truth for component references,
        # hierarchy, values, and custom fields.
        sch_loaded = False
        try:
            self.parser = SchematicParser(path)
            if self.parser.parse() and self.parser.all_components:
                sheets = self.parser.get_sheet_paths()
                self.sheet_panel.set_sheets(sheets)
                self._ensure_default_components()
                self._save_reliability_data()
                self._calculate_sheets()
                n = len(self.parser.all_components)
                self.status.SetLabel(
                    f"Loaded {n} component(s) from {len(sheets)} sheet(s)"
                )
                sch_loaded = True
        except Exception as e:
            sch_loaded = False

        if not sch_loaded:
            # FALLBACK: try loading from PCB board via pcbnew (less reliable)
            board_loaded = False
            try:
                board_loaded = self._load_from_board(path)
            except Exception:
                board_loaded = False
            if not board_loaded:
                wx.MessageBox(
                    f"Could not load components from schematics or PCB in:\n{path}\n\n"
                    "Ensure .kicad_sch files exist in the project directory.",
                    "Parse Error",
                    wx.OK | wx.ICON_WARNING,
                )

    def _init_default_data(self):
        """Initialize with default data (blank canvas, default settings)."""
        default = ProjectManager.default_data()
        self.component_edits = default.get("components", {})
        self.editor.load_structure(default.get("structure", {}))
        settings = default.get("settings", {})
        self.settings_panel.years.SetValue(settings.get("years", 5))
        self.settings_panel.cycles.SetValue(settings.get("cycles", 5256))
        self.settings_panel.dt.SetValue(settings.get("dt", 3.0))
        self.settings_panel.tau_on.SetValue(settings.get("tau_on", 1.0))
        mp = default.get("mission_profile")
        if mp:
            self.settings_panel.set_mission_profile(MissionProfile.from_dict(mp))
        self._save_reliability_data()

    def _apply_loaded_data(self, data: dict):
        """Apply loaded reliability_data.json."""
        self.component_edits = data.get("components") or {}
        self.editor.load_structure(data.get("structure") or {})
        settings = data.get("settings") or {}
        # Coerce to valid numbers (JSON null becomes None; default only applies when key missing)
        def _num(v, default):
            if v is None: return default
            try: return type(default)(v)
            except (TypeError, ValueError): return default
        self.settings_panel.years.SetValue(_num(settings.get("years"), 5))
        self.settings_panel.cycles.SetValue(_num(settings.get("cycles"), 5256))
        self.settings_panel.dt.SetValue(_num(settings.get("dt"), 3.0))
        self.settings_panel.tau_on.SetValue(_num(settings.get("tau_on"), 1.0))
        mp = data.get("mission_profile")
        if mp:
            self.settings_panel.set_mission_profile(MissionProfile.from_dict(mp))
        else:
            self.settings_panel.set_mission_profile(MissionProfile.single_phase(
                years=_num(settings.get("years"), 5),
                n_cycles=_num(settings.get("cycles"), 5256),
                delta_t=_num(settings.get("dt"), 3.0),
                tau_on=_num(settings.get("tau_on"), 1.0),
            ))

    def _load_from_board(self, project_path: str) -> bool:
        """Load components directly from pcbnew board (native API).

        Uses pcbnew.GetBoard().GetFootprints() to enumerate all footprints.
        This is the most reliable way to get reference designators because
        pcbnew always has the correctly annotated references from the netlist.

        The ENTIRE method is wrapped in try/except so it can NEVER crash the
        caller. If anything goes wrong, it returns False and the caller falls
        back to schematic parsing.

        Returns True if components were loaded successfully.
        """
        if not _HAS_PCBNEW:
            return False

        try:
            board = pcbnew.GetBoard()
            if not board:
                return False

            footprints = board.GetFootprints()
            if not footprints:
                return False

            default_sheet = f"/{Path(project_path).name}/"

            # Build a synthetic parser-like structure so the rest of the tool works
            # Group components by sheet path (from footprint's sheet name)
            sheet_comps: Dict[str, List[Component]] = {}
            skipped = 0
            for fp in footprints:
                try:
                    ref = fp.GetReference()
                    if not ref or ref.startswith("#") or ref.endswith("?"):
                        skipped += 1
                        continue

                    value = fp.GetValue()

                    # Footprint string -- use safe accessors
                    footprint_str = ""
                    try:
                        fpid = fp.GetFPID()
                        # Try multiple API names across KiCad versions
                        for method in ("GetUniStringLibItemName", "GetLibItemName",
                                       "Format"):
                            fn = getattr(fpid, method, None)
                            if fn is not None:
                                footprint_str = str(fn())
                                break
                    except Exception:
                        footprint_str = ""

                    lib_id = ""
                    try:
                        fpid = fp.GetFPID()
                        for method in ("GetUniStringLibId", "GetLibNickname",
                                       "GetFullLibId"):
                            fn = getattr(fpid, method, None)
                            if fn is not None:
                                lib_id = str(fn())
                                break
                    except Exception:
                        lib_id = ""

                    # Get the sheet path -- group all under one sheet if unavailable
                    sheet_name = ""
                    for method in ("GetSheetname", "GetPath"):
                        fn = getattr(fp, method, None)
                        if fn is not None:
                            try:
                                val = fn()
                                if val:
                                    sheet_name = str(val).strip("/").split("/")[-1]
                                    break
                            except Exception:
                                pass

                    sheet_path = f"/{sheet_name}/" if sheet_name else default_sheet

                    # Collect all custom fields (safe)
                    fields = {}
                    try:
                        fp_fields = fp.GetFields()
                        if fp_fields:
                            for f in fp_fields:
                                fname = f.GetName()
                                fval = f.GetText()
                                if fname and fname not in (
                                    "Reference", "Value", "Footprint", "Datasheet"
                                ):
                                    fields[fname] = fval
                    except Exception:
                        fields = {}

                    comp = Component(
                        reference=ref,
                        value=value,
                        lib_id=lib_id,
                        sheet_path=sheet_path,
                        footprint=footprint_str,
                        fields=fields,
                    )
                    sheet_comps.setdefault(sheet_path, []).append(comp)

                except Exception:
                    skipped += 1
                    continue

            if not sheet_comps:
                return False

            # Build a synthetic SchematicParser from board data
            parser = SchematicParser(project_path)
            for spath, comps in sheet_comps.items():
                name = spath.strip("/").split("/")[-1] or "Root"
                sheet = Sheet(
                    name=name, path=spath,
                    filename=str(Path(project_path) / f"{name}.kicad_sch"),
                    components=comps,
                )
                parser.sheets[spath] = sheet
                parser.all_components.extend(comps)

            self.parser = parser
            sheets = parser.get_sheet_paths()
            self.sheet_panel.set_sheets(sheets)
            self._ensure_default_components()
            self._save_reliability_data()
            self._calculate_sheets()

            total_comps = sum(len(c) for c in sheet_comps.values())
            msg = f"Loaded {total_comps} components from PCB ({len(sheets)} sheet(s))"
            if skipped:
                msg += f" [{skipped} skipped]"
            self.status.SetLabel(msg)
            return True

        except Exception:
            # NEVER crash the caller -- return False and let fallback handle it
            return False

    def _ensure_default_components(self):
        """Ensure every parsed component has a properly classified entry."""
        if not self.parser:
            return
        for path in self.parser.get_sheet_paths():
            for c in self.parser.get_sheet_components(path):
                if path not in self.component_edits:
                    self.component_edits[path] = {}
                if c.reference not in self.component_edits[path]:
                    self.component_edits[path][c.reference] = {
                        "_component_type": classify_component(
                            c.reference, c.value, c.fields),
                    }

    def _save_reliability_data(self):
        """Save all data to Reliability/reliability_data.json."""
        if not self.project_manager:
            self.status.SetLabel("Warning: no project manager -- data not saved")
            return
        def _val(ctrl, default):
            v = ctrl.GetValue()
            if v is None: return default
            try: return type(default)(v)
            except (TypeError, ValueError): return default
        sp = self.settings_panel
        data = {
            "components": self.component_edits,
            "structure": self.editor.get_structure(),
            "settings": {
                "years": _val(sp.years, 5),
                "cycles": _val(sp.cycles, 5256),
                "dt": _val(sp.dt, 3.0),
                "tau_on": _val(sp.tau_on, 1.0),
            },
            "mission_profile": sp.get_mission_profile().to_dict(),
        }
        if self.project_manager.save_data(data):
            self.status.SetLabel("Saved")
        else:
            self.status.SetLabel("Error: failed to save reliability data")
            wx.MessageBox(
                "Failed to save reliability data to disk.\n"
                "Check that the Reliability/ folder exists and is writable.",
                "Save Error", wx.OK | wx.ICON_ERROR)

    def _load_test_data(self):
        sheets = [
            "/Project/",
            "/Project/Power/",
            "/Project/Power/LDO_3v3/",
            "/Project/Power/Buck_5V/",
            "/Project/MCU/",
            "/Project/Sensors/",
        ]
        self.parser = create_test_data(sheets)
        self.sheet_panel.set_sheets(sheets)
        self.txt_project.SetValue("Test Data")
        self.project_badge.SetLabel("Test Data")
        self._calculate_sheets()
        self.status.SetLabel("Loaded test data")

    def _build_sheet_data_from_edits(self):
        """Build sheet_data from component_edits when no parser (e.g. loaded from JSON file)."""
        if not self.component_edits:
            return
        hours = self.settings_panel.get_hours()
        cycles = self.settings_panel.get_cycles()
        dt = self.settings_panel.get_dt()
        tau_on = self.settings_panel.get_tau_on()

        for path, edits in self.component_edits.items():
            if not isinstance(edits, dict):
                continue
            comp_data = []
            total_lam = 0.0
            for ref, edited in edits.items():
                if not isinstance(edited, dict):
                    continue
                ct = edited.get("_component_type", "Resistor")
                ovr = edited.get("override_lambda")
                if ovr is not None and float(ovr) > 0:
                    lam = float(ovr) * 1e-9
                    params = edited.copy()
                else:
                    params = edited.copy()
                    params.setdefault("n_cycles", cycles)
                    params.setdefault("delta_t", dt)
                    params.setdefault("tau_on", tau_on)
                    try:
                        res = calculate_component_lambda(ct, params)
                        lam = float(res.get("lambda_total", 0) or 0)
                    except Exception:
                        lam = 0.0
                r = reliability_from_lambda(lam, hours)
                total_lam += lam
                comp_entry = {
                    "ref": ref,
                    "value": edited.get("value", ""),
                    "class": ct,
                    "lambda": lam,
                    "r": r,
                    "params": params,
                }
                if ovr is not None and float(ovr) > 0:
                    comp_entry["override_lambda"] = ovr
                comp_data.append(comp_entry)
            sheet_r = reliability_from_lambda(total_lam, hours)
            self.sheet_data[path] = {"components": comp_data, "lambda": total_lam, "r": sheet_r}

    def _calculate_sheets(self):
        if not self.parser:
            return
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
                    # Check for lambda override FIRST
                    ovr = edited.get("override_lambda")
                    if ovr is not None and float(ovr) > 0:
                        lam = float(ovr) * 1e-9   # override is in FIT, convert to /h
                        cls_name = ct
                        params = edited.copy()
                    else:
                        params = edited.copy()
                        params.setdefault("n_cycles", cycles)
                        params.setdefault("delta_t", dt)
                        params.setdefault("tau_on", tau_on)
                        result = calculate_component_lambda(ct, params)
                        lam = float(result.get("lambda_total", 0) or 0)
                        cls_name = ct
                else:
                    cls = c.get_field("Reliability_Class", c.get_field("Class", ""))
                    if not cls:
                        cls = classify_component(c.reference, c.value, {})
                    params = {
                        "n_cycles": cycles,
                        "delta_t": dt,
                        "tau_on": tau_on,
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
                comp_entry = {
                    "ref": c.reference,
                    "value": c.value,
                    "class": cls_name,
                    "lambda": lam,
                    "r": r,
                    "params": params,
                }
                # Propagate override flag so sensitivity/MC can skip this component
                ovr_val = edited.get("override_lambda") if edited else None
                if ovr_val is not None and float(ovr_val) > 0:
                    comp_entry["override_lambda"] = ovr_val
                comp_data.append(comp_entry)

            sheet_r = reliability_from_lambda(total_lam, hours)
            self.sheet_data[path] = {
                "components": comp_data,
                "lambda": total_lam,
                "r": sheet_r,
            }

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
        sheet_path = sheet_path or ""
        components = self.parser.get_sheet_components(sheet_path) if self.parser else []
        # Fallback: use sheet_data components if parser has none (e.g. path mismatch after load)
        if not components and sheet_path:
            sheet_entry = self.sheet_data.get(sheet_path, {})
            comps = sheet_entry.get("components", [])
            if comps:
                components = None  # signal to use comps from sheet_data
        if not components and (not sheet_path or not self.sheet_data.get(sheet_path, {}).get("components")):
            wx.MessageBox(
                f"No components found for sheet '{sheet_path or '(empty)'}'.\n\n"
                "Ensure the schematic has components and the block's sheet path matches.",
                "No Components",
                wx.OK | wx.ICON_INFORMATION,
            )
            return
        if components is not None:
            comp_list = []
            for c in components:
                edited = self.component_edits.get(sheet_path, {}).get(c.reference, {})
                ct = edited.get("_component_type",
                                classify_component(c.reference, c.value, c.fields))
                fields = dict(edited) if edited else dict(c.fields)
                comp_list.append(ComponentData(c.reference, c.value, ct, fields))
        else:
            comps = self.sheet_data[sheet_path]["components"]
            comp_list = [
                ComponentData(
                    c.get("ref", "?"),
                    c.get("value", ""),
                    c.get("class", "Miscellaneous"),
                    c.get("params", {}),
                )
                for c in comps
            ]
        dlg = BatchComponentEditorDialog(
            self, comp_list, self.settings_panel.get_hours()
        )
        if dlg.ShowModal() == wx.ID_OK:
            if sheet_path not in self.component_edits:
                self.component_edits[sheet_path] = {}
            for ref, fields in dlg.get_results().items():
                self.component_edits[sheet_path][ref] = fields
            self._recalculate_all()
            self._save_reliability_data()
        dlg.Destroy()

    def _on_calculate(self, event):
        sys_r, sys_lam = self._calculate_system()
        hours = self.settings_panel.get_hours()
        years = hours / (365 * 24)

        lines = [
            "=" * 45,
            "       SYSTEM RELIABILITY ANALYSIS",
            "=" * 45,
            "",
            f"  Mission: {years:.1f} years ({hours:.0f} h)",
            "",
            f"   Reliability:  R = {sys_r:.6f}",
            f"   Failure Rate: L = {sys_lam*1e9:.2f} FIT",
        ]
        if sys_lam > 0:
            mttf = 1 / sys_lam
            lines.append(f"   MTTF: {mttf/(365*24):.1f} years")

        lines.extend(["", "=" * 45, "            BLOCK DETAILS", "=" * 45])
        for bid, b in sorted(self.editor.blocks.items()):
            if bid.startswith("__"):
                continue
            if b.is_group:
                lines.append(f"\n  [{b.label}] ({len(b.children)} blocks)")
                lines.append(f"    R = {float(b.reliability or 1.0):.6f}")
            else:
                lines.append(f"\n  {b.label}")
                lines.append(
                    f"    L = {float(b.lambda_val or 0)*1e9:.1f} FIT, R = {float(b.reliability or 1.0):.6f}"
                )

        self.results.SetValue("\n".join(lines))
        self.status.SetLabel(f"System R = {sys_r:.6f} ({years:.1f}y)")

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
                if sheet_path not in self.component_edits:
                    self.component_edits[sheet_path] = {}
                self.component_edits[sheet_path][ref] = result
                self._recalculate_all()
                self._save_reliability_data()
                data = self.sheet_data.get(sheet_path, {})
                self.comp_panel.set_data(
                    sheet_path,
                    data.get("components", []),
                    float(data.get("lambda", 0) or 0),
                    float(data.get("r", 1) or 1),
                )
        dlg.Destroy()

    def _edit_sheet_components(self, sheets: List[str]):
        if not self.parser:
            return
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
            wx.MessageBox(
                "No components found.", "No Components", wx.OK | wx.ICON_INFORMATION
            )
            return

        dlg = BatchComponentEditorDialog(
            self, all_comps, self.settings_panel.get_hours()
        )
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
            self._save_reliability_data()
        dlg.Destroy()

    def _on_open(self, event):
        """Open reliability_data.json file (or KiCad project folder)."""
        default_dir = ""
        if self.project_manager:
            default_dir = str(self.project_manager.get_reliability_folder())
        dlg = wx.FileDialog(
            self,
            "Open Reliability Data (JSON)",
            defaultDir=default_dir,
            defaultFile="reliability_data.json",
            wildcard="JSON (*.json)|*.json|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            dlg.Destroy()
            self._load_json_file(path)
            return
        dlg.Destroy()

    def _load_json_file(self, json_path: str):
        """Load reliability data from a JSON file."""
        import json
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            wx.MessageBox(f"Could not load file:\n{e}", "Open Error", wx.OK | wx.ICON_ERROR)
            return

        self.editor.clear()
        self.sheet_data.clear()
        self.component_edits.clear()
        self.parser = None

        # Derive project path: if file is in .../Reliability/reliability_data.json, use parent
        path_obj = Path(json_path)
        if path_obj.name.lower() == "reliability_data.json" and path_obj.parent.name == "Reliability":
            self.project_path = str(path_obj.parent.parent)
        else:
            self.project_path = str(path_obj.parent)
        self.project_manager = ProjectManager(self.project_path) if self.project_path else None

        self.txt_project.SetValue(path_obj.parent.name)
        self.project_badge.SetLabel(path_obj.name)

        self._apply_loaded_data(data)
        self._build_sheet_data_from_edits()
        self._recalculate_all()
        self.sheet_panel.set_sheets(list(self.component_edits.keys()) or ["/"])
        self._save_reliability_data()
        self.status.SetLabel(f"Loaded: {path_obj.name}")

    def _on_save(self, event):
        self._save_reliability_data()
        self.status.SetLabel("Saved to Reliability/reliability_data.json")

    def _on_monte_carlo(self, event):
        """Open comprehensive analysis dialog with Monte Carlo and Sobol sensitivity."""
        try:
            from .analysis_dialog import AnalysisDialog

            sys_r, sys_lam = self._calculate_system()

            if sys_lam <= 0:
                wx.MessageBox(
                    "System failure rate is zero. Add components first.",
                    "No Data",
                    wx.OK | wx.ICON_WARNING,
                )
                return

            dlg = AnalysisDialog(
                self,
                system_lambda=sys_lam,
                mission_hours=self.settings_panel.get_hours(),
                sheet_data=self.sheet_data,
                blocks=self.editor.blocks,
                root_id=getattr(self.editor, "root_id", None),
                project_path=self.project_path,
                logo_path=(
                    str(self.project_manager.get_available_logo_path())
                    if self.project_manager and self.project_manager.logo_exists()
                    else None
                ),
                logo_mime=self.project_manager.get_logo_mime_type() if self.project_manager else "image/png",
                n_cycles=int(self.settings_panel.cycles.GetValue()),
                delta_t=float(self.settings_panel.dt.GetValue()),
                title="System Reliability Analysis",
            )
            dlg.ShowModal()
            dlg.Destroy()

        except Exception as e:
            wx.MessageBox(f"Error: {e}", "Analysis Error", wx.OK | wx.ICON_ERROR)

    def _on_export(self, event):
        # Determine initial directory and filename
        if self.project_manager:
            reports_folder = self.project_manager.get_reports_folder()
            initial_dir = str(reports_folder)
            default_name = (
                f"reliability_report_{wx.DateTime.Now().Format('%Y%m%d_%H%M%S')}"
            )
        else:
            initial_dir = ""
            default_name = "reliability_report"

        dlg = wx.FileDialog(
            self,
            "Export Report",
            defaultFile=default_name + ".html",
            defaultDir=initial_dir,
            wildcard="HTML (*.html)|*.html|Markdown (*.md)|*.md|CSV (*.csv)|*.csv|JSON (*.json)|*.json",
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
            project_name=(
                Path(self.project_path).name
                if self.project_path
                else "Reliability Report"
            ),
            mission_hours=hours,
            mission_years=years,
            n_cycles=int(self.settings_panel.cycles.GetValue()),
            delta_t=float(self.settings_panel.dt.GetValue()),
            system_reliability=sys_r,
            system_lambda=sys_lam,
            system_mttf_hours=1 / sys_lam if sys_lam > 0 else float("inf"),
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
- **Failure Rate:** L = {sys_lam*1e9:.2f} FIT

## Sheet Analysis

"""
        for path, data in sorted(self.sheet_data.items()):
            fit = float(data["lambda"]) * 1e9
            md += f"""### {path}

R = {float(data["r"]):.6f}, L = {fit:.2f} FIT

| Ref | Value | Type | L (FIT) | R |
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
                lines.append(
                    f'"{path}","{c["ref"]}","{c["value"]}","{c["class"]}",{c_fit:.2f},{float(c["r"]):.6f}'
                )
        return "\n".join(lines)

    def _generate_json(self, sys_r: float, sys_lam: float, hours: float) -> str:
        return json.dumps(
            {
                "system": {
                    "reliability": sys_r,
                    "lambda_fit": sys_lam * 1e9,
                    "mission_hours": hours,
                },
                "sheets": self.sheet_data,
            },
            indent=2,
        )


if __name__ == "__main__":
    app = wx.App()
    dlg = ReliabilityMainDialog(None)
    dlg.ShowModal()
    dlg.Destroy()
