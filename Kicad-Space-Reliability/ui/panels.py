"""
Reliability Calculator UI Panels
================================
Sheet panel, mission settings, component list, and mission phase dialog.
"""

from typing import List

import wx
import wx.lib.scrolledpanel as scrolled

try:
    from ..mission_profile import MissionPhase, MissionProfile, MISSION_TEMPLATES
    from ..component_editor import QuickReferenceDialog
except ImportError:
    from mission_profile import MissionPhase, MissionProfile, MISSION_TEMPLATES
    from component_editor import QuickReferenceDialog


class Colors:
    """Design system colors."""
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
            if self.on_add:
                self.on_add(self.sheets[i])

    def _on_add_all(self, event):
        for s in self.sheets:
            if self.on_add:
                self.on_add(s)

    def _on_dclick(self, event):
        self._on_add(event)

    def _on_edit(self, event):
        selections = self.list.GetSelections()
        if selections and self.on_edit:
            self.on_edit([self.sheets[i] for i in selections])
        elif not selections:
            wx.MessageBox("Select a sheet first.", "No Selection", wx.OK | wx.ICON_INFORMATION)


class MissionPhaseDialog(wx.Dialog):
    """Dialog for editing a single mission phase."""

    def __init__(self, parent, phase: MissionPhase = None, title="Edit Mission Phase"):
        super().__init__(parent, title=title, size=(380, 320),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetBackgroundColour(Colors.PANEL_BG)

        sizer = wx.BoxSizer(wx.VERTICAL)
        form = wx.FlexGridSizer(7, 2, 8, 10)
        form.AddGrowableCol(1, 1)

        form.Add(wx.StaticText(self, label="Phase Name:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.name_ctrl = wx.TextCtrl(self, value=phase.name if phase else "Phase 1")
        form.Add(self.name_ctrl, 0, wx.EXPAND)

        form.Add(wx.StaticText(self, label="Duration (%):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.duration_ctrl = wx.SpinCtrlDouble(self, min=0.1, max=100.0,
            initial=(phase.duration_frac * 100) if phase else 100.0, inc=1.0, size=(100, -1))
        self.duration_ctrl.SetDigits(1)
        form.Add(self.duration_ctrl, 0)

        form.Add(wx.StaticText(self, label="T_ambient (degC):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.t_amb_ctrl = wx.SpinCtrlDouble(self, min=-55, max=200,
            initial=phase.t_ambient if phase else 25.0, inc=5.0, size=(100, -1))
        self.t_amb_ctrl.SetDigits(1)
        form.Add(self.t_amb_ctrl, 0)

        form.Add(wx.StaticText(self, label="T_junction (degC):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.t_junc_ctrl = wx.SpinCtrlDouble(self, min=-55, max=250,
            initial=phase.t_junction if phase else 85.0, inc=5.0, size=(100, -1))
        self.t_junc_ctrl.SetDigits(1)
        form.Add(self.t_junc_ctrl, 0)

        form.Add(wx.StaticText(self, label="Thermal cycles/yr:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.cycles_ctrl = wx.SpinCtrl(self, min=0, max=50000,
            initial=phase.n_cycles if phase else 5256, size=(100, -1))
        form.Add(self.cycles_ctrl, 0)

        form.Add(wx.StaticText(self, label="Delta_T (degC):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.dt_ctrl = wx.SpinCtrlDouble(self, min=0.0, max=100.0,
            initial=phase.delta_t if phase else 3.0, inc=0.5, size=(100, -1))
        self.dt_ctrl.SetDigits(1)
        form.Add(self.dt_ctrl, 0)

        form.Add(wx.StaticText(self, label="tau_on (0-1):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tau_ctrl = wx.SpinCtrlDouble(self, min=0.0, max=1.0,
            initial=phase.tau_on if phase else 1.0, inc=0.05, size=(100, -1))
        self.tau_ctrl.SetDigits(2)
        form.Add(self.tau_ctrl, 0)

        sizer.Add(form, 1, wx.EXPAND | wx.ALL, 15)
        btn_sizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)

    def get_phase(self) -> MissionPhase:
        return MissionPhase(
            name=self.name_ctrl.GetValue(),
            duration_frac=self.duration_ctrl.GetValue() / 100.0,
            t_ambient=self.t_amb_ctrl.GetValue(),
            t_junction=self.t_junc_ctrl.GetValue(),
            n_cycles=self.cycles_ctrl.GetValue(),
            delta_t=self.dt_ctrl.GetValue(),
            tau_on=self.tau_ctrl.GetValue(),
        )


class SettingsPanel(wx.Panel):
    """Mission profile settings with single-phase defaults and multi-phase support."""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.on_change = None
        self._mission_profile = None

        main = wx.BoxSizer(wx.VERTICAL)
        lbl = wx.StaticText(self, label="Mission Profile")
        lbl.SetFont(lbl.GetFont().Bold())
        main.Add(lbl, 0, wx.ALL, 10)

        tmpl_row = wx.BoxSizer(wx.HORIZONTAL)
        tmpl_row.Add(wx.StaticText(self, label="Template:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        templates = ["(Single-Phase)"] + sorted(MISSION_TEMPLATES.keys())
        self.template_combo = wx.Choice(self, choices=templates, size=(160, -1))
        self.template_combo.SetSelection(0)
        self.template_combo.Bind(wx.EVT_CHOICE, self._on_template_select)
        tmpl_row.Add(self.template_combo, 1)
        main.Add(tmpl_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        main.AddSpacer(6)

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
        self.cycles.SetToolTip("Annual thermal cycles (5256, approx. LEO satellite)")
        form.Add(self.cycles, 0)
        form.Add(wx.StaticText(self, label="/year"), 0, wx.ALIGN_CENTER_VERTICAL)

        form.Add(wx.StaticText(self, label="dT per cycle"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.dt = wx.SpinCtrlDouble(self, min=0.5, max=50, initial=3.0, inc=0.5, size=(80, -1))
        self.dt.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        form.Add(self.dt, 0)
        form.Add(wx.StaticText(self, label="degC"), 0, wx.ALIGN_CENTER_VERTICAL)

        form.Add(wx.StaticText(self, label="Default tau_on"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tau_on = wx.SpinCtrlDouble(self, min=0.01, max=1.0, initial=1.0, inc=0.05, size=(80, -1))
        self.tau_on.SetDigits(2)
        self.tau_on.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        self.tau_on.SetToolTip("Working time ratio (1.0 = continuous, 0.5 = 50% duty)")
        form.Add(self.tau_on, 0)
        form.Add(wx.StaticText(self, label="(0-1)"), 0, wx.ALIGN_CENTER_VERTICAL)

        main.Add(form, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
        main.AddSpacer(6)

        self.phase_label = wx.StaticText(self, label="Phases: (single-phase mode)")
        self.phase_label.SetFont(self.phase_label.GetFont().MakeItalic())
        main.Add(self.phase_label, 0, wx.LEFT | wx.RIGHT, 10)

        self.phase_list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SIMPLE, size=(-1, 90))
        self.phase_list.InsertColumn(0, "Phase", width=80)
        self.phase_list.InsertColumn(1, "Dur%", width=45)
        self.phase_list.InsertColumn(2, "T_amb", width=45)
        self.phase_list.InsertColumn(3, "T_junc", width=48)
        self.phase_list.InsertColumn(4, "Cycles", width=50)
        self.phase_list.InsertColumn(5, "dT", width=35)
        self.phase_list.InsertColumn(6, "tau", width=35)
        self.phase_list.Hide()
        main.Add(self.phase_list, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_add_phase = wx.Button(self, label="+ Phase", size=(65, -1))
        self.btn_edit_phase = wx.Button(self, label="Edit", size=(50, -1))
        self.btn_remove_phase = wx.Button(self, label="- Phase", size=(65, -1))
        self.btn_add_phase.Bind(wx.EVT_BUTTON, self._on_add_phase)
        self.btn_edit_phase.Bind(wx.EVT_BUTTON, self._on_edit_phase)
        self.btn_remove_phase.Bind(wx.EVT_BUTTON, self._on_remove_phase)
        btn_row.Add(self.btn_add_phase, 0, wx.RIGHT, 4)
        btn_row.Add(self.btn_edit_phase, 0, wx.RIGHT, 4)
        btn_row.Add(self.btn_remove_phase, 0)
        self.btn_add_phase.Hide()
        self.btn_edit_phase.Hide()
        self.btn_remove_phase.Hide()
        main.Add(btn_row, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
        main.AddSpacer(4)

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

    def get_mission_profile(self) -> MissionProfile:
        if self._mission_profile and not self._mission_profile.is_single_phase:
            return self._mission_profile
        return MissionProfile.single_phase(
            years=self.years.GetValue(),
            n_cycles=self.cycles.GetValue(),
            delta_t=self.dt.GetValue(),
            tau_on=self.tau_on.GetValue(),
        )

    def set_mission_profile(self, profile: MissionProfile):
        self._mission_profile = profile
        if profile and not profile.is_single_phase:
            self.years.SetValue(int(profile.mission_years or 5))
            self._show_multi_phase()
            self._refresh_phase_list()
            prof_name = getattr(profile, "name", None)
            for i, name in enumerate(["(Single-Phase)"] + sorted(MISSION_TEMPLATES.keys())):
                if name == prof_name:
                    self.template_combo.SetSelection(i)
                    break
        else:
            self._show_single_phase()
            self.template_combo.SetSelection(0)

    def _on_template_select(self, event):
        sel = self.template_combo.GetStringSelection()
        if sel == "(Single-Phase)":
            self._mission_profile = None
            self._show_single_phase()
        elif sel in MISSION_TEMPLATES:
            self._mission_profile = MISSION_TEMPLATES[sel]
            self.years.SetValue(int(self._mission_profile.mission_years))
            self._show_multi_phase()
            self._refresh_phase_list()
        self._on_change(event)

    def _show_multi_phase(self):
        self.phase_list.Show()
        self.btn_add_phase.Show()
        self.btn_edit_phase.Show()
        self.btn_remove_phase.Show()
        n = len(self._mission_profile.phases) if self._mission_profile else 0
        self.phase_label.SetLabel(f"Phases: ({n} defined)")
        self.GetSizer().Layout()
        self.GetParent().Layout()

    def _show_single_phase(self):
        self.phase_list.Hide()
        self.btn_add_phase.Hide()
        self.btn_edit_phase.Hide()
        self.btn_remove_phase.Hide()
        self.phase_label.SetLabel("Phases: (single-phase mode)")
        self.GetSizer().Layout()
        self.GetParent().Layout()

    def _refresh_phase_list(self):
        self.phase_list.DeleteAllItems()
        if not self._mission_profile:
            return
        for i, p in enumerate(self._mission_profile.phases):
            idx = self.phase_list.InsertItem(i, (p.name or "Phase")[:15])
            self.phase_list.SetItem(idx, 1, f"{(p.duration_frac or 0)*100:.0f}")
            self.phase_list.SetItem(idx, 2, f"{p.t_ambient or 0:.0f}")
            tj = p.t_junction if p.t_junction is not None else p.t_ambient
            self.phase_list.SetItem(idx, 3, f"{tj:.0f}")
            self.phase_list.SetItem(idx, 4, f"{p.n_cycles or 0}")
            self.phase_list.SetItem(idx, 5, f"{p.delta_t or 0:.1f}")
            self.phase_list.SetItem(idx, 6, f"{p.tau_on or 0:.2f}")

    def _on_add_phase(self, event):
        dlg = MissionPhaseDialog(self, title="Add Mission Phase")
        if dlg.ShowModal() == wx.ID_OK:
            phase = dlg.get_phase()
            if self._mission_profile is None:
                self._mission_profile = MissionProfile(
                    name="Custom", mission_years=self.years.GetValue(), phases=[])
            self._mission_profile.phases.append(phase)
            total = sum(p.duration_frac for p in self._mission_profile.phases)
            if total > 0:
                for p in self._mission_profile.phases:
                    p.duration_frac = p.duration_frac / total
            self._refresh_phase_list()
            self._on_change(event)
        dlg.Destroy()

    def _on_edit_phase(self, event):
        sel = self.phase_list.GetFirstSelected()
        if sel < 0 or not self._mission_profile:
            return
        dlg = MissionPhaseDialog(self, phase=self._mission_profile.phases[sel],
                                 title="Edit Mission Phase")
        if dlg.ShowModal() == wx.ID_OK:
            self._mission_profile.phases[sel] = dlg.get_phase()
            self._refresh_phase_list()
            self._on_change(event)
        dlg.Destroy()

    def _on_remove_phase(self, event):
        sel = self.phase_list.GetFirstSelected()
        if sel < 0 or not self._mission_profile:
            return
        self._mission_profile.phases.pop(sel)
        if len(self._mission_profile.phases) == 0:
            self._mission_profile = None
            self._show_single_phase()
            self.template_combo.SetSelection(0)
        else:
            total = sum(p.duration_frac for p in self._mission_profile.phases)
            if total > 0:
                for p in self._mission_profile.phases:
                    p.duration_frac = p.duration_frac / total
            self._refresh_phase_list()
        self._on_change(event)

    def _on_change(self, event):
        if self.on_change:
            self.on_change()

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
        self.list.InsertColumn(3, "L (FIT)", width=70)
        self.list.InsertColumn(4, "R", width=70)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_dclick)
        sizer.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        self.summary = wx.StaticText(self, label="")
        self.summary.SetForegroundColour(Colors.TEXT_SECONDARY)
        sizer.Add(self.summary, 0, wx.ALL, 10)

        self.SetSizer(sizer)
        self.SetupScrolling(scroll_x=False)

    def set_data(self, sheet: str, components: list, total_lam: float, r: float):
        self.current_sheet = sheet
        label = sheet.rstrip("/").split("/")[-1] or "Root"
        self.header.SetLabel(f"Components -- {label}")
        self.list.DeleteAllItems()
        for i, c in enumerate(components):
            idx = self.list.InsertItem(i, c.get("ref", "?"))
            self.list.SetItem(idx, 1, (c.get("value", "") or "")[:20])
            self.list.SetItem(idx, 2, (c.get("class", "") or "")[:20])
            lam = float(c.get("lambda", 0) or 0)
            self.list.SetItem(idx, 3, f"{lam*1e9:.1f}")
            self.list.SetItem(idx, 4, f"{float(c.get('r', 1) or 1):.5f}")
        self.summary.SetLabel(f"Sheet: L = {total_lam*1e9:.1f} FIT  R = {r:.5f}")
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
