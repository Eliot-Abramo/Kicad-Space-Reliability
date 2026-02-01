"""
Component Editor Dialog
=======================
UI for editing reliability fields with new IEC TR 62380 features:
- Configurable EOS (interface type selection)
- Working time ratio (τ_on)
- Thermal expansion materials
"""

import wx
import wx.lib.scrolledpanel as scrolled
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

try:
    from .reliability_math import (
        get_component_types,
        get_field_definitions,
        calculate_component_lambda,
        reliability_from_lambda,
        INTERFACE_EOS_VALUES,
        THERMAL_EXPANSION_SUBSTRATE,
        IC_TYPE_CHOICES,
        IC_PACKAGE_CHOICES,
        DIODE_BASE_RATES,
        TRANSISTOR_BASE_RATES,
        CAPACITOR_PARAMS,
        RESISTOR_PARAMS,
        INDUCTOR_PARAMS,
        MISC_COMPONENT_RATES,
    )
except ImportError:  # pragma: no cover
    from reliability_math import (
        get_component_types,
        get_field_definitions,
        calculate_component_lambda,
        reliability_from_lambda,
        INTERFACE_EOS_VALUES,
        THERMAL_EXPANSION_SUBSTRATE,
        IC_TYPE_CHOICES,
        IC_PACKAGE_CHOICES,
        DIODE_BASE_RATES,
        TRANSISTOR_BASE_RATES,
        CAPACITOR_PARAMS,
        RESISTOR_PARAMS,
        INDUCTOR_PARAMS,
        MISC_COMPONENT_RATES,
    )


@dataclass
class ComponentData:
    reference: str
    value: str
    component_type: str
    fields: Dict[str, Any]


def classify_component(
    reference: str, value: str, existing_fields: Dict[str, str] = None
) -> str:
    """Classify component from reference designator."""
    ref = reference.upper()
    if existing_fields and existing_fields.get("Reliability_Class"):
        rc = existing_fields["Reliability_Class"].lower()
        if "ic" in rc or "integrated" in rc:
            return "Integrated Circuit"
        if "diode" in rc:
            return "Diode"
        if "transistor" in rc or "mosfet" in rc:
            return "Transistor"
        if "capacitor" in rc:
            return "Capacitor"
        if "resistor" in rc:
            return "Resistor"
        if "inductor" in rc:
            return "Inductor/Transformer"

    if ref.startswith("R"):
        return "Resistor"
    elif ref.startswith("C"):
        return "Capacitor"
    elif ref.startswith("L"):
        return "Inductor/Transformer"
    elif ref.startswith("D"):
        return "Diode"
    elif ref.startswith("Q") or ref.startswith("T"):
        return "Transistor"
    elif ref.startswith("U") or ref.startswith("IC"):
        return "Integrated Circuit"
    elif ref.startswith("Y") or ref.startswith("X"):
        return "Miscellaneous"
    elif ref.startswith("J") or ref.startswith("P"):
        return "Miscellaneous"
    return "Miscellaneous"


class FieldEditorPanel(scrolled.ScrolledPanel):
    """Panel for editing component fields with appropriate controls."""

    def __init__(
        self,
        parent,
        component_type: str,
        initial_values: Dict[str, Any] = None,
        on_change: Callable = None,
    ):
        super().__init__(parent, style=wx.VSCROLL)
        self.component_type = component_type
        self.field_controls = {}
        self.on_change = on_change
        self._create_ui(initial_values or {})
        self.SetupScrolling(scroll_x=False)

    def _create_ui(self, initial_values: Dict[str, Any]):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        fields = get_field_definitions(self.component_type)

        # Group fields
        required = {k: v for k, v in fields.items() if v.get("required")}
        optional = {k: v for k, v in fields.items() if not v.get("required")}

        if required:
            box = wx.StaticBox(self, label="Required Fields")
            box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            for name, defn in required.items():
                ctrl = self._create_field(name, defn, initial_values)
                box_sizer.Add(ctrl, 0, wx.EXPAND | wx.ALL, 4)
            main_sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        if optional:
            box = wx.StaticBox(self, label="Optional / Environmental")
            box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            for name, defn in optional.items():
                ctrl = self._create_field(name, defn, initial_values)
                box_sizer.Add(ctrl, 0, wx.EXPAND | wx.ALL, 4)
            main_sizer.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(main_sizer)

    def _create_field(self, name: str, defn: Dict, initial: Dict[str, Any]) -> wx.Sizer:
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Label
        label_text = name.replace("_", " ").title()
        if defn.get("required"):
            label_text += " *"
        label = wx.StaticText(self, label=label_text)
        label.SetFont(label.GetFont().Bold())
        sizer.Add(label, 0, wx.LEFT, 2)

        # Help text
        help_text = defn.get("help", "")
        if help_text:
            help_lbl = wx.StaticText(self, label=help_text)
            help_lbl.SetForegroundColour(wx.Colour(100, 100, 100))
            font = help_lbl.GetFont()
            font.SetPointSize(font.GetPointSize() - 1)
            help_lbl.SetFont(font)
            sizer.Add(help_lbl, 0, wx.LEFT | wx.BOTTOM, 2)

        # Control
        ftype = defn.get("type", "text")
        default = defn.get("default")
        value = initial.get(name, default)

        if ftype == "choice":
            choices = defn.get("choices", [])
            ctrl = wx.ComboBox(
                self, choices=choices, style=wx.CB_DROPDOWN | wx.CB_READONLY
            )
            if value and value in choices:
                ctrl.SetValue(value)
            elif choices:
                ctrl.SetValue(choices[0])
            ctrl.Bind(wx.EVT_COMBOBOX, self._on_change)
        elif ftype == "bool":
            ctrl = wx.CheckBox(self, label="Yes")
            ctrl.SetValue(bool(value))
            ctrl.Bind(wx.EVT_CHECKBOX, self._on_change)
        elif ftype == "int":
            ctrl = wx.SpinCtrl(self, min=0, max=10000000, initial=int(value or 0))
            ctrl.Bind(wx.EVT_SPINCTRL, self._on_change)
        elif ftype == "float":
            ctrl = wx.SpinCtrlDouble(
                self,
                min=defn.get("min", 0),
                max=defn.get("max", 1000),
                initial=float(value or 0),
                inc=0.1,
            )
            ctrl.SetDigits(3)
            ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        else:
            ctrl = wx.TextCtrl(self, value=str(value or ""))
            ctrl.Bind(wx.EVT_TEXT, self._on_change)

        sizer.Add(ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        self.field_controls[name] = (ctrl, ftype, defn)
        return sizer

    def _on_change(self, event):
        if self.on_change:
            self.on_change()
        event.Skip()

    def get_values(self) -> Dict[str, Any]:
        values = {}
        for name, (ctrl, ftype, defn) in self.field_controls.items():
            try:
                if ftype == "choice":
                    values[name] = ctrl.GetValue()
                elif ftype == "bool":
                    values[name] = ctrl.GetValue()
                elif ftype == "int":
                    values[name] = ctrl.GetValue()
                elif ftype == "float":
                    values[name] = ctrl.GetValue()
                else:
                    values[name] = ctrl.GetValue()
            except:
                values[name] = defn.get("default")
        return values

    def set_component_type(
        self, component_type: str, initial_values: Dict[str, Any] = None
    ):
        self.component_type = component_type
        self.field_controls.clear()
        self.DestroyChildren()
        self._create_ui(initial_values or {})
        self.SetupScrolling(scroll_x=False)
        self.Layout()


class ComponentEditorDialog(wx.Dialog):
    """Dialog for editing reliability fields on a single component."""

    def __init__(self, parent, component: ComponentData, mission_hours: float = 43800):
        super().__init__(
            parent,
            title=f"Edit: {component.reference}",
            size=(520, 650),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.component = component
        self.mission_hours = mission_hours
        self.result_fields = None
        self._create_ui()
        self._update_preview()
        self.Centre()

    def _create_ui(self):
        panel = wx.Panel(self)
        main = wx.BoxSizer(wx.VERTICAL)

        # Header
        header = wx.BoxSizer(wx.HORIZONTAL)
        ref_lbl = wx.StaticText(panel, label=f"Reference: {self.component.reference}")
        ref_lbl.SetFont(ref_lbl.GetFont().Bold())
        header.Add(ref_lbl, 0, wx.ALL, 5)
        val_lbl = wx.StaticText(panel, label=f"Value: {self.component.value}")
        header.Add(val_lbl, 0, wx.ALL, 5)
        main.Add(header, 0, wx.EXPAND)

        # Type selector
        type_sizer = wx.BoxSizer(wx.HORIZONTAL)
        type_sizer.Add(
            wx.StaticText(panel, label="Component Type:"),
            0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL,
            5,
        )
        self.type_combo = wx.ComboBox(
            panel, choices=get_component_types(), style=wx.CB_READONLY
        )
        self.type_combo.SetValue(self.component.component_type)
        self.type_combo.Bind(wx.EVT_COMBOBOX, self._on_type_change)
        type_sizer.Add(self.type_combo, 1, wx.ALL | wx.EXPAND, 5)
        main.Add(type_sizer, 0, wx.EXPAND)

        main.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 5)

        # Field editor
        self.field_panel = FieldEditorPanel(
            panel,
            self.component.component_type,
            self.component.fields,
            on_change=self._update_preview,
        )
        main.Add(self.field_panel, 1, wx.EXPAND | wx.ALL, 5)

        # Preview
        preview_box = wx.StaticBox(panel, label="Calculated Results")
        preview_sizer = wx.StaticBoxSizer(preview_box, wx.VERTICAL)
        self.preview = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 80)
        )
        self.preview.SetBackgroundColour(wx.Colour(245, 245, 245))
        preview_sizer.Add(self.preview, 1, wx.EXPAND | wx.ALL, 5)
        main.Add(preview_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(panel, wx.ID_OK, "Apply")
        ok_btn.Bind(wx.EVT_BUTTON, self._on_ok)
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(wx.Button(panel, wx.ID_CANCEL, "Cancel"))
        btn_sizer.Realize()
        main.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main)

    def _on_type_change(self, event):
        self.field_panel.set_component_type(self.type_combo.GetValue(), {})
        self._update_preview()

    def _update_preview(self):
        try:
            ct = self.type_combo.GetValue()
            params = self.field_panel.get_values()
            result = calculate_component_lambda(ct, params)
            lam = result.get("lambda_total", 0)
            fit = result.get("fit_total", lam * 1e9)
            r = reliability_from_lambda(lam, self.mission_hours)
            mttf_h = 1 / lam if lam > 0 else float("inf")
            mttf_y = mttf_h / 8760

            text = f"λ (failure rate): {fit:.2f} FIT ({lam:.2e} /h)\n"
            text += f"R({self.mission_hours/8760:.1f} yr): {r:.6f} ({r*100:.4f}%)\n"
            text += f"MTTF: {mttf_y:.1f} years"

            # Show EOS contribution if applicable
            if result.get("lambda_eos", 0) > 0:
                eos_fit = result["lambda_eos"] * 1e9
                text += f"\n(includes {eos_fit:.1f} FIT from EOS)"

            self.preview.SetValue(text)
        except Exception as e:
            self.preview.SetValue(f"Error: {e}")

    def _on_ok(self, event):
        self.result_fields = self.field_panel.get_values()
        self.result_fields["_component_type"] = self.type_combo.GetValue()
        self.EndModal(wx.ID_OK)

    def get_result(self) -> Optional[Dict[str, Any]]:
        return self.result_fields


class BatchComponentEditorDialog(wx.Dialog):
    """Dialog for editing multiple components at once."""

    def __init__(
        self, parent, components: List[ComponentData], mission_hours: float = 43800
    ):
        super().__init__(
            parent,
            title="Batch Component Editor",
            size=(950, 700),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.components = components
        self.mission_hours = mission_hours
        self.results = {}
        self._create_ui()
        self.Centre()

    def _create_ui(self):
        panel = wx.Panel(self)
        main = wx.BoxSizer(wx.HORIZONTAL)

        # Left: component list
        left = wx.Panel(panel)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(wx.StaticText(left, label="Components:"), 0, wx.ALL, 5)

        self.list = wx.ListCtrl(left, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.list.InsertColumn(0, "Ref", width=60)
        self.list.InsertColumn(1, "Value", width=90)
        self.list.InsertColumn(2, "Type", width=130)
        self.list.InsertColumn(3, "λ (FIT)", width=80)

        for i, comp in enumerate(self.components):
            self.list.InsertItem(i, comp.reference)
            self.list.SetItem(i, 1, comp.value or "")
            self.list.SetItem(i, 2, comp.component_type)
            try:
                result = calculate_component_lambda(comp.component_type, comp.fields)
                self.list.SetItem(i, 3, f"{result.get('fit_total', 0):.1f}")
            except:
                self.list.SetItem(i, 3, "?")

        self.list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_select)
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit)
        left_sizer.Add(self.list, 1, wx.EXPAND | wx.ALL, 5)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        edit_btn = wx.Button(left, label="Edit Selected...")
        edit_btn.Bind(wx.EVT_BUTTON, self._on_edit)
        btn_row.Add(edit_btn, 1, wx.ALL, 3)
        auto_btn = wx.Button(left, label="Auto-Classify All")
        auto_btn.Bind(wx.EVT_BUTTON, self._on_auto)
        btn_row.Add(auto_btn, 1, wx.ALL, 3)
        left_sizer.Add(btn_row, 0, wx.EXPAND)

        left.SetSizer(left_sizer)
        main.Add(left, 1, wx.EXPAND | wx.ALL, 5)

        # Right: quick edit panel
        right = wx.Panel(panel)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        right_sizer.Add(wx.StaticText(right, label="Quick Edit:"), 0, wx.ALL, 5)

        type_row = wx.BoxSizer(wx.HORIZONTAL)
        type_row.Add(
            wx.StaticText(right, label="Type:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3
        )
        self.type_combo = wx.ComboBox(
            right, choices=get_component_types(), style=wx.CB_READONLY
        )
        self.type_combo.Bind(wx.EVT_COMBOBOX, self._on_quick_type)
        type_row.Add(self.type_combo, 1, wx.ALL, 3)
        right_sizer.Add(type_row, 0, wx.EXPAND)

        self.field_panel = FieldEditorPanel(right, "Resistor", {}, None)
        right_sizer.Add(self.field_panel, 1, wx.EXPAND | wx.ALL, 5)

        apply_btn = wx.Button(right, label="Apply to Selected")
        apply_btn.Bind(wx.EVT_BUTTON, self._on_apply_quick)
        right_sizer.Add(apply_btn, 0, wx.EXPAND | wx.ALL, 5)

        right.SetSizer(right_sizer)
        main.Add(right, 1, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(main)

        # Dialog buttons
        dlg_sizer = wx.BoxSizer(wx.VERTICAL)
        dlg_sizer.Add(panel, 1, wx.EXPAND)
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(self, wx.ID_OK, "Save All")
        ok_btn.Bind(wx.EVT_BUTTON, self._on_ok)
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(wx.Button(self, wx.ID_CANCEL, "Cancel"))
        btn_sizer.Realize()
        dlg_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(dlg_sizer)

        if self.components:
            self.list.Select(0)
            self._load_component(0)

    def _on_select(self, event):
        self._load_component(event.GetIndex())

    def _load_component(self, idx: int):
        if 0 <= idx < len(self.components):
            comp = self.components[idx]
            fields = self.results.get(comp.reference, comp.fields)
            ct = fields.get("_component_type", comp.component_type)
            self.type_combo.SetValue(ct)
            self.field_panel.set_component_type(ct, fields)

    def _on_quick_type(self, event):
        self.field_panel.set_component_type(self.type_combo.GetValue(), {})

    def _on_apply_quick(self, event):
        idx = self.list.GetFirstSelected()
        if idx < 0:
            return
        comp = self.components[idx]
        fields = self.field_panel.get_values()
        fields["_component_type"] = self.type_combo.GetValue()
        self.results[comp.reference] = fields
        comp.component_type = self.type_combo.GetValue()
        comp.fields = fields
        self.list.SetItem(idx, 2, comp.component_type)
        try:
            result = calculate_component_lambda(comp.component_type, fields)
            self.list.SetItem(idx, 3, f"{result.get('fit_total', 0):.1f}")
        except:
            self.list.SetItem(idx, 3, "?")

    def _on_edit(self, event):
        idx = self.list.GetFirstSelected()
        if idx < 0:
            wx.MessageBox(
                "Select a component first.", "No Selection", wx.OK | wx.ICON_INFORMATION
            )
            return
        comp = self.components[idx]
        fields = self.results.get(comp.reference, comp.fields)
        ct = fields.get("_component_type", comp.component_type)
        edit_comp = ComponentData(
            reference=comp.reference, value=comp.value, component_type=ct, fields=fields
        )
        dlg = ComponentEditorDialog(self, edit_comp, self.mission_hours)
        if dlg.ShowModal() == wx.ID_OK:
            result = dlg.get_result()
            if result:
                self.results[comp.reference] = result
                comp.component_type = result.get("_component_type", comp.component_type)
                comp.fields = result
                self.list.SetItem(idx, 2, comp.component_type)
                try:
                    calc = calculate_component_lambda(comp.component_type, result)
                    self.list.SetItem(idx, 3, f"{calc.get('fit_total', 0):.1f}")
                except:
                    self.list.SetItem(idx, 3, "?")
                self._load_component(idx)
        dlg.Destroy()

    def _on_auto(self, event):
        for i, comp in enumerate(self.components):
            if comp.reference not in self.results:
                new_type = classify_component(comp.reference, comp.value, comp.fields)
                comp.component_type = new_type
                self.list.SetItem(i, 2, new_type)
                try:
                    result = calculate_component_lambda(new_type, comp.fields)
                    self.list.SetItem(i, 3, f"{result.get('fit_total', 0):.1f}")
                except:
                    self.list.SetItem(i, 3, "?")

    def _on_ok(self, event):
        for comp in self.components:
            if comp.reference not in self.results:
                fields = comp.fields.copy()
                fields["_component_type"] = comp.component_type
                self.results[comp.reference] = fields
        self.EndModal(wx.ID_OK)

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        return self.results


class QuickReferenceDialog(wx.Dialog):
    """IEC TR 62380 quick reference."""

    def __init__(self, parent):
        super().__init__(
            parent,
            title="IEC TR 62380 Quick Reference",
            size=(650, 550),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        nb = wx.Notebook(self)

        # Overview tab
        overview = wx.TextCtrl(nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        overview.SetValue(
            """IEC TR 62380 - Reliability Data Handbook
=========================================

Key Concepts:
• λ (Lambda): Failure rate in FIT (Failures In Time = failures per 10^9 hours)
• R(t): Reliability = probability of survival = exp(-λ × t)
• MTTF: Mean Time To Failure = 1/λ

General Model:
λ_component = (λ_die + λ_package + λ_EOS) × 10^-9 /h

Temperature Factor (Arrhenius):
π_t = exp(Ea × (1/T_ref - 1/(273+T_j)))

Thermal Cycling Factor π_n:
• n ≤ 8760: π_n = n^0.76
• n > 8760: π_n = 1.7 × n^0.6

Working Time Ratio (τ_on):
• Scales die contribution for duty-cycled operation
• τ_on = 1.0 for continuous operation
• τ_on = 0.5 for 50% duty cycle

EOS (Electrical Overstress):
• Interface circuits add λ_EOS based on environment type
• Computer: 10 FIT, Switching: 15 FIT
• Telecoms transmitting/access/subscriber cards: 40 FIT; Subscriber equipment: 70 FIT
• Railways, payphone: 100 FIT; Civilian avionics (on board calculators): 20 FIT
• Voltage supply, converters: 40 FIT; Non interfaces: 0 FIT
"""
        )
        nb.AddPage(overview, "Overview")

        # EOS tab
        eos = wx.TextCtrl(nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        eos_text = "EOS (Electrical Overstress) Values\n" + "=" * 40 + "\n\n"
        eos_text += f"{'Interface Type':<25} {'λ_EOS (FIT)':<15}\n"
        eos_text += "-" * 40 + "\n"
        for name, vals in INTERFACE_EOS_VALUES.items():
            eos_text += f"{name:<25} {vals['l_eos']:<15}\n"
        eos.SetValue(eos_text)
        nb.AddPage(eos, "EOS Values")

        # Materials tab
        mat = wx.TextCtrl(nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        mat_text = "Thermal Expansion Coefficients (ppm/°C)\n" + "=" * 45 + "\n\n"
        mat_text += "SUBSTRATES:\n"
        for name, cte in THERMAL_EXPANSION_SUBSTRATE.items():
            mat_text += f"  {name:<30} α = {cte}\n"
        mat_text += "\nPACKAGES:\n"
        mat_text += "  Plastic (SOIC, QFP, BGA)       α = 21.5\n"
        mat_text += "  Ceramic (CQFP, CPGA)           α = 6.5\n"
        mat_text += "  Metal Can (TO)                  α = 17.0\n"
        mat.SetValue(mat_text)
        nb.AddPage(mat, "Materials")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(nb, 1, wx.EXPAND | wx.ALL, 5)
        close_btn = wx.Button(self, wx.ID_CLOSE, "Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CLOSE))
        sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        self.SetSizer(sizer)
        self.Centre()
