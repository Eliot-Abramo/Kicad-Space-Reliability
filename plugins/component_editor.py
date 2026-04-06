"""
Component Editor Dialog
=======================
UI for editing reliability fields per IEC TR 62380:
- Configurable EOS (interface type selection)
- Working time ratio (tau_on)
- Thermal expansion materials

Author:  Eliot Abramo
"""

import wx
import wx.lib.scrolledpanel as scrolled
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

from .reliability_math import (
    get_component_types, get_field_definitions, calculate_component_lambda,
    reliability_from_lambda, INTERFACE_EOS_VALUES, THERMAL_EXPANSION_SUBSTRATE,
    IC_TYPE_CHOICES, IC_PACKAGE_CHOICES, DIODE_BASE_RATES, TRANSISTOR_BASE_RATES,
    CAPACITOR_PARAMS, RESISTOR_PARAMS, INDUCTOR_PARAMS, MISC_COMPONENT_RATES,
    OPTOCOUPLER_BASE_RATES, THYRISTOR_BASE_RATES, RELAY_PARAMS,
    CONNECTOR_PARAMS, PCB_SOLDER_PARAMS, DISCRETE_PACKAGE_TABLE,
    analyze_component_criticality,
)
try:
    from .classification import (
        ClassificationResult,
        classify_component,
        classify_component_info,
        classification_to_fields,
    )
except ImportError:
    from import_compat import ensure_plugin_paths

    ensure_plugin_paths()
    from classification import (
        ClassificationResult,
        classify_component,
        classify_component_info,
        classification_to_fields,
    )
try:
    from .ui.theme import PALETTE, apply_compact_fonts, apply_theme_recursively, dip_px, dip_size, style_list_ctrl, style_panel, style_text_like
    from .ui.windowing import center_dialog
except ImportError:
    from import_compat import ensure_plugin_paths

    ensure_plugin_paths()
    from theme import PALETTE, apply_compact_fonts, apply_theme_recursively, dip_px, dip_size, style_list_ctrl, style_panel, style_text_like
    from windowing import center_dialog

@dataclass
class ComponentData:
    reference: str
    value: str
    component_type: str
    fields: Dict[str, Any]


class FieldEditorPanel(scrolled.ScrolledPanel):
    """Panel for editing component fields with appropriate controls."""
    
    def __init__(self, parent, component_type: str, initial_values: Dict[str, Any] = None,
                 on_change: Callable = None):
        super().__init__(parent, style=wx.VSCROLL)
        self.component_type = component_type
        self.field_controls = {}
        self.on_change = on_change
        style_panel(self, PALETTE.card_bg)
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
        apply_theme_recursively(self, background=PALETTE.card_bg)
    
    def _create_field(self, name: str, defn: Dict, initial: Dict[str, Any]) -> wx.Sizer:
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Label
        label_text = name.replace("_", " ").title()
        if defn.get("required"): label_text += " *"
        label = wx.StaticText(self, label=label_text)
        label.SetFont(label.GetFont().Bold())
        sizer.Add(label, 0, wx.LEFT, 2)
        
        # Help text
        help_text = defn.get("help", "")
        if help_text:
            help_lbl = wx.StaticText(self, label=help_text)
            help_lbl.SetForegroundColour(PALETTE.text_muted)
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
            ctrl = wx.ComboBox(self, choices=choices, style=wx.CB_DROPDOWN | wx.CB_READONLY)
            if value and value in choices: ctrl.SetValue(value)
            elif choices: ctrl.SetValue(choices[0])
            ctrl.Bind(wx.EVT_COMBOBOX, self._on_change)
        elif ftype == "bool":
            ctrl = wx.CheckBox(self, label="Yes")
            ctrl.SetValue(bool(value))
            ctrl.Bind(wx.EVT_CHECKBOX, self._on_change)
        elif ftype == "int":
            ctrl = wx.SpinCtrl(self, min=0, max=10000000, initial=int(value or 0))
            ctrl.Bind(wx.EVT_SPINCTRL, self._on_change)
        elif ftype == "float":
            ctrl = wx.SpinCtrlDouble(self, min=defn.get("min", 0), max=defn.get("max", 1000),
                                      initial=float(value or 0), inc=0.1)
            ctrl.SetDigits(3)
            ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        else:
            ctrl = wx.TextCtrl(self, value=str(value or ""))
            ctrl.Bind(wx.EVT_TEXT, self._on_change)
        
        sizer.Add(ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 2)
        self.field_controls[name] = (ctrl, ftype, defn)
        return sizer
    
    def _on_change(self, event):
        if self.on_change: self.on_change()
        event.Skip()
    
    def get_values(self) -> Dict[str, Any]:
        values = {}
        for name, (ctrl, ftype, defn) in self.field_controls.items():
            try:
                if ftype == "choice": values[name] = ctrl.GetValue()
                elif ftype == "bool": values[name] = ctrl.GetValue()
                elif ftype == "int": values[name] = ctrl.GetValue()
                elif ftype == "float": values[name] = ctrl.GetValue()
                else: values[name] = ctrl.GetValue()
            except: values[name] = defn.get("default")
        return values
    
    def set_component_type(self, component_type: str, initial_values: Dict[str, Any] = None):
        self.component_type = component_type
        self.field_controls.clear()
        self.DestroyChildren()
        self._create_ui(initial_values or {})
        self.SetupScrolling(scroll_x=False)
        apply_theme_recursively(self, background=PALETTE.card_bg)
        self.Layout()


class ComponentEditorDialog(wx.Dialog):
    """Dialog for editing reliability fields on a single component."""
    
    def __init__(self, parent, component: ComponentData, mission_hours: float = 43800):
        super().__init__(parent, title=f"Edit: {component.reference}",
                        size=(520, 650), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.component = component
        self.mission_hours = mission_hours
        self.result_fields = None
        style_panel(self, PALETTE.panel_bg)
        self.SetSize(dip_size(self, 520, 650))
        self._create_ui()
        apply_compact_fonts(self)
        apply_theme_recursively(self, background=PALETTE.panel_bg)
        self._update_preview()
        wx.CallAfter(center_dialog, self, parent)
    
    def _create_ui(self):
        panel = wx.Panel(self)
        style_panel(panel, PALETTE.panel_bg)
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
        type_sizer.Add(wx.StaticText(panel, label="Component Type:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.type_combo = wx.ComboBox(panel, choices=get_component_types(), style=wx.CB_READONLY)
        self.type_combo.SetValue(self.component.component_type)
        self.type_combo.Bind(wx.EVT_COMBOBOX, self._on_type_change)
        type_sizer.Add(self.type_combo, 1, wx.ALL | wx.EXPAND, 5)
        main.Add(type_sizer, 0, wx.EXPAND)
        
        main.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 5)
        
        # --- Lambda Override Box ---
        ovr_box = wx.StaticBox(panel, label="Lambda Override (use measured / datasheet value)")
        ovr_sizer = wx.StaticBoxSizer(ovr_box, wx.VERTICAL)
        ovr_row = wx.BoxSizer(wx.HORIZONTAL)
        self.override_cb = wx.CheckBox(panel, label="Override calculated lambda with fixed value")
        self.override_cb.SetToolTip(
            "When checked, the plugin uses your specified FIT value directly\n"
            "instead of computing it from the IEC TR 62380 model.\n"
            "Use this for components with manufacturer-provided failure rates,\n"
            "FIDES data, or field-measured reliability figures.")
        init_override = self.component.fields.get("override_lambda")
        self.override_cb.SetValue(init_override is not None)
        self.override_cb.Bind(wx.EVT_CHECKBOX, self._on_override_toggle)
        ovr_row.Add(self.override_cb, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        ovr_sizer.Add(ovr_row, 0, wx.EXPAND)
        
        ovr_val_row = wx.BoxSizer(wx.HORIZONTAL)
        ovr_val_row.Add(wx.StaticText(panel, label="Fixed Lambda:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        self.override_val = wx.SpinCtrlDouble(panel, min=0, max=1e9, initial=float(init_override or 0),
                                               inc=0.1, size=dip_size(panel, 136, -1))
        self.override_val.SetDigits(3)
        self.override_val.Enable(init_override is not None)
        self.override_val.Bind(wx.EVT_SPINCTRLDOUBLE, lambda e: self._update_preview())
        ovr_val_row.Add(self.override_val, 0, wx.ALL, 4)
        ovr_val_row.Add(wx.StaticText(panel, label="FIT"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        ovr_help = wx.StaticText(panel, label="(1 FIT = 1 failure per 10^9 hours)")
        ovr_help.SetForegroundColour(PALETTE.text_muted)
        ovr_val_row.Add(ovr_help, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        ovr_sizer.Add(ovr_val_row, 0, wx.EXPAND)
        main.Add(ovr_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Field editor
        self.field_panel = FieldEditorPanel(panel, self.component.component_type,
                                            self.component.fields, on_change=self._update_preview)
        main.Add(self.field_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Preview
        preview_box = wx.StaticBox(panel, label="Calculated Results")
        preview_sizer = wx.StaticBoxSizer(preview_box, wx.VERTICAL)
        self.preview = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=wx.Size(-1, dip_px(panel, 96)))
        style_text_like(self.preview, read_only=True)
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
    
    def _on_override_toggle(self, event):
        enabled = self.override_cb.GetValue()
        self.override_val.Enable(enabled)
        self.field_panel.Enable(not enabled)
        self._update_preview()
    
    def _update_preview(self):
        try:
            # Check override first
            if self.override_cb.GetValue():
                fit = self.override_val.GetValue()
                lam = fit * 1e-9
                r = reliability_from_lambda(lam, self.mission_hours)
                mttf_h = 1 / lam if lam > 0 else float('inf')
                mttf_y = mttf_h / 8760
                text = f"[OVERRIDE] lambda = {fit:.2f} FIT ({lam:.2e} /h)\n"
                text += f"R({self.mission_hours/8760:.1f} yr): {r:.6f} ({r*100:.4f}%)\n"
                text += f"MTTF: {mttf_y:.1f} years"
                self.preview.SetValue(text)
                return

            ct = self.type_combo.GetValue()
            params = self.field_panel.get_values()
            result = calculate_component_lambda(ct, params)
            lam = result.get("lambda_total", 0)
            fit = result.get("fit_total", lam * 1e9)
            r = reliability_from_lambda(lam, self.mission_hours)
            mttf_h = 1 / lam if lam > 0 else float('inf')
            mttf_y = mttf_h / 8760
            
            text = f"lambda (failure rate): {fit:.2f} FIT ({lam:.2e} /h)\n"
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
        self.result_fields["_classification_source"] = "manual"
        self.result_fields["_classification_reason"] = "User reviewed or edited this component."
        self.result_fields["_classification_confidence"] = "manual"
        self.result_fields["_classification_review_required"] = False
        if self.override_cb.GetValue():
            self.result_fields["override_lambda"] = self.override_val.GetValue()
        else:
            self.result_fields["override_lambda"] = None
        self.EndModal(wx.ID_OK)
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        return self.result_fields


class BatchComponentEditorDialog(wx.Dialog):
    """Dialog for editing multiple components at once."""
    
    def __init__(self, parent, components: List[ComponentData], mission_hours: float = 43800):
        super().__init__(parent, title="Batch Component Editor",
                        size=(950, 700), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.components = components
        self.mission_hours = mission_hours
        self.results = {}
        style_panel(self, PALETTE.panel_bg)
        self.SetSize(dip_size(self, 950, 700))
        self._create_ui()
        apply_compact_fonts(self)
        apply_theme_recursively(self, background=PALETTE.panel_bg)
        wx.CallAfter(center_dialog, self, parent)
    
    def _create_ui(self):
        panel = wx.Panel(self)
        style_panel(panel, PALETTE.panel_bg)
        main = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left: component list
        left = wx.Panel(panel)
        style_panel(left, PALETTE.card_bg)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(wx.StaticText(left, label="Components:"), 0, wx.ALL, 5)

        self.classification_summary = wx.StaticText(
            left,
            label="Auto-classification will review obvious parts and flag ambiguous ones.",
        )
        self.classification_summary.SetForegroundColour(PALETTE.text_muted)
        left_sizer.Add(self.classification_summary, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.list = wx.ListCtrl(left, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        style_list_ctrl(self.list)
        self.list.InsertColumn(0, "Ref", width=dip_px(left, 68))
        self.list.InsertColumn(1, "Value", width=dip_px(left, 96))
        self.list.InsertColumn(2, "Type", width=dip_px(left, 138))
        self.list.InsertColumn(3, "Lambda (FIT)", width=dip_px(left, 96))
        self.list.InsertColumn(4, "Status", width=dip_px(left, 108))
        self.list.InsertColumn(5, "Why", width=dip_px(left, 280))

        for i, comp in enumerate(self.components):
            self._refresh_row(i, comp)
        
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
        review_btn = wx.Button(left, label="Next Flagged")
        review_btn.Bind(wx.EVT_BUTTON, self._on_next_flagged)
        btn_row.Add(review_btn, 1, wx.ALL, 3)
        left_sizer.Add(btn_row, 0, wx.EXPAND)
        
        left.SetSizer(left_sizer)
        main.Add(left, 1, wx.EXPAND | wx.ALL, 5)
        
        # Right: quick edit panel
        right = wx.Panel(panel)
        style_panel(right, PALETTE.card_bg)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        right_sizer.Add(wx.StaticText(right, label="Quick Edit:"), 0, wx.ALL, 5)
        
        type_row = wx.BoxSizer(wx.HORIZONTAL)
        type_row.Add(wx.StaticText(right, label="Type:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        self.type_combo = wx.ComboBox(right, choices=get_component_types(), style=wx.CB_READONLY)
        self.type_combo.Bind(wx.EVT_COMBOBOX, self._on_quick_type)
        type_row.Add(self.type_combo, 1, wx.ALL, 3)
        right_sizer.Add(type_row, 0, wx.EXPAND)
        
        # --- Lambda Override Box in Quick Edit ---
        ovr_box = wx.StaticBox(right, label="Lambda Override")
        ovr_sizer = wx.StaticBoxSizer(ovr_box, wx.VERTICAL)
        self.quick_override_cb = wx.CheckBox(right, label="Override with fixed FIT value")
        self.quick_override_cb.SetToolTip(
            "Use a manufacturer-provided or measured failure rate\n"
            "instead of the IEC TR 62380 model.")
        self.quick_override_cb.Bind(wx.EVT_CHECKBOX, self._on_quick_override_toggle)
        ovr_sizer.Add(self.quick_override_cb, 0, wx.ALL, 4)
        ovr_val_row = wx.BoxSizer(wx.HORIZONTAL)
        ovr_val_row.Add(wx.StaticText(right, label="Lambda:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        self.quick_override_val = wx.SpinCtrlDouble(right, min=0, max=1e9, initial=0, inc=0.1, size=dip_size(right, 120, -1))
        self.quick_override_val.SetDigits(3)
        self.quick_override_val.Enable(False)
        ovr_val_row.Add(self.quick_override_val, 0, wx.ALL, 4)
        ovr_val_row.Add(wx.StaticText(right, label="FIT"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        ovr_sizer.Add(ovr_val_row, 0, wx.EXPAND)
        right_sizer.Add(ovr_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        self.field_panel = FieldEditorPanel(right, "Resistor", {}, None)
        right_sizer.Add(self.field_panel, 1, wx.EXPAND | wx.ALL, 5)

        self.quick_classification = wx.StaticText(
            right,
            label="Select a component to see why it was classified this way.",
        )
        self.quick_classification.SetForegroundColour(PALETTE.text_muted)
        self.quick_classification.Wrap(dip_px(right, 380))
        right_sizer.Add(self.quick_classification, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        
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
        self._update_summary()

    def _classification_status(self, fields: Dict[str, Any]) -> tuple[str, wx.Colour]:
        if fields.get("_classification_review_required"):
            return "Review", PALETTE.warning
        confidence = str(fields.get("_classification_confidence", "manual")).lower()
        mapping = {
            "high": ("High", PALETTE.success),
            "medium": ("Medium", PALETTE.primary),
            "low": ("Low", PALETTE.warning),
            "manual": ("Manual", PALETTE.text_muted),
        }
        return mapping.get(confidence, ("Manual", PALETTE.text_muted))

    def _row_fields(self, comp: ComponentData) -> Dict[str, Any]:
        return self.results.get(comp.reference, comp.fields)

    def _refresh_row(self, idx: int, comp: ComponentData):
        fields = self._row_fields(comp)
        calc_fields = {k: v for k, v in fields.items() if not str(k).startswith("_")}
        if idx >= self.list.GetItemCount():
            self.list.InsertItem(idx, comp.reference)
        self.list.SetItem(idx, 0, comp.reference)
        self.list.SetItem(idx, 1, comp.value or "")
        self.list.SetItem(idx, 2, fields.get("_component_type", comp.component_type))
        ovr = fields.get("override_lambda")
        if ovr is not None:
            self.list.SetItem(idx, 3, f"{ovr:.1f}")
        else:
            try:
                result = calculate_component_lambda(fields.get("_component_type", comp.component_type), calc_fields)
                self.list.SetItem(idx, 3, f"{result.get('fit_total', 0):.1f}")
            except Exception:
                self.list.SetItem(idx, 3, "?")
        status, color = self._classification_status(fields)
        self.list.SetItem(idx, 4, status)
        self.list.SetItem(idx, 5, str(fields.get("_classification_reason", "User-set values")))
        self.list.SetItemTextColour(idx, color)

    def _update_summary(self):
        review = 0
        high = 0
        manual = 0
        for comp in self.components:
            fields = self._row_fields(comp)
            if fields.get("_classification_review_required"):
                review += 1
            elif fields.get("_classification_confidence") == "high":
                high += 1
            elif fields.get("_classification_source") == "manual":
                manual += 1
        total = len(self.components)
        self.classification_summary.SetLabel(
            f"Auto-classify status: {high}/{total} high-confidence, {review} need review, {manual} manual."
        )

    def _on_select(self, event):
        self._load_component(event.GetIndex())
    
    def _load_component(self, idx: int):
        if 0 <= idx < len(self.components):
            comp = self.components[idx]
            fields = self.results.get(comp.reference, comp.fields)
            ct = fields.get("_component_type", comp.component_type)
            self.type_combo.SetValue(ct)
            self.field_panel.set_component_type(ct, fields)
            self.quick_classification.SetLabel(
                f"Status: {self._classification_status(fields)[0]}  |  "
                f"Source: {fields.get('_classification_source', 'manual')}  |  "
                f"{fields.get('_classification_reason', 'User-set classification')}"
            )
            # Populate override controls
            ovr = fields.get("override_lambda")
            if ovr is not None:
                self.quick_override_cb.SetValue(True)
                self.quick_override_val.SetValue(float(ovr))
                self.quick_override_val.Enable(True)
                self.field_panel.Enable(False)
            else:
                self.quick_override_cb.SetValue(False)
                self.quick_override_val.SetValue(0.0)
                self.quick_override_val.Enable(False)
                self.field_panel.Enable(True)
    
    def _on_quick_type(self, event):
        self.field_panel.set_component_type(self.type_combo.GetValue(), {})
    
    def _on_quick_override_toggle(self, event):
        enabled = self.quick_override_cb.GetValue()
        self.quick_override_val.Enable(enabled)
        self.field_panel.Enable(not enabled)
    
    def _on_apply_quick(self, event):
        idx = self.list.GetFirstSelected()
        if idx < 0: return
        comp = self.components[idx]
        fields = self.field_panel.get_values()
        fields["_component_type"] = self.type_combo.GetValue()
        # Handle override from quick edit panel
        if self.quick_override_cb.GetValue():
            fields["override_lambda"] = self.quick_override_val.GetValue()
        else:
            fields["override_lambda"] = None
        self.results[comp.reference] = fields
        comp.component_type = self.type_combo.GetValue()
        comp.fields = fields
        fields["_classification_source"] = "manual"
        fields["_classification_reason"] = "User reviewed or edited this component."
        fields["_classification_confidence"] = "manual"
        fields["_classification_review_required"] = False
        self._refresh_row(idx, comp)
        self._update_summary()
    
    def _on_edit(self, event):
        idx = self.list.GetFirstSelected()
        if idx < 0:
            wx.MessageBox("Select a component first.", "No Selection", wx.OK | wx.ICON_INFORMATION)
            return
        comp = self.components[idx]
        fields = self.results.get(comp.reference, comp.fields)
        ct = fields.get("_component_type", comp.component_type)
        edit_comp = ComponentData(reference=comp.reference, value=comp.value,
                                  component_type=ct, fields=fields)
        dlg = ComponentEditorDialog(self, edit_comp, self.mission_hours)
        if dlg.ShowModal() == wx.ID_OK:
            result = dlg.get_result()
            if result:
                self.results[comp.reference] = result
                comp.component_type = result.get("_component_type", comp.component_type)
                comp.fields = result
                result["_classification_source"] = "manual"
                result["_classification_reason"] = "User reviewed or edited this component."
                result["_classification_confidence"] = "manual"
                result["_classification_review_required"] = False
                self._refresh_row(idx, comp)
                self._update_summary()
                self._load_component(idx)
        dlg.Destroy()

    def _on_auto(self, event):
        for i, comp in enumerate(self.components):
            result = classify_component_info(comp.reference, comp.value, comp.fields)
            comp.component_type = result.component_type
            comp.fields.update(classification_to_fields(result))
            self.results[comp.reference] = dict(comp.fields)
            self._refresh_row(i, comp)
        self._update_summary()

    def _on_next_flagged(self, event):
        start = self.list.GetFirstSelected()
        start = start + 1 if start >= 0 else 0
        for idx in range(start, len(self.components)):
            if self._row_fields(self.components[idx]).get("_classification_review_required"):
                self.list.Select(idx)
                self.list.Focus(idx)
                self._load_component(idx)
                return
        for idx in range(0, start):
            if self._row_fields(self.components[idx]).get("_classification_review_required"):
                self.list.Select(idx)
                self.list.Focus(idx)
                self._load_component(idx)
                return
        wx.MessageBox(
            "No flagged components remain. Auto-classification looks fully reviewed.",
            "Review Complete",
            wx.OK | wx.ICON_INFORMATION,
        )
    
    def _on_ok(self, event):
        # Auto-apply the currently visible quick-edit panel state for the
        # selected component so the user doesn't need to click "Apply" first.
        idx = self.list.GetFirstSelected()
        if 0 <= idx < len(self.components):
            comp = self.components[idx]
            fields = self.field_panel.get_values()
            fields["_component_type"] = self.type_combo.GetValue()
            if self.quick_override_cb.GetValue():
                fields["override_lambda"] = self.quick_override_val.GetValue()
            else:
                fields["override_lambda"] = None
            fields["_classification_source"] = "manual"
            fields["_classification_reason"] = "User reviewed or edited this component."
            fields["_classification_confidence"] = "manual"
            fields["_classification_review_required"] = False
            self.results[comp.reference] = fields

        # For components never explicitly edited, preserve their current fields
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
        super().__init__(parent, title="IEC TR 62380 Quick Reference",
                        size=(650, 550), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetSize(dip_size(self, 650, 550))
        style_panel(self, PALETTE.panel_bg)
        
        nb = wx.Notebook(self)
        nb.SetBackgroundColour(PALETTE.panel_bg)
        nb.SetForegroundColour(PALETTE.text)
        
        # Overview tab
        overview = wx.TextCtrl(nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        style_text_like(overview, read_only=True)
        overview.SetValue("""IEC TR 62380 - Reliability Data Handbook
=========================================

Key Concepts:
 Lambda: Failure rate in FIT (Failures In Time = failures per 10^9 hours)
 R(t): Reliability = probability of survival = exp(-lambda * t)
 MTTF: Mean Time To Failure = 1 / lambda

General Model:
lambda_component = (lambda_die + lambda_package + lambda_EOS) * 10^-9 /h

Temperature Factor (Arrhenius):
pi_t = exp(Ea * (1/T_ref - 1/(273 + T_j)))

Thermal Cycling Factor pi_n:
 n <= 8760: pi_n = n^0.76
 n >  8760: pi_n = 1.7 * n^0.6

Working Time Ratio (tau_on):
 Scales die contribution for duty-cycled operation
 tau_on = 1.0 for continuous operation
 tau_on = 0.5 for 50% duty cycle

EOS (Electrical Overstress):
 Interface circuits add lambda_EOS based on environment type
 Computer: 10 FIT, Telecom: 15-70 FIT, Avionics: 20 FIT
 Power Supply: 40 FIT, Space: 25-35 FIT
""")
        nb.AddPage(overview, "Overview")
        
        # EOS tab
        eos = wx.TextCtrl(nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        style_text_like(eos, read_only=True)
        eos_text = "EOS (Electrical Overstress) Values\n" + "="*40 + "\n\n"
        eos_text += f"{'Interface Type':<25} {'lambda_EOS (FIT)':<15}\n"
        eos_text += "-"*40 + "\n"
        for name, vals in INTERFACE_EOS_VALUES.items():
            eos_text += f"{name:<25} {vals['l_eos']:<15}\n"
        eos.SetValue(eos_text)
        nb.AddPage(eos, "EOS Values")
        
        # Materials tab
        mat = wx.TextCtrl(nb, style=wx.TE_MULTILINE | wx.TE_READONLY)
        style_text_like(mat, read_only=True)
        mat_text = "Thermal Expansion Coefficients (ppm/degC)\n" + "="*45 + "\n\n"
        mat_text += "SUBSTRATES:\n"
        for name, cte in THERMAL_EXPANSION_SUBSTRATE.items():
            mat_text += f"  {name:<30} alpha = {cte}\n"
        mat_text += "\nPACKAGES (alpha_C):\n"
        mat_text += "  Epoxy (Plastic package)          = 21.5\n"
        mat_text += "  Alumina (Ceramic package)        = 6.5\n"
        mat_text += "  Kovar (Metallic package)         = 5.0\n"
        mat_text += "  Bare Die / Flip Chip             = 2.6\n"
        mat.SetValue(mat_text)
        nb.AddPage(mat, "Materials")
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(nb, 1, wx.EXPAND | wx.ALL, 5)
        close_btn = wx.Button(self, wx.ID_CLOSE, "Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_CLOSE))
        sizer.Add(close_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        self.SetSizer(sizer)
        apply_compact_fonts(self)
        apply_theme_recursively(self, background=PALETTE.panel_bg)
        wx.CallAfter(center_dialog, self, parent)
