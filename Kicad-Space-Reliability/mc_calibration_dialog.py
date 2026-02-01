"""
Monte Carlo Calibration Dialog
==============================
UI for configuring Monte Carlo simulation parameters and uncertainty distributions.
"""

import wx
import wx.lib.scrolledpanel as scrolled
from typing import Dict, List, Optional, Callable
import json
from pathlib import Path

from .monte_carlo import MonteCarloConfig, ParameterUncertainty


class ParameterUncertaintyPanel(wx.Panel):
    """Panel for editing a single parameter's uncertainty settings."""

    DISTRIBUTIONS = ["lognormal", "normal", "uniform", "triangular"]
    DIST_LABELS = {
        "lognormal": "Lognormal (always positive)",
        "normal": "Normal (Gaussian)",
        "uniform": "Uniform (flat)",
        "triangular": "Triangular (peaked at nominal)",
    }

    def __init__(self, parent, param: ParameterUncertainty, on_change: Callable = None):
        super().__init__(parent)
        self.param = param
        self.on_change = on_change

        self._create_ui()

    def _create_ui(self):
        # Format display name
        display_name = self.param.name.replace("_", " ").title()

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Enable checkbox
        self.chk_enabled = wx.CheckBox(self, label=display_name)
        self.chk_enabled.SetValue(self.param.enabled)
        self.chk_enabled.SetMinSize((180, -1))
        self.chk_enabled.Bind(wx.EVT_CHECKBOX, self._on_change)
        main_sizer.Add(self.chk_enabled, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        # Distribution dropdown
        dist_choices = [self.DIST_LABELS[d] for d in self.DISTRIBUTIONS]
        self.cmb_dist = wx.ComboBox(self, choices=dist_choices, style=wx.CB_READONLY)
        dist_idx = (
            self.DISTRIBUTIONS.index(self.param.distribution)
            if self.param.distribution in self.DISTRIBUTIONS
            else 0
        )
        self.cmb_dist.SetSelection(dist_idx)
        self.cmb_dist.SetMinSize((180, -1))
        self.cmb_dist.Bind(wx.EVT_COMBOBOX, self._on_change)
        main_sizer.Add(self.cmb_dist, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        # CV input
        cv_sizer = wx.BoxSizer(wx.HORIZONTAL)
        cv_sizer.Add(
            wx.StaticText(self, label="CV:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3
        )
        self.spin_cv = wx.SpinCtrlDouble(
            self, min=0.01, max=1.0, initial=self.param.cv, inc=0.05
        )
        self.spin_cv.SetDigits(2)
        self.spin_cv.SetMinSize((80, -1))
        self.spin_cv.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        cv_sizer.Add(self.spin_cv, 0)
        main_sizer.Add(cv_sizer, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        # Range (for uniform/triangular)
        range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        range_sizer.Add(
            wx.StaticText(self, label="Range:"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            3,
        )
        self.spin_low = wx.SpinCtrlDouble(
            self, min=0.1, max=1.0, initial=self.param.low_mult, inc=0.1
        )
        self.spin_low.SetDigits(1)
        self.spin_low.SetMinSize((60, -1))
        self.spin_low.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        range_sizer.Add(self.spin_low, 0)
        range_sizer.Add(
            wx.StaticText(self, label="×  to"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
            5,
        )
        self.spin_high = wx.SpinCtrlDouble(
            self, min=1.0, max=5.0, initial=self.param.high_mult, inc=0.1
        )
        self.spin_high.SetDigits(1)
        self.spin_high.SetMinSize((60, -1))
        self.spin_high.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_change)
        range_sizer.Add(self.spin_high, 0)
        range_sizer.Add(
            wx.StaticText(self, label="×"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 3
        )
        main_sizer.Add(range_sizer, 0, wx.ALIGN_CENTER_VERTICAL)

        self.SetSizer(main_sizer)
        self._update_visibility()

    def _on_change(self, event):
        self._update_param()
        self._update_visibility()
        if self.on_change:
            self.on_change()

    def _update_visibility(self):
        # Show/hide range controls based on distribution
        dist = self.DISTRIBUTIONS[self.cmb_dist.GetSelection()]
        show_range = dist in ("uniform", "triangular")
        show_cv = dist in ("lognormal", "normal")

        self.spin_low.Show(show_range)
        self.spin_high.Show(show_range)
        self.spin_cv.GetParent().Layout()

    def _update_param(self):
        self.param.enabled = self.chk_enabled.GetValue()
        self.param.distribution = self.DISTRIBUTIONS[self.cmb_dist.GetSelection()]
        self.param.cv = self.spin_cv.GetValue()
        self.param.low_mult = self.spin_low.GetValue()
        self.param.high_mult = self.spin_high.GetValue()

    def get_param(self) -> ParameterUncertainty:
        self._update_param()
        return self.param


class MonteCarloCalibrationDialog(wx.Dialog):
    """
    Dialog for calibrating Monte Carlo simulation parameters.

    Features:
    - Number of simulations
    - Convergence settings
    - Per-parameter uncertainty configuration
    - Save/load presets
    """

    def __init__(self, parent, config: MonteCarloConfig = None):
        super().__init__(
            parent,
            title="Monte Carlo Calibration",
            size=(750, 650),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        self.config = config if config else MonteCarloConfig()
        self.param_panels: Dict[str, ParameterUncertaintyPanel] = {}

        self._create_ui()
        self.Centre()

    def _create_ui(self):
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        header = wx.StaticText(panel, label="Monte Carlo Simulation Settings")
        header.SetFont(header.GetFont().Bold().Scaled(1.2))
        main_sizer.Add(header, 0, wx.ALL, 15)

        # General settings
        gen_box = wx.StaticBox(panel, label="General Settings")
        gen_sizer = wx.StaticBoxSizer(gen_box, wx.VERTICAL)

        grid = wx.FlexGridSizer(3, 2, 8, 15)
        grid.AddGrowableCol(1, 1)

        # Simulations
        grid.Add(
            wx.StaticText(panel, label="Number of simulations:"),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.spin_n = wx.SpinCtrl(
            panel, min=100, max=100000, initial=self.config.n_simulations
        )
        self.spin_n.SetToolTip("More simulations = more accurate but slower")
        grid.Add(self.spin_n, 0, wx.EXPAND)

        # Base uncertainty
        grid.Add(
            wx.StaticText(panel, label="Default uncertainty (CV):"),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.spin_cv = wx.SpinCtrlDouble(
            panel, min=0.05, max=0.5, initial=self.config.base_uncertainty_cv, inc=0.05
        )
        self.spin_cv.SetDigits(2)
        self.spin_cv.SetToolTip(
            "Coefficient of variation for parameters without specific settings"
        )
        grid.Add(self.spin_cv, 0, wx.EXPAND)

        # Convergence threshold
        grid.Add(
            wx.StaticText(panel, label="Convergence threshold:"),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.spin_conv = wx.SpinCtrlDouble(
            panel,
            min=0.0001,
            max=0.01,
            initial=self.config.convergence_threshold,
            inc=0.0005,
        )
        self.spin_conv.SetDigits(4)
        self.spin_conv.SetToolTip(
            "Stop early if mean changes by less than this (relative)"
        )
        grid.Add(self.spin_conv, 0, wx.EXPAND)

        gen_sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 10)

        # Check convergence checkbox
        self.chk_convergence = wx.CheckBox(
            panel, label="Enable early convergence detection"
        )
        self.chk_convergence.SetValue(self.config.check_convergence)
        gen_sizer.Add(self.chk_convergence, 0, wx.LEFT | wx.BOTTOM, 10)

        main_sizer.Add(gen_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 15)

        # Parameter uncertainties
        param_box = wx.StaticBox(panel, label="Parameter Uncertainty Distributions")
        param_sizer = wx.StaticBoxSizer(param_box, wx.VERTICAL)

        # Help text
        help_text = wx.StaticText(
            panel,
            label=(
                "Configure how each parameter varies in the simulation. "
                "CV = coefficient of variation (std/mean). "
                "Disable parameters you know precisely."
            ),
        )
        help_text.SetForegroundColour(wx.Colour(100, 100, 100))
        help_text.Wrap(680)
        param_sizer.Add(help_text, 0, wx.ALL, 10)

        # Scrolled panel for parameters
        scroll = scrolled.ScrolledPanel(panel, style=wx.VSCROLL)
        scroll.SetBackgroundColour(wx.WHITE)
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        # Add parameter panels
        param_order = [
            "t_junction",
            "t_ambient",
            "n_cycles",
            "delta_t",
            "operating_power",
            "rated_power",
            "transistor_count",
            "voltage_stress_vds",
            "voltage_stress_vgs",
            "voltage_stress_vce",
            "ripple_ratio",
        ]

        for param_name in param_order:
            param = self.config.get_uncertainty(param_name)
            param_panel = ParameterUncertaintyPanel(scroll, param)
            self.param_panels[param_name] = param_panel
            scroll_sizer.Add(param_panel, 0, wx.EXPAND | wx.ALL, 5)
            scroll_sizer.Add(
                wx.StaticLine(scroll), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10
            )

        scroll.SetSizer(scroll_sizer)
        scroll.SetupScrolling(scroll_x=False)
        param_sizer.Add(scroll, 1, wx.EXPAND | wx.ALL, 5)

        main_sizer.Add(param_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 15)

        # Presets
        preset_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_conservative = wx.Button(panel, label="Conservative (Low Uncertainty)")
        btn_conservative.Bind(wx.EVT_BUTTON, self._on_preset_conservative)
        btn_conservative.SetToolTip("Low uncertainty - for well-characterized systems")
        preset_sizer.Add(btn_conservative, 0, wx.RIGHT, 10)

        btn_moderate = wx.Button(panel, label="Moderate (Default)")
        btn_moderate.Bind(wx.EVT_BUTTON, self._on_preset_moderate)
        btn_moderate.SetToolTip("Balanced uncertainty - typical engineering estimates")
        preset_sizer.Add(btn_moderate, 0, wx.RIGHT, 10)

        btn_aggressive = wx.Button(panel, label="Aggressive (High Uncertainty)")
        btn_aggressive.Bind(wx.EVT_BUTTON, self._on_preset_aggressive)
        btn_aggressive.SetToolTip("High uncertainty - for early design phases")
        preset_sizer.Add(btn_aggressive, 0, wx.RIGHT, 20)

        preset_sizer.AddStretchSpacer()

        btn_save = wx.Button(panel, label="Save Config...")
        btn_save.Bind(wx.EVT_BUTTON, self._on_save)
        preset_sizer.Add(btn_save, 0, wx.RIGHT, 5)

        btn_load = wx.Button(panel, label="Load Config...")
        btn_load.Bind(wx.EVT_BUTTON, self._on_load)
        preset_sizer.Add(btn_load, 0)

        main_sizer.Add(preset_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 15)

        # Buttons
        btn_sizer = wx.StdDialogButtonSizer()
        btn_ok = wx.Button(panel, wx.ID_OK, "Apply")
        btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)
        btn_sizer.AddButton(btn_ok)
        btn_cancel = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_sizer.AddButton(btn_cancel)
        btn_sizer.Realize()
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 15)

        panel.SetSizer(main_sizer)

    def _on_preset_conservative(self, event):
        """Low uncertainty preset."""
        self.spin_cv.SetValue(0.10)
        for name, panel in self.param_panels.items():
            panel.param.cv = 0.10
            panel.param.low_mult = 0.7
            panel.param.high_mult = 1.3
            panel.spin_cv.SetValue(0.10)
            panel.spin_low.SetValue(0.7)
            panel.spin_high.SetValue(1.3)

    def _on_preset_moderate(self, event):
        """Default uncertainty preset."""
        self.spin_cv.SetValue(0.20)
        defaults = MonteCarloConfig._default_uncertainties()
        for name, panel in self.param_panels.items():
            if name in defaults:
                d = defaults[name]
                panel.param.cv = d.cv
                panel.param.low_mult = d.low_mult
                panel.param.high_mult = d.high_mult
                panel.param.distribution = d.distribution
                panel.param.enabled = d.enabled
                # Update UI
                panel.spin_cv.SetValue(d.cv)
                panel.spin_low.SetValue(d.low_mult)
                panel.spin_high.SetValue(d.high_mult)
                panel.chk_enabled.SetValue(d.enabled)
                dist_idx = ParameterUncertaintyPanel.DISTRIBUTIONS.index(d.distribution)
                panel.cmb_dist.SetSelection(dist_idx)

    def _on_preset_aggressive(self, event):
        """High uncertainty preset."""
        self.spin_cv.SetValue(0.30)
        for name, panel in self.param_panels.items():
            panel.param.cv = 0.30
            panel.param.low_mult = 0.5
            panel.param.high_mult = 2.0
            panel.spin_cv.SetValue(0.30)
            panel.spin_low.SetValue(0.5)
            panel.spin_high.SetValue(2.0)

    def _on_save(self, event):
        """Save configuration to file."""
        dlg = wx.FileDialog(
            self,
            "Save Configuration",
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            config = self.get_config()
            try:
                with open(dlg.GetPath(), "w") as f:
                    json.dump(config.to_dict(), f, indent=2)
                wx.MessageBox(
                    "Configuration saved.", "Saved", wx.OK | wx.ICON_INFORMATION
                )
            except Exception as e:
                wx.MessageBox(f"Error saving: {e}", "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _on_load(self, event):
        """Load configuration from file."""
        dlg = wx.FileDialog(
            self,
            "Load Configuration",
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dlg.ShowModal() == wx.ID_OK:
            try:
                with open(dlg.GetPath(), "r") as f:
                    data = json.load(f)
                self.config = MonteCarloConfig.from_dict(data)
                self._update_ui_from_config()
                wx.MessageBox(
                    "Configuration loaded.", "Loaded", wx.OK | wx.ICON_INFORMATION
                )
            except Exception as e:
                wx.MessageBox(f"Error loading: {e}", "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _update_ui_from_config(self):
        """Update all UI elements from current config."""
        self.spin_n.SetValue(self.config.n_simulations)
        self.spin_cv.SetValue(self.config.base_uncertainty_cv)
        self.spin_conv.SetValue(self.config.convergence_threshold)
        self.chk_convergence.SetValue(self.config.check_convergence)

        for name, panel in self.param_panels.items():
            param = self.config.get_uncertainty(name)
            panel.param = param
            panel.chk_enabled.SetValue(param.enabled)
            panel.spin_cv.SetValue(param.cv)
            panel.spin_low.SetValue(param.low_mult)
            panel.spin_high.SetValue(param.high_mult)
            dist_idx = (
                ParameterUncertaintyPanel.DISTRIBUTIONS.index(param.distribution)
                if param.distribution in ParameterUncertaintyPanel.DISTRIBUTIONS
                else 0
            )
            panel.cmb_dist.SetSelection(dist_idx)

    def _on_ok(self, event):
        self.EndModal(wx.ID_OK)

    def get_config(self) -> MonteCarloConfig:
        """Get the configured MonteCarloConfig."""
        config = MonteCarloConfig(
            n_simulations=self.spin_n.GetValue(),
            base_uncertainty_cv=self.spin_cv.GetValue(),
            check_convergence=self.chk_convergence.GetValue(),
            convergence_threshold=self.spin_conv.GetValue(),
        )

        for name, panel in self.param_panels.items():
            config.parameter_uncertainties[name] = panel.get_param()

        return config


class MCProgressDialog(wx.Dialog):
    """Progress dialog for Monte Carlo with cancel button."""

    def __init__(self, parent, title="Monte Carlo Simulation"):
        super().__init__(
            parent, title=title, size=(400, 150), style=wx.DEFAULT_DIALOG_STYLE
        )

        self.cancelled = False

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.label = wx.StaticText(panel, label="Running Monte Carlo simulation...")
        sizer.Add(self.label, 0, wx.ALL, 15)

        self.gauge = wx.Gauge(panel, range=100)
        sizer.Add(self.gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)

        self.status = wx.StaticText(panel, label="0%")
        sizer.Add(self.status, 0, wx.ALL, 10)

        btn_cancel = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        sizer.Add(btn_cancel, 0, wx.ALIGN_CENTER | wx.BOTTOM, 15)

        panel.SetSizer(sizer)
        self.Centre()

    def _on_cancel(self, event):
        self.cancelled = True
        self.label.SetLabel("Cancelling...")

    def update(self, progress: float, status: str = None):
        """Update progress (0-100)."""
        self.gauge.SetValue(int(progress))
        self.status.SetLabel(f"{progress:.1f}%" + (f" - {status}" if status else ""))
        wx.Yield()
        return not self.cancelled


if __name__ == "__main__":
    app = wx.App()
    dlg = MonteCarloCalibrationDialog(None)
    if dlg.ShowModal() == wx.ID_OK:
        config = dlg.get_config()
        print("Config:", config.to_dict())
    dlg.Destroy()
    app.MainLoop()
