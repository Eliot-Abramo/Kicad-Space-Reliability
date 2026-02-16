"""
Unified Sensitivity & Uncertainty Analysis Dialog
==================================================
Single coherent UI: parameter selection, Monte Carlo, Sobol, what-if, report.
"""

import wx
import wx.lib.scrolledpanel as scrolled
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    from .engine import (
        UncertainParam,
        WhatIfShift,
        run_monte_carlo,
        run_sobol,
        run_whatif,
        UncertaintyResult,
        SobolResult,
        WhatIfResult,
    )
except ImportError:
    from engine import (
        UncertainParam,
        WhatIfShift,
        run_monte_carlo,
        run_sobol,
        run_whatif,
        UncertaintyResult,
        SobolResult,
        WhatIfResult,
    )


class C:
    BG = wx.Colour(248, 249, 250)
    WHITE = wx.Colour(255, 255, 255)
    HEADER = wx.Colour(30, 64, 120)
    TXT = wx.Colour(33, 37, 41)
    TXT_M = wx.Colour(108, 117, 125)
    BORDER = wx.Colour(222, 226, 230)
    OK = wx.Colour(25, 135, 84)
    WARN = wx.Colour(255, 193, 7)


def _collect_numeric_fields(sheet_data: Dict) -> List[Tuple[str, str, str, float]]:
    """(sheet_path, ref, field_name, nominal_value) for numeric fields."""
    out = []
    for path, data in sheet_data.items():
        for comp in data.get("components", []):
            if comp.get("override_lambda") is not None:
                continue
            ref = comp.get("ref", "?")
            params = comp.get("params", {})
            for fn, val in params.items():
                if fn.startswith("_"):
                    continue
                try:
                    v = float(val)
                    out.append((path, ref, fn, v))
                except (TypeError, ValueError):
                    pass
    return out


class ParameterSelector(scrolled.ScrolledPanel):
    """User selects uncertain parameters (component/field specific)."""

    def __init__(self, parent, sheet_data: Dict):
        super().__init__(parent)
        self.SetBackgroundColour(C.WHITE)
        self.sheet_data = sheet_data
        self.checks = {}  # (path, ref, fn) -> (wx.CheckBox, low_ctrl, high_ctrl, dist_choice)
        self._build()

    def _build(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        lbl = wx.StaticText(self, label="Select uncertain parameters (no defaults)")
        lbl.SetFont(lbl.GetFont().Bold())
        sizer.Add(lbl, 0, wx.ALL, 5)

        fields = _collect_numeric_fields(self.sheet_data)
        if not fields:
            sizer.Add(wx.StaticText(self, label="No numeric parameters found."), 0, wx.ALL, 10)
        else:
            grid = wx.FlexGridSizer(0, 7, 4, 4)
            grid.Add(wx.StaticText(self, label=""), 0)  # checkbox col
            grid.Add(wx.StaticText(self, label="Ref"), 0)
            grid.Add(wx.StaticText(self, label="Field"), 0)
            grid.Add(wx.StaticText(self, label="Nominal"), 0)
            grid.Add(wx.StaticText(self, label="Low"), 0)
            grid.Add(wx.StaticText(self, label="High"), 0)
            grid.Add(wx.StaticText(self, label="Dist"), 0)

            for path, ref, fn, nom in fields:
                key = (path, ref, fn)
                cb = wx.CheckBox(self, label="")
                cb.SetValue(False)
                rng = abs(nom) * 0.2 if nom != 0 else 1.0
                rng = max(rng, 0.1)  # avoid zero range
                lo = wx.TextCtrl(self, value=f"{nom - rng:.3g}", size=(80, -1))
                hi = wx.TextCtrl(self, value=f"{nom + rng:.3g}", size=(80, -1))
                dist = wx.Choice(self, choices=["uniform", "pert"], size=(60, -1))
                dist.SetSelection(0)
                self.checks[key] = (cb, lo, hi, dist, nom)
                grid.Add(cb, 0)
                grid.Add(wx.StaticText(self, label=ref), 0)
                grid.Add(wx.StaticText(self, label=fn), 0)
                grid.Add(wx.StaticText(self, label=f"{nom:.3g}"), 0)
                grid.Add(lo, 0)
                grid.Add(hi, 0)
                grid.Add(dist, 0)

            sizer.Add(grid, 0, wx.ALL, 5)
        self.SetSizer(sizer)
        self.SetupScrolling(scroll_x=True)

    def get_selected(self) -> List[UncertainParam]:
        out = []
        for (path, ref, fn), (cb, lo_ctrl, hi_ctrl, dist_ctrl, nom) in self.checks.items():
            if not cb.GetValue():
                continue
            try:
                low = float(lo_ctrl.GetValue())
                high = float(hi_ctrl.GetValue())
            except ValueError:
                continue
            dist = "pert" if dist_ctrl.GetSelection() == 1 else "uniform"
            out.append(UncertainParam(
                sheet_path=path, reference=ref, field_name=fn,
                nominal=nom, low=low, high=high, distribution=dist,
            ))
        return out


class WhatIfPanel(wx.Panel):
    """User-defined parameter shifts."""

    def __init__(self, parent, sheet_data: Dict):
        super().__init__(parent)
        self.SetBackgroundColour(C.WHITE)
        self.sheet_data = sheet_data
        self.scenarios: List[Tuple[str, List[WhatIfShift]]] = []
        self._build()

    def _build(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        lbl = wx.StaticText(self, label="What-If: user-defined parameter shifts")
        lbl.SetFont(lbl.GetFont().Bold())
        sizer.Add(lbl, 0, wx.ALL, 5)

        self.name_ctrl = wx.TextCtrl(self, value="Scenario 1", size=(150, -1))
        sizer.Add(wx.StaticText(self, label="Scenario name:"), 0)
        sizer.Add(self.name_ctrl, 0, wx.ALL, 2)

        self.shifts_list = wx.ListCtrl(self, style=wx.LC_REPORT, size=(-1, 100))
        self.shifts_list.InsertColumn(0, "Ref", width=60)
        self.shifts_list.InsertColumn(1, "Field", width=100)
        self.shifts_list.InsertColumn(2, "New value", width=80)
        sizer.Add(self.shifts_list, 1, wx.EXPAND | wx.ALL, 5)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_add = wx.Button(self, label="Add shift...")
        self.btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        self.btn_remove = wx.Button(self, label="Remove")
        self.btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)
        btn_row.Add(self.btn_add, 0)
        btn_row.Add(self.btn_remove, 0)
        sizer.Add(btn_row, 0, wx.ALL, 5)

        self.SetSizer(sizer)
        self._shifts: List[WhatIfShift] = []

    def _on_add(self, event):
        fields = _collect_numeric_fields(self.sheet_data)
        if not fields:
            wx.MessageBox("No numeric parameters.", "Info", wx.OK)
            return
        choices = [f"{ref} / {fn} = {nom:.3g}" for _, ref, fn, nom in fields]
        dlg = wx.SingleChoiceDialog(self, "Select parameter:", "Add shift", choices)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        idx = dlg.GetSelection()
        path, ref, fn, nom = fields[idx]
        val_dlg = wx.TextEntryDialog(self, "New value:", "Value", str(nom))
        if val_dlg.ShowModal() != wx.ID_OK:
            val_dlg.Destroy()
            dlg.Destroy()
            return
        try:
            new_val = float(val_dlg.GetValue())
        except ValueError:
            wx.MessageBox("Invalid number.", "Error", wx.OK | wx.ICON_ERROR)
            val_dlg.Destroy()
            dlg.Destroy()
            return
        val_dlg.Destroy()
        dlg.Destroy()

        shift = WhatIfShift(sheet_path=path, reference=ref, field_name=fn, new_value=new_val)
        self._shifts.append(shift)
        i = self.shifts_list.GetItemCount()
        self.shifts_list.InsertItem(i, ref)
        self.shifts_list.SetItem(i, 1, fn)
        self.shifts_list.SetItem(i, 2, str(new_val))

    def _on_remove(self, event):
        idx = self.shifts_list.GetFirstSelected()
        if idx >= 0:
            self.shifts_list.DeleteItem(idx)
            self._shifts.pop(idx)

    def get_scenario(self) -> Optional[Tuple[str, List[WhatIfShift]]]:
        if not self._shifts:
            return None
        return (self.name_ctrl.GetValue(), list(self._shifts))


class UnifiedAnalysisDialog(wx.Dialog):
    """Unified sensitivity & uncertainty analysis."""

    def __init__(
        self,
        parent,
        sheet_data: Dict,
        blocks: Dict,
        root_id: Optional[str],
        mission_hours: float,
        project_path: Optional[str] = None,
        title: str = "Sensitivity & Uncertainty Analysis",
    ):
        super().__init__(parent, title=title, size=(900, 750),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetBackgroundColour(C.BG)
        self.sheet_data = sheet_data
        self.blocks = blocks
        self.root_id = root_id
        self.mission_hours = mission_hours
        self.project_path = project_path
        self.mc_result: Optional[UncertaintyResult] = None
        self.sobol_result: Optional[SobolResult] = None
        self.whatif_result: Optional[WhatIfResult] = None

        nb = wx.Notebook(self)
        nb.AddPage(self._build_analysis_page(nb), "Analysis")
        nb.AddPage(self._build_report_page(nb), "Report")
        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(nb, 1, wx.EXPAND | wx.ALL, 5)
        main.Add(wx.StaticLine(self), 0, wx.EXPAND)
        btn = wx.Button(self, wx.ID_CLOSE, "Close")
        main.Add(btn, 0, wx.ALL, 10)
        self.SetSizer(main)
        self.Centre()
        self._update_report()  # initial report (empty until analyses run)

    def _build_analysis_page(self, parent):
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(C.WHITE)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Parameter selection
        self.param_selector = ParameterSelector(panel, self.sheet_data)
        sizer.Add(self.param_selector, 0, wx.EXPAND | wx.ALL, 5)

        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 10)

        # Run buttons
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_mc = wx.Button(panel, label="Run Monte Carlo (3000)")
        self.btn_mc.Bind(wx.EVT_BUTTON, self._on_run_mc)
        self.btn_sobol = wx.Button(panel, label="Run Sobol (1000)")
        self.btn_sobol.Bind(wx.EVT_BUTTON, self._on_run_sobol)
        btn_row.Add(self.btn_mc, 0, wx.RIGHT, 5)
        btn_row.Add(self.btn_sobol, 0)
        sizer.Add(btn_row, 0, wx.ALL, 5)

        # What-if
        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 10)
        self.whatif_panel = WhatIfPanel(panel, self.sheet_data)
        sizer.Add(self.whatif_panel, 0, wx.EXPAND | wx.ALL, 5)
        self.btn_whatif = wx.Button(panel, label="Run What-If")
        self.btn_whatif.Bind(wx.EVT_BUTTON, self._on_run_whatif)
        sizer.Add(self.btn_whatif, 0, wx.ALL, 5)

        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.ALL, 10)

        # Results
        self.results_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 200))
        self.results_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sizer.Add(wx.StaticText(panel, label="Results:"), 0)
        sizer.Add(self.results_text, 1, wx.EXPAND | wx.ALL, 5)

        self.progress = wx.Gauge(panel, range=100, style=wx.GA_HORIZONTAL)
        sizer.Add(self.progress, 0, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(sizer)
        return panel

    def _build_report_page(self, parent):
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(C.WHITE)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.report_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 400))
        self.report_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sizer.Add(self.report_text, 1, wx.EXPAND | wx.ALL, 5)
        btn_export = wx.Button(panel, label="Export report (HTML)")
        btn_export.Bind(wx.EVT_BUTTON, self._on_export)
        sizer.Add(btn_export, 0, wx.ALL, 5)
        panel.SetSizer(sizer)
        return panel

    def _update_report(self):
        lines = self._generate_report_lines()
        self.report_text.SetValue("\n".join(lines))

    def _generate_report_lines(self) -> List[str]:
        """Full report: summary first, then all results in clear form."""
        lines = [
            "=" * 70,
            "   SENSITIVITY & UNCERTAINTY ANALYSIS REPORT",
            "=" * 70,
            "",
            "Mission hours: {:.0f}".format(self.mission_hours),
            "",
        ]

        # ----- SUMMARY -----
        lines.append("-" * 70)
        lines.append("SUMMARY (Key figures)")
        lines.append("-" * 70)
        if self.mc_result:
            r = self.mc_result
            lines.append("Monte Carlo:")
            lines.append("  Nominal R = {:.6f}  |  Mean R = {:.6f}  |  {:.0f}% CI = [{:.6f}, {:.6f}]".format(
                r.nominal_R, r.mean_R, r.confidence_level * 100, r.ci_low, r.ci_high))
        if self.sobol_result:
            s = self.sobol_result
            lines.append("Sobol:")
            lines.append("  Nominal R = {:.6f}".format(s.nominal_R))
            if s.parameter_level:
                top = s.parameter_level[0]
                lines.append("  Top parameter: {} (S_i = {:.4f})".format(top.name, top.sobol_index))
            if s.component_level:
                top_c = s.component_level[0]
                lines.append("  Top component: {} (S_i = {:.4f})".format(top_c.name, top_c.sobol_index))
        if self.whatif_result:
            w = self.whatif_result
            lines.append("What-If [{}]:".format(w.name))
            lines.append("  R {:.6f} -> {:.6f}  |  Delta R = {:+.6f}  |  Delta FIT = {:+.1f}%".format(
                w.baseline_R, w.shifted_R, w.delta_R, w.delta_lambda_pct))
        if not (self.mc_result or self.sobol_result or self.whatif_result):
            lines.append("(No analysis run yet)")
        lines.append("")

        # ----- MONTE CARLO (full) -----
        lines.append("-" * 70)
        lines.append("MONTE CARLO (full results)")
        lines.append("-" * 70)
        if self.mc_result:
            r = self.mc_result
            lines.append("Nominal R:              {:.6f}".format(r.nominal_R))
            lines.append("Nominal FIT:            {:.2f}".format(r.nominal_lambda_fit))
            lines.append("Mean R:                 {:.6f}".format(r.mean_R))
            lines.append("Median R:               {:.6f}".format(r.median_R))
            lines.append("Std R:                  {:.6f}".format(r.std_R))
            lines.append("{:.0f}% CI:                 [{:.6f}, {:.6f}]".format(
                r.confidence_level * 100, r.ci_low, r.ci_high))
            lines.append("Simulations:            {:d}".format(r.n_simulations))
            lines.append("Runtime:                {:.2f} s".format(r.runtime_seconds))
            if r.jensen_note:
                lines.append("")
                lines.append(r.jensen_note)
        else:
            lines.append("(Not run)")
        lines.append("")

        # ----- SOBOL Parameter-level -----
        lines.append("-" * 70)
        lines.append("SOBOL SENSITIVITY (Parameter-level)")
        lines.append("-" * 70)
        if self.sobol_result and self.sobol_result.parameter_level:
            for e in self.sobol_result.parameter_level:
                lines.append("  {}  S_i = {:.4f}  +/- {:.4f}".format(e.name.ljust(24), e.sobol_index, e.sobol_std))
            lines.append("Samples: {:d}  |  Runtime: {:.2f} s".format(
                self.sobol_result.n_samples, self.sobol_result.runtime_seconds))
        else:
            lines.append("(Not run)")
        lines.append("")

        # ----- SOBOL Component-level -----
        lines.append("-" * 70)
        lines.append("SOBOL SENSITIVITY (Component-level)")
        lines.append("-" * 70)
        if self.sobol_result and self.sobol_result.component_level:
            for e in self.sobol_result.component_level:
                lines.append("  {}  S_i = {:.4f}  +/- {:.4f}".format(e.name.ljust(12), e.sobol_index, e.sobol_std))
        else:
            lines.append("(Not run)")
        lines.append("")

        # ----- WHAT-IF -----
        lines.append("-" * 70)
        lines.append("WHAT-IF (user-defined parameter shifts)")
        lines.append("-" * 70)
        if self.whatif_result:
            w = self.whatif_result
            lines.append("Scenario:               {}".format(w.name))
            lines.append("Baseline R:             {:.6f}".format(w.baseline_R))
            lines.append("Baseline FIT:           {:.2f}".format(w.baseline_lambda_fit))
            lines.append("Shifted R:              {:.6f}".format(w.shifted_R))
            lines.append("Shifted FIT:            {:.2f}".format(w.shifted_lambda_fit))
            lines.append("Delta R:                {:+.6f}".format(w.delta_R))
            lines.append("Delta FIT:              {:+.1f}%".format(w.delta_lambda_pct))
        else:
            lines.append("(Not run)")

        lines.append("")
        lines.append("=" * 70)
        return lines

    def _on_run_mc(self, event):
        params = self.param_selector.get_selected()
        if not params:
            wx.MessageBox("Select at least one uncertain parameter.", "No selection", wx.OK)
            return
        self.progress.SetValue(0)
        self.results_text.SetValue("Running Monte Carlo...")

        def progress(current, total, msg):
            wx.CallAfter(self.progress.SetValue, int(100 * current / total) if total > 0 else 0)
            wx.CallAfter(self.results_text.SetValue, f"{msg} {current}/{total}")

        try:
            result = run_monte_carlo(
                self.sheet_data,
                self.blocks,
                self.root_id,
                self.mission_hours,
                params,
                n_simulations=3000,
                progress_callback=progress,
            )
            self.mc_result = result
            txt = (f"Monte Carlo: nominal R = {result.nominal_R:.6f}, mean R = {result.mean_R:.6f}\n"
                   f"CI [{result.ci_low:.6f}, {result.ci_high:.6f}]\n{result.jensen_note or ''}")
            self.results_text.SetValue(txt)
            self._update_report()
        except Exception as e:
            self.results_text.SetValue(f"Error: {e}")
        self.progress.SetValue(100)

    def _on_run_sobol(self, event):
        params = self.param_selector.get_selected()
        if not params:
            wx.MessageBox("Select at least one uncertain parameter.", "No selection", wx.OK)
            return
        self.progress.SetValue(0)
        self.results_text.SetValue("Running Sobol...")

        def progress(current, total, msg):
            wx.CallAfter(self.progress.SetValue, int(100 * current / total) if total > 0 else 0)

        try:
            result = run_sobol(
                self.sheet_data,
                self.blocks,
                self.root_id,
                self.mission_hours,
                params,
                n_samples=1000,
                progress_callback=progress,
            )
            self.sobol_result = result
            lines = ["Sobol (parameter-level):"]
            for e in result.parameter_level[:10]:
                lines.append(f"  {e.name}: S_i = {e.sobol_index:.4f}")
            lines.append("\nSobol (component-level):")
            for e in result.component_level[:10]:
                lines.append(f"  {e.name}: S_i = {e.sobol_index:.4f}")
            self.results_text.SetValue("\n".join(lines))
            self._update_report()
        except Exception as e:
            self.results_text.SetValue(f"Error: {e}")
        self.progress.SetValue(100)

    def _on_run_whatif(self, event):
        scenario = self.whatif_panel.get_scenario()
        if not scenario:
            wx.MessageBox("Add at least one parameter shift.", "No shifts", wx.OK)
            return
        name, shifts = scenario
        try:
            result = run_whatif(
                self.sheet_data,
                self.blocks,
                self.root_id,
                self.mission_hours,
                shifts,
                scenario_name=name,
            )
            self.whatif_result = result
            txt = (f"What-If [{name}]: R {result.baseline_R:.6f} -> {result.shifted_R:.6f}\n"
                   f"Delta R = {result.delta_R:+.6f}, Delta FIT = {result.delta_lambda_pct:+.1f}%")
            self.results_text.SetValue(txt)
            self._update_report()
        except Exception as e:
            self.results_text.SetValue(f"Error: {e}")

    def _on_export(self, event):
        lines = self._generate_report_lines()
        html = self._report_to_html("\n".join(lines))
        default_dir = ""
        if self.project_path:
            try:
                from ..project_manager import ProjectManager
                pm = ProjectManager(self.project_path)
                default_dir = str(pm.get_reports_folder())
            except ImportError:
                pass
        dlg = wx.FileDialog(
            self, "Export report", defaultDir=default_dir,
            defaultFile="analysis_report.html",
            wildcard="HTML (*.html)|*.html",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            wx.MessageBox(f"Saved: {path}", "Export", wx.OK)
        dlg.Destroy()

    def _report_to_html(self, text: str) -> str:
        """Convert plain-text report to clean HTML."""
        lines = text.split("\n")
        html_parts = []
        for line in lines:
            s = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            s = s.replace("  ", "&nbsp;&nbsp;")
            if line.strip().startswith("="):
                html_parts.append(f'<div class="hr">{s}</div>')
            elif line.strip().startswith("-"):
                html_parts.append(f'<div class="hr2">{s}</div>')
            else:
                html_parts.append(f"<div>{s}</div>")
        body = "\n".join(html_parts)
        return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sensitivity & Uncertainty Analysis Report</title>
<style>
body { font-family: 'Segoe UI', Consolas, monospace; margin: 2em; max-width: 800px; }
.hr { font-weight: bold; color: #1e4078; margin: 1em 0 0.5em; }
.hr2 { color: #666; margin: 0.8em 0 0.3em; }
div { line-height: 1.4; }
</style></head>
<body>""" + body + "</body></html>"
