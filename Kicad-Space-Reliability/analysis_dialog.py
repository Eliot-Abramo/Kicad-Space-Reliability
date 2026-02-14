"""
Analysis Dialog - Reliability Analysis Suite
=============================================================
Monte Carlo uncertainty, Tornado sensitivity, Design-margin analysis,
component criticality with field selection, and comprehensive reporting.

Respects block diagram structure (active sheets only).
Supports component lambda overrides and configurable confidence intervals.

Author:  Eliot Abramo
"""

import wx
import wx.lib.scrolledpanel as scrolled
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .monte_carlo import (
    MonteCarloResult, quick_monte_carlo, monte_carlo_sheet,
    monte_carlo_blocks, SheetMCResult, ComponentMCInput,
)
from .sensitivity_analysis import (
    SobolResult, SobolAnalyzer, TornadoResult, DesignMarginResult,
    tornado_sheet_sensitivity, tornado_parameter_sensitivity,
    design_margin_analysis, analyze_board_criticality, get_active_sheet_paths,
)
from .reliability_math import reliability_from_lambda, calculate_component_lambda
from .report_generator import ReportGenerator, ReportData


# =============================================================================
# Color Scheme
# =============================================================================
class Colors:
    BG_LIGHT = wx.Colour(252, 252, 253)
    BG_WHITE = wx.Colour(255, 255, 255)
    BG_HEADER = wx.Colour(37, 99, 235)
    TEXT_DARK = wx.Colour(17, 24, 39)
    TEXT_MEDIUM = wx.Colour(75, 85, 99)
    TEXT_LIGHT = wx.Colour(156, 163, 175)
    TEXT_WHITE = wx.Colour(255, 255, 255)
    PRIMARY = wx.Colour(37, 99, 235)
    SUCCESS = wx.Colour(34, 197, 94)
    WARNING = wx.Colour(245, 158, 11)
    DANGER = wx.Colour(239, 68, 68)
    CHART = [wx.Colour(59,130,246), wx.Colour(16,185,129), wx.Colour(245,158,11),
             wx.Colour(239,68,68), wx.Colour(139,92,246), wx.Colour(236,72,153),
             wx.Colour(20,184,166), wx.Colour(249,115,22)]
    BORDER = wx.Colour(229, 231, 235)
    GRID = wx.Colour(243, 244, 246)


# =============================================================================
# Chart Panels
# =============================================================================
class HistogramPanel(wx.Panel):
    def __init__(self, parent, title="Distribution"):
        super().__init__(parent, size=(500, 320))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.samples = None
        self.mean = self.p5 = self.p95 = None
        self.ci_lo = self.ci_hi = None
        self.ci_label = "90% CI"
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, samples, mean, p5, p95, ci_lo=None, ci_hi=None, ci_label="90% CI"):
        self.samples = samples
        self.mean, self.p5, self.p95 = mean, p5, p95
        self.ci_lo, self.ci_hi = ci_lo, ci_hi
        self.ci_label = ci_label
        self.Refresh()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()
        dc.SetBrush(wx.Brush(Colors.BG_WHITE))
        dc.SetPen(wx.Pen(Colors.BORDER, 1))
        dc.DrawRectangle(0, 0, w, h)
        dc.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 16, 12)
        if self.samples is None or len(self.samples) == 0:
            dc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            dc.SetTextForeground(Colors.TEXT_LIGHT)
            dc.DrawText("Run analysis to see distribution", w // 2 - 100, h // 2)
            return
        ml, mr, mt, mb = 60, 30, 45, 55
        cw = w - ml - mr
        ch = h - mt - mb
        if cw <= 0 or ch <= 0:
            return
        n_bins = 35
        hist, edges = np.histogram(self.samples, bins=n_bins)
        max_count = max(hist) if max(hist) > 0 else 1
        bar_width = max(1, cw // n_bins - 1)
        dc.SetPen(wx.Pen(Colors.GRID, 1))
        for i in range(5):
            y = mt + ch * i // 4
            dc.DrawLine(ml, y, w - mr, y)

        # CI shading
        min_val, max_val = edges[0], edges[-1]
        val_range = max_val - min_val
        if val_range <= 0:
            val_range = 1
        def val_to_x(v):
            return ml + (v - min_val) / val_range * cw
        if self.ci_lo is not None and self.ci_hi is not None:
            x1 = int(val_to_x(self.ci_lo))
            x2 = int(val_to_x(self.ci_hi))
            dc.SetBrush(wx.Brush(wx.Colour(34, 197, 94, 30)))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(x1, mt, x2 - x1, ch)

        dc.SetBrush(wx.Brush(Colors.PRIMARY))
        dc.SetPen(wx.Pen(Colors.PRIMARY.ChangeLightness(85), 1))
        for i, count in enumerate(hist):
            if count > 0:
                x = ml + i * cw // n_bins
                bar_h = int((count / max_count) * ch)
                dc.DrawRectangle(int(x), mt + ch - bar_h, int(bar_width), bar_h)

        if self.mean is not None:
            dc.SetPen(wx.Pen(Colors.DANGER, 2))
            xm = int(val_to_x(self.mean))
            dc.DrawLine(xm, mt, xm, mt + ch)
        dc.SetPen(wx.Pen(Colors.WARNING, 2, wx.PENSTYLE_SHORT_DASH))
        for pv in [self.p5, self.p95]:
            if pv is not None:
                xp = int(val_to_x(pv))
                dc.DrawLine(xp, mt, xp, mt + ch)
        if self.ci_lo is not None:
            dc.SetPen(wx.Pen(Colors.SUCCESS, 2, wx.PENSTYLE_SHORT_DASH))
            dc.DrawLine(int(val_to_x(self.ci_lo)), mt, int(val_to_x(self.ci_lo)), mt + ch)
            dc.DrawLine(int(val_to_x(self.ci_hi)), mt, int(val_to_x(self.ci_hi)), mt + ch)

        dc.SetPen(wx.Pen(Colors.TEXT_MEDIUM, 1))
        dc.DrawLine(ml, mt + ch, w - mr, mt + ch)
        dc.DrawLine(ml, mt, ml, mt + ch)
        dc.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        dc.SetTextForeground(Colors.TEXT_MEDIUM)
        for i in range(5):
            val = min_val + val_range * i / 4
            x = ml + cw * i // 4
            label = f"{val:.4f}"
            tw, _ = dc.GetTextExtent(label)
            dc.DrawText(label, x - tw // 2, mt + ch + 8)
        dc.DrawText("Reliability R(t)", ml + cw // 2 - 50, h - 18)
        # Legend
        lx = w - mr - 140
        ly = mt + 5
        dc.SetFont(wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        dc.SetPen(wx.Pen(Colors.DANGER, 2))
        dc.DrawLine(lx, ly+6, lx+20, ly+6)
        dc.DrawText("Mean", lx+25, ly)
        dc.SetPen(wx.Pen(Colors.WARNING, 2, wx.PENSTYLE_SHORT_DASH))
        dc.DrawLine(lx, ly+20, lx+20, ly+20)
        dc.DrawText("5th/95th %ile", lx+25, ly+14)
        if self.ci_lo is not None:
            dc.SetPen(wx.Pen(Colors.SUCCESS, 2, wx.PENSTYLE_SHORT_DASH))
            dc.DrawLine(lx, ly+34, lx+20, ly+34)
            dc.DrawText(self.ci_label, lx+25, ly+28)


class HorizontalBarPanel(wx.Panel):
    def __init__(self, parent, title="Chart"):
        super().__init__(parent, size=(500, 400))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.data = []
        self.max_value = 1.0
        self.x_label = "Value"
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, data, max_value=None, x_label="Value"):
        self.data = [(name, val, i % len(Colors.CHART)) for i, (name, val) in enumerate(data)]
        self.max_value = max_value if max_value else (max(d[1] for d in self.data) if self.data else 1.0)
        self.max_value = max(self.max_value, 0.001)
        self.x_label = x_label
        self.Refresh()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()
        dc.SetBrush(wx.Brush(Colors.BG_WHITE))
        dc.SetPen(wx.Pen(Colors.BORDER, 1))
        dc.DrawRectangle(0, 0, w, h)
        dc.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 16, 12)
        if not self.data:
            return
        ml, mr, mt, mb = 140, 30, 45, 45
        cw = w - ml - mr
        ch = h - mt - mb
        if cw <= 0 or ch <= 0:
            return
        n = min(len(self.data), 15)
        bh = min(22, max(12, (ch - 10) // n))
        sp = max(2, (ch - n * bh) // (n + 1))
        dc.SetPen(wx.Pen(Colors.GRID, 1))
        for i in range(5):
            x = ml + cw * i // 4
            dc.DrawLine(x, mt, x, mt + ch)
        dc.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        for i, (name, value, ci) in enumerate(self.data[:n]):
            y = mt + sp + i * (bh + sp)
            bw = max(2, int((value / self.max_value) * cw))
            color = Colors.CHART[ci]
            dc.SetBrush(wx.Brush(color))
            dc.SetPen(wx.Pen(color.ChangeLightness(85), 1))
            dc.DrawRoundedRectangle(ml, y, bw, bh, 3)
            dn = name[:18] + "..." if len(name) > 18 else name
            dc.SetTextForeground(Colors.TEXT_DARK)
            tw, th = dc.GetTextExtent(dn)
            dc.DrawText(dn, ml - tw - 8, y + (bh - th) // 2)
            vt = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            vw, vh = dc.GetTextExtent(vt)
            if bw > vw + 10:
                dc.SetTextForeground(Colors.TEXT_WHITE)
                dc.DrawText(vt, ml + 6, y + (bh - vh) // 2)
            else:
                dc.SetTextForeground(Colors.TEXT_DARK)
                dc.DrawText(vt, ml + bw + 6, y + (bh - vh) // 2)


class ConvergencePanel(wx.Panel):
    def __init__(self, parent, title="Convergence"):
        super().__init__(parent, size=(400, 200))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.samples = None
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, samples):
        self.samples = samples
        self.Refresh()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()
        dc.SetBrush(wx.Brush(Colors.BG_WHITE))
        dc.SetPen(wx.Pen(Colors.BORDER, 1))
        dc.DrawRectangle(0, 0, w, h)
        dc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 12, 8)
        if self.samples is None or len(self.samples) == 0:
            return
        ml, mr, mt, mb = 55, 20, 35, 35
        cw = w - ml - mr
        ch = h - mt - mb
        if cw <= 0 or ch <= 0:
            return
        cumsum = np.cumsum(self.samples)
        rm = cumsum / np.arange(1, len(self.samples) + 1)
        step = max(1, len(rm) // 100)
        pts = [(i, rm[i]) for i in range(0, len(rm), step)]
        vals = [p[1] for p in pts]
        v_min, v_max = min(vals), max(vals)
        vr = v_max - v_min
        if vr < 1e-9:
            vr = abs(v_max) * 0.1 if v_max != 0 else 0.01
        v_min -= vr * 0.1
        v_max += vr * 0.1
        vr = v_max - v_min
        dc.SetPen(wx.Pen(Colors.GRID, 1))
        for i in range(5):
            y = mt + ch * i // 4
            dc.DrawLine(ml, y, w - mr, y)
        dc.SetPen(wx.Pen(Colors.PRIMARY, 2))
        prev = None
        for n_, v in pts:
            x = ml + (n_ / len(self.samples)) * cw
            y = mt + ch - ((v - v_min) / vr) * ch
            if prev:
                dc.DrawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)


class StatsCard(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_WHITE)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(self, label="Summary Statistics")
        title.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        title.SetForegroundColour(Colors.TEXT_DARK)
        self.sizer.Add(title, 0, wx.ALL, 12)
        self.content_sizer = wx.FlexGridSizer(0, 2, 6, 16)
        self.content_sizer.AddGrowableCol(1)
        self.sizer.Add(self.content_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        self.SetSizer(self.sizer)

    def set_stats(self, stats: Dict[str, Any]):
        self.content_sizer.Clear(True)
        display_order = [
            ("Mean (mu)", "mean", ".6f"),
            ("Std Dev (sigma)", "std", ".6f"),
            ("Median", "median", ".6f"),
            ("CI Lower", "ci_lower", ".6f"),
            ("CI Upper", "ci_upper", ".6f"),
            ("CI Width", "ci_width", ".6f"),
            ("CV", "cv", ".2%"),
            ("Simulations", "n_sims", ",d"),
            ("Converged", "converged", "bool"),
        ]
        for label, key, fmt in display_order:
            if key not in stats:
                continue
            lbl = wx.StaticText(self, label=label + ":")
            lbl.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            lbl.SetForegroundColour(Colors.TEXT_MEDIUM)
            self.content_sizer.Add(lbl, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
            val = stats[key]
            if fmt == "bool":
                text = "Yes" if val else "No"
                color = Colors.SUCCESS if val else Colors.WARNING
            elif fmt == ".2%":
                text = f"{val * 100:.2f}%"
                color = Colors.TEXT_DARK
            elif fmt == ",d":
                text = f"{int(val):,}"
                color = Colors.TEXT_DARK
            else:
                text = f"{val:{fmt}}"
                color = Colors.TEXT_DARK
            vl = wx.StaticText(self, label=text)
            vl.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            vl.SetForegroundColour(color)
            self.content_sizer.Add(vl, 0, wx.ALIGN_CENTER_VERTICAL)
        self.Layout()


# =============================================================================
# Main Analysis Dialog
# =============================================================================
class AnalysisDialog(wx.Dialog):
    """Professional reliability analysis dialog with co-design tools."""

    def __init__(self, parent, system_lambda, mission_hours, sheet_data=None,
                 block_structure=None, blocks=None, project_path=None,
                 logo_path=None, logo_mime=None, n_cycles=5256, delta_t=3.0,
                 title="Reliability Analysis Suite"):
        display = wx.Display(0)
        rect = display.GetClientArea()
        w = min(1400, int(rect.Width * 0.85))
        h = min(950, int(rect.Height * 0.88))
        super().__init__(parent, title=title, size=(w, h),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        self.SetMinSize((1000, 700))
        self.SetBackgroundColour(Colors.BG_LIGHT)

        self.system_lambda = system_lambda
        self.mission_hours = mission_hours
        self.sheet_data = sheet_data or {}
        self.block_structure = block_structure or {}
        self.blocks = blocks  # Block editor blocks dict
        self.project_path = project_path
        self.logo_path = logo_path
        self.logo_mime = logo_mime or "image/png"
        self.n_cycles = n_cycles
        self.delta_t = delta_t

        # Determine active sheets from block diagram
        self.active_sheets = get_active_sheet_paths(self.blocks)
        if self.active_sheets:
            self._active_data = {k: v for k, v in self.sheet_data.items() if k in self.active_sheets}
        else:
            self._active_data = self.sheet_data

        self.mc_result: Optional[MonteCarloResult] = None
        self.sobol_result: Optional[SobolResult] = None
        self.tornado_result: Optional[TornadoResult] = None
        self.param_tornado_result: Optional[TornadoResult] = None
        self.design_margin_result: Optional[DesignMarginResult] = None
        self.sheet_mc_results: Dict[str, SheetMCResult] = {}
        self.criticality_results: List[Dict] = []

        self._create_ui()
        self.Centre()

    def _get_active_label(self) -> str:
        if self.active_sheets:
            return f"{len(self.active_sheets)} active sheets (from block diagram)"
        return f"{len(self.sheet_data)} sheets (all)"

    def _create_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        header = self._create_header()
        main_sizer.Add(header, 0, wx.EXPAND)

        self.notebook = wx.Notebook(self)
        self.notebook.SetBackgroundColour(Colors.BG_LIGHT)
        self.notebook.AddPage(self._create_mc_tab(), "Monte Carlo")
        self.notebook.AddPage(self._create_tornado_tab(), "Sensitivity (Tornado)")
        self.notebook.AddPage(self._create_design_margin_tab(), "Design Margin")
        self.notebook.AddPage(self._create_contributions_tab(), "Contributions")
        self.notebook.AddPage(self._create_criticality_tab(), "Criticality")
        self.notebook.AddPage(self._create_report_tab(), "Full Report")
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 12)

        footer = self._create_footer()
        main_sizer.Add(footer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        self.SetSizer(main_sizer)

    def _create_header(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour(Colors.BG_HEADER)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        title = wx.StaticText(panel, label="Reliability Analysis Suite")
        title.SetFont(wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        title.SetForegroundColour(Colors.TEXT_WHITE)
        sizer.Add(title, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 14)
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        info_text = f"L = {self.system_lambda*1e9:.2f} FIT  |  R = {r:.6f}  |  {years:.1f}y  |  {self._get_active_label()}"
        info = wx.StaticText(panel, label=info_text)
        info.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        info.SetForegroundColour(wx.Colour(191, 219, 254))
        sizer.Add(info, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 14)
        panel.SetSizer(sizer)
        return panel

    # =========================================================================
    # Monte Carlo Tab
    # =========================================================================
    def _create_mc_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        ctrl_panel = wx.Panel(panel)
        ctrl_panel.SetBackgroundColour(Colors.BG_WHITE)
        ctrl = wx.BoxSizer(wx.HORIZONTAL)

        ctrl.Add(wx.StaticText(ctrl_panel, label="Simulations:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.mc_n = wx.SpinCtrl(ctrl_panel, min=1000, max=100000, initial=5000, size=(100, -1))
        ctrl.Add(self.mc_n, 0, wx.ALL, 8)

        ctrl.Add(wx.StaticText(ctrl_panel, label="Uncertainty (%):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.mc_unc = wx.SpinCtrlDouble(ctrl_panel, min=1, max=100, initial=25, inc=1, size=(70, -1))
        ctrl.Add(self.mc_unc, 0, wx.ALL, 8)

        ctrl.Add(wx.StaticText(ctrl_panel, label="Confidence:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.mc_ci = wx.Choice(ctrl_panel, choices=["80%", "90%", "95%", "99%"])
        self.mc_ci.SetSelection(1)  # Default 90%
        ctrl.Add(self.mc_ci, 0, wx.ALL, 8)

        self.btn_mc = wx.Button(ctrl_panel, label="Run System MC")
        self.btn_mc.SetBackgroundColour(Colors.PRIMARY)
        self.btn_mc.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_mc.Bind(wx.EVT_BUTTON, self._on_run_mc)
        ctrl.Add(self.btn_mc, 0, wx.ALL, 8)

        self.btn_mc_sheets = wx.Button(ctrl_panel, label="Run Per-Sheet MC")
        self.btn_mc_sheets.SetBackgroundColour(Colors.WARNING)
        self.btn_mc_sheets.SetForegroundColour(Colors.TEXT_DARK)
        self.btn_mc_sheets.Bind(wx.EVT_BUTTON, self._on_run_mc_sheets)
        ctrl.Add(self.btn_mc_sheets, 0, wx.ALL, 8)

        ctrl_panel.SetSizer(ctrl)
        main.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 8)

        charts = wx.BoxSizer(wx.HORIZONTAL)
        left = wx.BoxSizer(wx.VERTICAL)
        self.histogram = HistogramPanel(panel, "Reliability Distribution")
        left.Add(self.histogram, 2, wx.EXPAND)
        self.convergence = ConvergencePanel(panel, "Mean Convergence")
        left.Add(self.convergence, 1, wx.EXPAND | wx.TOP, 8)
        charts.Add(left, 2, wx.EXPAND | wx.RIGHT, 8)
        self.stats_card = StatsCard(panel)
        charts.Add(self.stats_card, 1, wx.EXPAND)
        main.Add(charts, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        panel.SetSizer(main)
        return panel

    # =========================================================================
    # Tornado Tab
    # =========================================================================
    def _create_tornado_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        ctrl_panel = wx.Panel(panel)
        ctrl_panel.SetBackgroundColour(Colors.BG_WHITE)
        ctrl = wx.BoxSizer(wx.HORIZONTAL)

        ctrl.Add(wx.StaticText(ctrl_panel, label="Perturbation (%):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.tornado_pct = wx.SpinCtrl(ctrl_panel, min=5, max=50, initial=20, size=(70, -1))
        ctrl.Add(self.tornado_pct, 0, wx.ALL, 8)

        ctrl.Add(wx.StaticText(ctrl_panel, label="Mode:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.tornado_mode = wx.Choice(ctrl_panel, choices=["Sheet-level", "Parameter-level"])
        self.tornado_mode.SetSelection(0)
        ctrl.Add(self.tornado_mode, 0, wx.ALL, 8)

        self.btn_tornado = wx.Button(ctrl_panel, label="Run Tornado Analysis")
        self.btn_tornado.SetBackgroundColour(Colors.SUCCESS)
        self.btn_tornado.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_tornado.Bind(wx.EVT_BUTTON, self._on_run_tornado)
        ctrl.Add(self.btn_tornado, 0, wx.ALL, 8)

        ctrl_panel.SetSizer(ctrl)
        main.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 8)

        self.tornado_chart = HorizontalBarPanel(panel, "Tornado Sensitivity (Swing in FIT)")
        main.Add(self.tornado_chart, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        self.tornado_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.tornado_list.SetBackgroundColour(Colors.BG_WHITE)
        self.tornado_list.InsertColumn(0, "Parameter", width=180)
        self.tornado_list.InsertColumn(1, "Low (FIT)", width=90)
        self.tornado_list.InsertColumn(2, "Base (FIT)", width=90)
        self.tornado_list.InsertColumn(3, "High (FIT)", width=90)
        self.tornado_list.InsertColumn(4, "Swing (FIT)", width=90)
        main.Add(self.tornado_list, 1, wx.EXPAND | wx.ALL, 8)

        self.tornado_info = wx.StaticText(panel, label="Run analysis to see which factors most affect system reliability.")
        self.tornado_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.tornado_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    # =========================================================================
    # Design Margin Tab
    # =========================================================================
    def _create_design_margin_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        ctrl_panel = wx.Panel(panel)
        ctrl_panel.SetBackgroundColour(Colors.BG_WHITE)
        ctrl = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_dm = wx.Button(ctrl_panel, label="Run Design Margin Analysis")
        self.btn_dm.SetBackgroundColour(Colors.PRIMARY)
        self.btn_dm.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_dm.Bind(wx.EVT_BUTTON, self._on_run_design_margin)
        ctrl.Add(self.btn_dm, 0, wx.ALL, 8)
        note = wx.StaticText(ctrl_panel, label="Evaluates built-in what-if scenarios on active sheets only.")
        note.SetForegroundColour(Colors.TEXT_MEDIUM)
        ctrl.Add(note, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        ctrl_panel.SetSizer(ctrl)
        main.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 8)

        self.dm_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.dm_list.SetBackgroundColour(Colors.BG_WHITE)
        self.dm_list.InsertColumn(0, "Scenario", width=160)
        self.dm_list.InsertColumn(1, "Description", width=250)
        self.dm_list.InsertColumn(2, "Lambda (FIT)", width=100)
        self.dm_list.InsertColumn(3, "R(t)", width=100)
        self.dm_list.InsertColumn(4, "Delta Lambda", width=100)
        self.dm_list.InsertColumn(5, "Delta R", width=100)
        main.Add(self.dm_list, 1, wx.EXPAND | wx.ALL, 8)

        panel.SetSizer(main)
        return panel

    # =========================================================================
    # Contributions Tab
    # =========================================================================
    def _create_contributions_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)
        self.contrib_chart = HorizontalBarPanel(panel, "Failure Rate Contributions")
        main.Add(self.contrib_chart, 1, wx.EXPAND | wx.ALL, 8)
        self.contrib_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.contrib_list.SetBackgroundColour(Colors.BG_WHITE)
        self.contrib_list.InsertColumn(0, "Component/Sheet", width=200)
        self.contrib_list.InsertColumn(1, "Lambda (FIT)", width=100)
        self.contrib_list.InsertColumn(2, "Contribution", width=100)
        self.contrib_list.InsertColumn(3, "Cumulative", width=100)
        main.Add(self.contrib_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        panel.SetSizer(main)
        self._update_contributions()
        return panel

    # =========================================================================
    # Criticality Tab (with field picker)
    # =========================================================================
    def _create_criticality_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        # Controls row 1
        ctrl1 = wx.Panel(panel)
        ctrl1.SetBackgroundColour(Colors.BG_WHITE)
        row1 = wx.BoxSizer(wx.HORIZONTAL)
        row1.Add(wx.StaticText(ctrl1, label="Top N:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.crit_top_n = wx.SpinCtrl(ctrl1, min=1, max=50, initial=10, size=(70, -1))
        row1.Add(self.crit_top_n, 0, wx.ALL, 6)
        row1.Add(wx.StaticText(ctrl1, label="Perturbation (%):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.crit_perturb = wx.SpinCtrl(ctrl1, min=1, max=50, initial=10, size=(70, -1))
        row1.Add(self.crit_perturb, 0, wx.ALL, 6)
        self.btn_crit = wx.Button(ctrl1, label="Run Criticality Analysis")
        self.btn_crit.SetBackgroundColour(Colors.PRIMARY)
        self.btn_crit.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_crit.Bind(wx.EVT_BUTTON, self._on_run_criticality)
        row1.Add(self.btn_crit, 0, wx.ALL, 6)
        ctrl1.SetSizer(row1)
        main.Add(ctrl1, 0, wx.EXPAND | wx.ALL, 8)

        # Field picker
        fp_panel = wx.Panel(panel)
        fp_panel.SetBackgroundColour(Colors.BG_WHITE)
        fp_sizer = wx.BoxSizer(wx.VERTICAL)
        fp_label = wx.StaticText(fp_panel, label="Field Selection per Component Category (leave empty = analyze all fields):")
        fp_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        fp_sizer.Add(fp_label, 0, wx.ALL, 8)

        # Detect categories present in active data
        self._categories_in_data = self._get_categories_in_data()
        self._field_checkboxes = {}  # {category: {field_name: wx.CheckBox}}

        scroll = scrolled.ScrolledPanel(fp_panel, size=(-1, 150))
        scroll.SetBackgroundColour(Colors.BG_WHITE)
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        try:
            from .ecss_fields import get_category_fields
        except ImportError:
            from ecss_fields import get_category_fields

        for cat_key in sorted(self._categories_in_data):
            cat_def = get_category_fields(cat_key)
            display = cat_def.get("display_name", cat_key)
            fields = cat_def.get("fields", {})
            if not fields:
                continue

            cat_label = wx.StaticText(scroll, label=f"  {display}:")
            cat_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            cat_label.SetForegroundColour(Colors.PRIMARY)
            scroll_sizer.Add(cat_label, 0, wx.TOP | wx.LEFT, 4)

            field_row = wx.WrapSizer(wx.HORIZONTAL)
            self._field_checkboxes[cat_key] = {}
            for fname, fdef in fields.items():
                cb = wx.CheckBox(scroll, label=fdef.get("label", fname))
                cb.SetValue(True)  # All checked by default
                cb.SetToolTip(fdef.get("help", ""))
                field_row.Add(cb, 0, wx.ALL, 3)
                self._field_checkboxes[cat_key][fname] = cb
            scroll_sizer.Add(field_row, 0, wx.LEFT, 20)

        scroll.SetSizer(scroll_sizer)
        scroll.SetupScrolling(scroll_x=False)
        fp_sizer.Add(scroll, 1, wx.EXPAND | wx.ALL, 4)
        fp_panel.SetSizer(fp_sizer)
        main.Add(fp_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Results
        self.crit_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.crit_list.SetBackgroundColour(Colors.BG_WHITE)
        self.crit_list.InsertColumn(0, "Reference", width=100)
        self.crit_list.InsertColumn(1, "Type", width=100)
        self.crit_list.InsertColumn(2, "Base FIT", width=80)
        self.crit_list.InsertColumn(3, "Top Parameter", width=150)
        self.crit_list.InsertColumn(4, "Elasticity", width=80)
        self.crit_list.InsertColumn(5, "Impact (%)", width=80)
        main.Add(self.crit_list, 1, wx.EXPAND | wx.ALL, 8)

        self.crit_info = wx.StaticText(panel, label="Select fields and run to see which inputs most affect component failure rates.")
        self.crit_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.crit_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    # =========================================================================
    # Report Tab
    # =========================================================================
    def _create_report_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)
        self.report_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.report_text.SetBackgroundColour(Colors.BG_WHITE)
        self.report_text.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main.Add(self.report_text, 1, wx.EXPAND | wx.ALL, 8)
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        btn_html = wx.Button(panel, label="Export HTML")
        btn_html.Bind(wx.EVT_BUTTON, self._on_export_html)
        btn_row.Add(btn_html, 0, wx.RIGHT, 8)
        btn_csv = wx.Button(panel, label="Export CSV")
        btn_csv.Bind(wx.EVT_BUTTON, self._on_export_csv)
        btn_row.Add(btn_csv, 0)
        main.Add(btn_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        panel.SetSizer(main)
        self._update_report()
        return panel

    def _create_footer(self):
        panel = wx.Panel(self)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.status = wx.StaticText(panel, label="Ready")
        self.status.SetForegroundColour(Colors.TEXT_MEDIUM)
        sizer.Add(self.status, 1, wx.ALIGN_CENTER_VERTICAL)
        close_btn = wx.Button(panel, label="Close", size=(100, -1))
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_OK))
        sizer.Add(close_btn, 0)
        panel.SetSizer(sizer)
        return panel

    # =========================================================================
    # Helpers
    # =========================================================================
    def _get_ci_level(self) -> float:
        sel = self.mc_ci.GetStringSelection()
        return {"80%": 0.80, "90%": 0.90, "95%": 0.95, "99%": 0.99}.get(sel, 0.90)

    def _get_categories_in_data(self) -> set:
        """Find which ECSS categories are present in the active data."""
        try:
            from .ecss_fields import infer_category_from_class
        except ImportError:
            from ecss_fields import infer_category_from_class
        cats = set()
        for data in self._active_data.values():
            for comp in data.get("components", []):
                cls = comp.get("class", "")
                if cls:
                    cats.add(infer_category_from_class(cls))
        return cats

    def _get_target_fields(self) -> Optional[Dict[str, List[str]]]:
        """Build target_fields dict from checkbox state. Returns None if all checked."""
        try:
            from .ecss_fields import infer_category_from_class
        except ImportError:
            from ecss_fields import infer_category_from_class

        result = {}
        all_checked = True
        for cat_key, fields in self._field_checkboxes.items():
            selected = [fname for fname, cb in fields.items() if cb.GetValue()]
            if len(selected) < len(fields):
                all_checked = False
            result[cat_key] = selected

        # Map component class names to ECSS category keys for the target_fields
        # The analyze_board_criticality uses comp["class"] as the category key
        # So we need to map both ways
        expanded = {}
        for cat_key, field_list in result.items():
            expanded[cat_key] = field_list
            # Also add common class names that map to this category
        return None if all_checked else expanded

    def _extract_components_for_mc(self) -> List[Dict]:
        """Extract component data from active sheet_data."""
        components = []
        for sheet_path, data in self._active_data.items():
            for comp in data.get("components", []):
                comp_type = comp.get("class", "Resistor")
                if comp_type in ("Unknown", "", None):
                    comp_type = "Resistor"
                params = comp.get("params", {})
                if not params:
                    params = {"t_ambient": 25.0, "t_junction": 85.0, "n_cycles": 5256,
                              "delta_t": 3.0, "operating_power": 0.01, "rated_power": 0.125}
                components.append({
                    "ref": comp.get("ref", "?"),
                    "type": comp_type,
                    "params": params,
                    "override_lambda": comp.get("override_lambda"),
                })
        return components

    # =========================================================================
    # Analysis Methods
    # =========================================================================
    def _on_run_mc(self, event):
        self.status.SetLabel("Running Monte Carlo...")
        self.btn_mc.Disable()
        wx.Yield()
        try:
            n = self.mc_n.GetValue()
            unc = self.mc_unc.GetValue()
            ci = self._get_ci_level()
            components = self._extract_components_for_mc()

            if components:
                self.mc_result = quick_monte_carlo(
                    self.system_lambda, self.mission_hours,
                    uncertainty_percent=unc, n_simulations=n,
                    components=components, confidence_level=ci)
            else:
                self.mc_result = quick_monte_carlo(
                    self.system_lambda, self.mission_hours,
                    uncertainty_percent=unc, n_simulations=n,
                    confidence_level=ci)

            mc = self.mc_result
            ci_lo, ci_hi = mc.confidence_interval(ci)
            ci_label = f"{ci*100:.0f}% CI"

            self.histogram.set_data(mc.samples, mc.mean, mc.percentile_5, mc.percentile_95,
                                    ci_lo=ci_lo, ci_hi=ci_hi, ci_label=ci_label)
            self.convergence.set_data(mc.samples)
            self.stats_card.set_stats({
                "mean": mc.mean, "std": mc.std, "median": mc.percentile_50,
                "ci_lower": ci_lo, "ci_upper": ci_hi,
                "ci_width": ci_hi - ci_lo,
                "cv": mc.std / mc.mean if mc.mean > 0 else 0,
                "n_sims": n, "converged": mc.converged,
            })

            status_msg = f"MC complete: {n:,} sims, {ci*100:.0f}% CI: [{ci_lo:.6f}, {ci_hi:.6f}]"
            if components:
                status_msg += f" ({len(components)} components)"
            self.status.SetLabel(status_msg)
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
            self.status.SetLabel(f"Error: {e}")
        finally:
            self.btn_mc.Enable()

    def _on_run_mc_sheets(self, event):
        if not self._active_data:
            wx.MessageBox("No active sheet data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Running per-sheet MC...")
        self.btn_mc_sheets.Disable()
        wx.Yield()
        try:
            n = min(self.mc_n.GetValue(), 2000)
            unc = self.mc_unc.GetValue()
            ci = self._get_ci_level()
            self.sheet_mc_results.clear()
            total = len(self._active_data)
            for idx, (path, data) in enumerate(self._active_data.items()):
                name = path.rstrip("/").split("/")[-1] or "Root"
                self.status.SetLabel(f"Processing {name} ({idx+1}/{total})...")
                wx.Yield()
                comps = data.get("components", [])
                if not comps:
                    continue
                mc_comps = [{"ref": c.get("ref","?"), "value": c.get("value",""),
                             "class": c.get("class","Resistor"), "params": c.get("params",{}),
                             "override_lambda": c.get("override_lambda")}
                            for c in comps]
                mc_result, lam_s = monte_carlo_sheet(mc_comps, self.mission_hours,
                                                     n_simulations=n, uncertainty_percent=unc,
                                                     seed=42+idx, confidence_level=ci)
                self.sheet_mc_results[path] = SheetMCResult(path, mc_result, lam_s)
            self.status.SetLabel(f"Per-sheet MC: {total} sheets, {n} sims each")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.btn_mc_sheets.Enable()

    def _on_run_tornado(self, event):
        if not self._active_data:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Running tornado analysis...")
        self.btn_tornado.Disable()
        wx.Yield()
        try:
            pct = self.tornado_pct.GetValue() / 100.0
            mode = self.tornado_mode.GetSelection()

            if mode == 0:  # Sheet-level
                result = tornado_sheet_sensitivity(
                    self.sheet_data, self.mission_hours, pct, self.active_sheets)
                self.tornado_result = result
            else:  # Parameter-level
                result = tornado_parameter_sensitivity(
                    self.sheet_data, self.mission_hours, pct, self.active_sheets)
                self.param_tornado_result = result

            # Update chart
            chart_data = [(e.name, e.swing) for e in result.entries[:15]]
            max_swing = max(e.swing for e in result.entries) if result.entries else 1
            self.tornado_chart.set_data(chart_data, max_value=max_swing, x_label="Swing (FIT)")

            # Update list
            self.tornado_list.DeleteAllItems()
            for e in result.entries[:20]:
                idx = self.tornado_list.InsertItem(self.tornado_list.GetItemCount(), e.name)
                self.tornado_list.SetItem(idx, 1, f"{e.low_value:.2f}")
                self.tornado_list.SetItem(idx, 2, f"{e.base_value:.2f}")
                self.tornado_list.SetItem(idx, 3, f"{e.high_value:.2f}")
                self.tornado_list.SetItem(idx, 4, f"{e.swing:.2f}")

            mode_label = "sheet-level" if mode == 0 else "parameter-level"
            self.tornado_info.SetLabel(
                f"Tornado ({mode_label}, +/-{pct*100:.0f}%): {len(result.entries)} factors analyzed. "
                f"Base: {result.base_lambda_fit:.2f} FIT.")
            self.tornado_info.SetForegroundColour(Colors.SUCCESS)
            self.status.SetLabel(f"Tornado complete: {len(result.entries)} factors")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
            self.status.SetLabel(f"Error: {e}")
        finally:
            self.btn_tornado.Enable()

    def _on_run_design_margin(self, event):
        if not self._active_data:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Running design margin analysis...")
        self.btn_dm.Disable()
        wx.Yield()
        try:
            self.design_margin_result = design_margin_analysis(
                self.sheet_data, self.mission_hours, self.active_sheets)

            self.dm_list.DeleteAllItems()
            # Baseline row
            dm = self.design_margin_result
            idx = self.dm_list.InsertItem(0, "BASELINE")
            self.dm_list.SetItem(idx, 1, "Current design")
            self.dm_list.SetItem(idx, 2, f"{dm.baseline_lambda_fit:.2f}")
            self.dm_list.SetItem(idx, 3, f"{dm.baseline_reliability:.6f}")
            self.dm_list.SetItem(idx, 4, "---")
            self.dm_list.SetItem(idx, 5, "---")

            for s in dm.scenarios:
                idx = self.dm_list.InsertItem(self.dm_list.GetItemCount(), s.scenario_name)
                self.dm_list.SetItem(idx, 1, s.description)
                self.dm_list.SetItem(idx, 2, f"{s.lambda_fit:.2f}")
                self.dm_list.SetItem(idx, 3, f"{s.reliability:.6f}")
                self.dm_list.SetItem(idx, 4, f"{s.delta_lambda_pct:+.1f}%")
                self.dm_list.SetItem(idx, 5, f"{s.delta_reliability:+.6f}")

            self.status.SetLabel(f"Design margin: {len(dm.scenarios)} scenarios evaluated")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.btn_dm.Enable()

    def _on_run_criticality(self, event):
        if not self._active_data:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Running criticality analysis...")
        self.btn_crit.Disable()
        wx.Yield()
        try:
            top_n = self.crit_top_n.GetValue()
            perturb = self.crit_perturb.GetValue() / 100.0
            target_fields = self._get_target_fields()

            all_comps = []
            for data in self._active_data.values():
                all_comps.extend(data.get("components", []))

            self.criticality_results = analyze_board_criticality(
                all_comps, self.mission_hours, top_n, perturb, target_fields)

            self.crit_list.DeleteAllItems()
            for entry in self.criticality_results:
                ref = entry.get("reference", "?")
                comp_type = entry.get("component_type", "Unknown")
                base_fit = entry.get("base_lambda_fit", 0)
                fields = entry.get("fields", [])
                if fields:
                    top = max(fields, key=lambda f: abs(f.get("impact_pct", 0)))
                    idx = self.crit_list.InsertItem(self.crit_list.GetItemCount(), ref)
                    self.crit_list.SetItem(idx, 1, comp_type)
                    self.crit_list.SetItem(idx, 2, f"{base_fit:.2f}")
                    self.crit_list.SetItem(idx, 3, top.get("name", ""))
                    self.crit_list.SetItem(idx, 4, f"{top.get('elasticity',0):+.3f}")
                    self.crit_list.SetItem(idx, 5, f"{top.get('impact_pct',0):.1f}%")

            n_analyzed = len(self.criticality_results)
            field_note = "selected fields" if target_fields else "all fields"
            self.crit_info.SetLabel(f"Analyzed {n_analyzed} components ({field_note}, +/-{perturb*100:.0f}%).")
            self.crit_info.SetForegroundColour(Colors.SUCCESS)
            self.status.SetLabel(f"Criticality: {n_analyzed} components")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.btn_crit.Enable()

    def _update_contributions(self):
        if not self._active_data:
            return
        contribs = []
        total_lam = 0
        for path, data in self._active_data.items():
            lam = data.get("lambda", 0)
            if lam > 0:
                contribs.append((path.rstrip("/").split("/")[-1] or "Root", lam))
                total_lam += lam
        if total_lam == 0:
            return
        contribs.sort(key=lambda x: -x[1])
        chart_data = [(name, lam / total_lam) for name, lam in contribs]
        self.contrib_chart.set_data(chart_data, max_value=1.0, x_label="Relative Contribution")
        self.contrib_list.DeleteAllItems()
        cum = 0
        for i, (name, lam) in enumerate(contribs):
            pct = lam / total_lam * 100
            cum += pct
            idx = self.contrib_list.InsertItem(i, name)
            self.contrib_list.SetItem(idx, 1, f"{lam*1e9:.2f}")
            self.contrib_list.SetItem(idx, 2, f"{pct:.1f}%")
            self.contrib_list.SetItem(idx, 3, f"{cum:.1f}%")

    def _update_report(self):
        lines = []
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        lines.append("=" * 70)
        lines.append("           RELIABILITY ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"\nACTIVE SCOPE: {self._get_active_label()}")
        lines.append(f"\nSYSTEM PARAMETERS")
        lines.append("-" * 50)
        lines.append(f"  Mission: {years:.2f} years ({self.mission_hours:.0f} h)")
        lines.append(f"  System FIT: {self.system_lambda*1e9:.2f}")
        lines.append(f"  R(t): {r:.6f}")

        if self.mc_result:
            mc = self.mc_result
            ci_lo, ci_hi = mc.confidence_interval()
            cl_pct = mc.confidence_level * 100
            lines.append(f"\nMONTE CARLO ({mc.n_simulations:,} sims)")
            lines.append("-" * 50)
            lines.append(f"  Mean: {mc.mean:.6f}")
            lines.append(f"  Std:  {mc.std:.6f}")
            lines.append(f"  {cl_pct:.0f}% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")

        if self.tornado_result:
            lines.append(f"\nTORNADO (Sheet-level, +/-{self.tornado_result.perturbation_pct:.0f}%)")
            lines.append("-" * 50)
            for e in self.tornado_result.entries[:10]:
                lines.append(f"  {e.name:<25} swing={e.swing:.2f} FIT")

        if self.design_margin_result:
            dm = self.design_margin_result
            lines.append(f"\nDESIGN MARGIN")
            lines.append("-" * 50)
            for s in dm.scenarios:
                lines.append(f"  {s.scenario_name:<22} {s.lambda_fit:.2f} FIT ({s.delta_lambda_pct:+.1f}%)")

        if self.criticality_results:
            lines.append(f"\nCRITICALITY ({len(self.criticality_results)} components)")
            lines.append("-" * 50)
            for entry in self.criticality_results:
                ref = entry.get("reference", "?")
                fields = entry.get("fields", [])
                if fields:
                    top = max(fields, key=lambda f: abs(f.get("impact_pct", 0)))
                    lines.append(f"  {ref:<10} top: {top['name']} (impact={top['impact_pct']:.1f}%)")

        lines.append("\n" + "=" * 70)
        lines.append("  Eliot Abramo | KiCad Reliability Plugin v3.0.0 | IEC TR 62380")
        lines.append("=" * 70)
        self.report_text.SetValue("\n".join(lines))

    # =========================================================================
    # Export
    # =========================================================================
    def _on_export_html(self, event):
        dlg = wx.FileDialog(self, "Export HTML", wildcard="HTML (*.html)|*.html",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            html = self._generate_html()
            with open(dlg.GetPath(), "w", encoding="utf-8") as f:
                f.write(html)
            self.status.SetLabel(f"Exported: {dlg.GetPath()}")
        dlg.Destroy()

    def _on_export_csv(self, event):
        dlg = wx.FileDialog(self, "Export CSV", wildcard="CSV (*.csv)|*.csv",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            csv = self._generate_csv()
            with open(dlg.GetPath(), "w", encoding="utf-8") as f:
                f.write(csv)
            self.status.SetLabel(f"Exported: {dlg.GetPath()}")
        dlg.Destroy()

    def _generate_html(self) -> str:
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)

        mc_dict = None
        if self.mc_result:
            mc_dict = self.mc_result.to_dict()
            mc_dict["samples"] = self.mc_result.samples.tolist() if hasattr(self.mc_result.samples, "tolist") else list(self.mc_result.samples)

        sens_dict = self.sobol_result.to_dict() if self.sobol_result else None

        # Tornado
        tornado_dict = None
        tr = self.tornado_result or self.param_tornado_result
        if tr:
            tornado_dict = tr.to_dict()

        # Design margin
        dm_dict = self.design_margin_result.to_dict() if self.design_margin_result else None

        sheet_mc_dict = None
        if self.sheet_mc_results:
            sheet_mc_dict = {}
            for path, smc in self.sheet_mc_results.items():
                sheet_mc_dict[path] = smc.mc_result.to_dict()

        from pathlib import Path
        project_name = Path(self.project_path).name if self.project_path else "Reliability Report"

        report_data = ReportData(
            project_name=project_name,
            mission_hours=self.mission_hours,
            mission_years=years,
            n_cycles=self.n_cycles,
            delta_t=self.delta_t,
            system_reliability=r,
            system_lambda=self.system_lambda,
            system_mttf_hours=1.0 / self.system_lambda if self.system_lambda > 0 else float("inf"),
            sheets=self._active_data,
            blocks=[],
            monte_carlo=mc_dict,
            sensitivity=sens_dict,
            sheet_mc=sheet_mc_dict,
            criticality=self.criticality_results if self.criticality_results else None,
            tornado=tornado_dict,
            design_margin=dm_dict,
        )

        generator = ReportGenerator(logo_path=self.logo_path, logo_mime=self.logo_mime)
        return generator.generate_html(report_data)

    def _generate_csv(self) -> str:
        lines = ["Section,Item,Parameter,Value"]
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        lines.append(f"System,Summary,Lambda_FIT,{self.system_lambda*1e9:.6f}")
        lines.append(f"System,Summary,Reliability,{r:.6f}")
        if self.mc_result:
            mc = self.mc_result
            ci_lo, ci_hi = mc.confidence_interval()
            lines.append(f"MonteCarlo,Summary,Mean,{mc.mean:.6f}")
            lines.append(f"MonteCarlo,Summary,StdDev,{mc.std:.6f}")
            lines.append(f"MonteCarlo,Summary,CI_Lower,{ci_lo:.6f}")
            lines.append(f"MonteCarlo,Summary,CI_Upper,{ci_hi:.6f}")
            lines.append(f"MonteCarlo,Summary,Confidence,{mc.confidence_level:.2f}")
        for path, data in self._active_data.items():
            name = path.rstrip("/").split("/")[-1] or "Root"
            for c in data.get("components", []):
                ref = c.get("ref", "?")
                ovr = "override" if c.get("override_lambda") is not None else "calc"
                lines.append(f"Component,{ref},Lambda_FIT,{c.get('lambda',0)*1e9:.6f},{ovr}")
        return "\n".join(lines)
