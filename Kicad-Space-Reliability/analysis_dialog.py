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
from .ecss_fields import (
    get_category_fields, infer_category_from_class, get_display_group,
    get_ordered_categories_present, math_type_to_ecss,
)
from .budget_allocation import allocate_budget, BudgetAllocationResult
from .derating_engine import compute_derating_guidance, DeratingResult
from .component_swap import (
    analyze_package_swaps, analyze_type_swaps,
    quick_swap_comparison, rank_all_swaps, SwapAnalysisResult,
)
from .growth_tracking import (
    create_snapshot, save_snapshot, load_snapshots,
    compare_revisions, build_growth_timeline,
    ReliabilitySnapshot, RevisionComparison,
)
from .correlated_mc import (
    correlated_monte_carlo, CorrelationGroup, CorrelatedMCResult,
    auto_group_by_sheet, auto_group_by_type, auto_group_all_on_board,
)
from .mission_profile import MissionProfile, MISSION_TEMPLATES


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
        self.excluded_types: set = set()  # Component types excluded from analysis
        self._type_checkboxes: Dict[str, wx.CheckBox] = {}  # type_name -> checkbox
        self.budget_result: Optional[BudgetAllocationResult] = None
        self.derating_result: Optional[DeratingResult] = None
        self.swap_results: List[Dict] = []
        self.correlated_mc_result: Optional[CorrelatedMCResult] = None
        self.growth_snapshots: List[Dict] = []
        self.growth_comparisons: List[Dict] = []

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
        self.notebook.AddPage(self._create_budget_tab(), "Budget Allocation")
        self.notebook.AddPage(self._create_derating_tab(), "Derating Guidance")
        self.notebook.AddPage(self._create_swap_tab(), "Component Swap")
        self.notebook.AddPage(self._create_correlated_mc_tab(), "Correlated MC")
        self.notebook.AddPage(self._create_growth_tab(), "Growth Tracking")
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

        # Explanation panel
        expl_panel = wx.Panel(panel)
        expl_panel.SetBackgroundColour(wx.Colour(239, 246, 255))
        expl_sizer = wx.BoxSizer(wx.VERTICAL)
        expl_title = wx.StaticText(expl_panel, label="One-At-a-Time (OAT) Sensitivity Analysis")
        expl_title.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        expl_title.SetForegroundColour(Colors.PRIMARY)
        expl_sizer.Add(expl_title, 0, wx.ALL, 6)
        expl_text = (
            "For each factor (sheet or parameter), the analysis perturbs it by +/- X% while holding "
            "all others constant, then measures the resulting change in system FIT. This is the standard "
            "one-at-a-time (OAT) deterministic sensitivity method per IEC 60300-3-1.\n\n"
            "How to use:\n"
            "  Sheet-level: identifies which subsystem (PCB sheet) contributes most to system FIT.\n"
            "    -> Focus redesign on the sheet with the largest swing.\n"
            "  Parameter-level: identifies which design parameter (temperature, power, ...) has the\n"
            "    greatest leverage across all components. -> Prioritise thermal management or derating."
        )
        expl = wx.StaticText(expl_panel, label=expl_text)
        expl.SetForegroundColour(Colors.TEXT_MEDIUM)
        expl.Wrap(900)
        expl_sizer.Add(expl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        expl_panel.SetSizer(expl_sizer)
        main.Add(expl_panel, 0, wx.EXPAND | wx.ALL, 8)

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

        # Explanation panel
        expl_panel = wx.Panel(panel)
        expl_panel.SetBackgroundColour(wx.Colour(255, 251, 235))
        expl_sizer = wx.BoxSizer(wx.VERTICAL)
        expl_title = wx.StaticText(expl_panel, label="Design Margin / What-If Scenario Analysis")
        expl_title.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        expl_title.SetForegroundColour(wx.Colour(146, 64, 14))
        expl_sizer.Add(expl_title, 0, wx.ALL, 6)
        expl_text = (
            "This tab evaluates your design's robustness against realistic environmental and "
            "operational changes. Each scenario modifies one design parameter globally and "
            "recomputes every component's lambda from scratch using the IEC TR 62380 models.\n\n"
            "How to use:\n"
            "  - 'Temp +10/+20 C': Will the board survive a hotter enclosure? If Delta Lambda > 10%,\n"
            "    consider better thermal management or component derating.\n"
            "  - 'Power derate 50/70%': Quantifies the reliability gain from running components below\n"
            "    rated power -- a key ECSS/MIL derating strategy.\n"
            "  - 'Thermal cycles x2 / Delta-T x2': Evaluates vulnerability to harsher thermal profiles\n"
            "    (vibration, altitude changes, day/night cycling).\n"
            "  - '50% duty cycle': Shows impact of reducing operating time (sleep modes, intermittent use)."
        )
        expl = wx.StaticText(expl_panel, label=expl_text)
        expl.SetForegroundColour(Colors.TEXT_MEDIUM)
        expl.Wrap(900)
        expl_sizer.Add(expl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        expl_panel.SetSizer(expl_sizer)
        main.Add(expl_panel, 0, wx.EXPAND | wx.ALL, 8)

        ctrl_panel = wx.Panel(panel)
        ctrl_panel.SetBackgroundColour(Colors.BG_WHITE)
        ctrl = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_dm = wx.Button(ctrl_panel, label="Run Design Margin Analysis")
        self.btn_dm.SetBackgroundColour(Colors.PRIMARY)
        self.btn_dm.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_dm.Bind(wx.EVT_BUTTON, self._on_run_design_margin)
        ctrl.Add(self.btn_dm, 0, wx.ALL, 8)
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
    # Contributions Tab (with component type exclusion)
    # =========================================================================
    def _create_contributions_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        # Component type filter panel
        filter_panel = wx.Panel(panel)
        filter_panel.SetBackgroundColour(Colors.BG_WHITE)
        filter_sizer = wx.BoxSizer(wx.VERTICAL)
        filter_label = wx.StaticText(filter_panel, label="Include Component Types in Analysis:")
        filter_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        filter_sizer.Add(filter_label, 0, wx.ALL, 6)

        filter_help = wx.StaticText(filter_panel,
            label="Uncheck a type to exclude it from ALL analyses (Monte Carlo, Tornado, Design Margin, Criticality). "
                  "Useful for ignoring connectors, test points, or mechanical parts.")
        filter_help.SetForegroundColour(Colors.TEXT_MEDIUM)
        filter_help.Wrap(900)
        filter_sizer.Add(filter_help, 0, wx.LEFT | wx.BOTTOM, 6)

        type_row = wx.WrapSizer(wx.HORIZONTAL)
        types_in_data = self._get_component_types_in_data()
        self._type_checkboxes = {}
        for type_name in sorted(types_in_data):
            cb = wx.CheckBox(filter_panel, label=type_name)
            cb.SetValue(True)
            cb.Bind(wx.EVT_CHECKBOX, self._on_type_filter_change)
            type_row.Add(cb, 0, wx.ALL, 4)
            self._type_checkboxes[type_name] = cb
        filter_sizer.Add(type_row, 0, wx.LEFT, 12)
        filter_panel.SetSizer(filter_sizer)
        main.Add(filter_panel, 0, wx.EXPAND | wx.ALL, 8)

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
    # Criticality Tab (with improved field picker)
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

        # Field picker with proper IC support
        fp_panel = wx.Panel(panel)
        fp_panel.SetBackgroundColour(Colors.BG_WHITE)
        fp_sizer = wx.BoxSizer(wx.VERTICAL)
        fp_label = wx.StaticText(fp_panel, label="Field Selection per Component Category (uncheck to exclude from criticality analysis):")
        fp_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        fp_sizer.Add(fp_label, 0, wx.ALL, 8)

        # Detect categories present in active data using the improved mapper
        self._categories_in_data = self._get_ecss_categories_in_data()
        self._field_checkboxes = {}  # {category_key: {field_name: wx.CheckBox}}

        scroll = scrolled.ScrolledPanel(fp_panel, size=(-1, 180))
        scroll.SetBackgroundColour(Colors.BG_WHITE)
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        # Use canonical ordering so ICs appear first
        ordered_cats = get_ordered_categories_present(self._categories_in_data)

        for cat_key in ordered_cats:
            cat_def = get_category_fields(cat_key)
            display = get_display_group(cat_key)
            fields = cat_def.get("fields", {})
            if not fields:
                # Even categories without ECSS JSON fields should appear
                # (e.g. IC types that use reliability_math fields)
                no_fields_label = wx.StaticText(scroll, label=f"  {display}:  (uses reliability_math model fields)")
                no_fields_label.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
                no_fields_label.SetForegroundColour(Colors.TEXT_MEDIUM)
                scroll_sizer.Add(no_fields_label, 0, wx.TOP | wx.LEFT, 4)
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
    # Budget Allocation Tab (Phase 3)
    # =========================================================================
    def _create_budget_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        # Controls
        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_sizer.Add(wx.StaticText(panel, label="Target R:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.budget_target_r = wx.TextCtrl(panel, value="0.999", size=(80, -1))
        ctrl_sizer.Add(self.budget_target_r, 0, wx.RIGHT, 15)

        ctrl_sizer.Add(wx.StaticText(panel, label="Strategy:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.budget_strategy = wx.Choice(panel, choices=["Proportional (ARINC)", "Equal", "Complexity-Weighted", "Criticality-Weighted"])
        self.budget_strategy.SetSelection(0)
        ctrl_sizer.Add(self.budget_strategy, 0, wx.RIGHT, 15)

        ctrl_sizer.Add(wx.StaticText(panel, label="Margin:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.budget_margin = wx.SpinCtrl(panel, min=0, max=50, initial=10, size=(60, -1))
        ctrl_sizer.Add(self.budget_margin, 0, wx.RIGHT, 5)
        ctrl_sizer.Add(wx.StaticText(panel, label="%"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 15)

        self.btn_budget = wx.Button(panel, label="Allocate Budget")
        self.btn_budget.Bind(wx.EVT_BUTTON, self._on_run_budget)
        ctrl_sizer.Add(self.btn_budget, 0)
        main.Add(ctrl_sizer, 0, wx.EXPAND | wx.ALL, 12)

        # Results list
        self.budget_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.budget_list.SetBackgroundColour(Colors.BG_WHITE)
        self.budget_list.InsertColumn(0, "Reference", width=90)
        self.budget_list.InsertColumn(1, "Type", width=110)
        self.budget_list.InsertColumn(2, "Actual FIT", width=85)
        self.budget_list.InsertColumn(3, "Budget FIT", width=85)
        self.budget_list.InsertColumn(4, "Margin FIT", width=85)
        self.budget_list.InsertColumn(5, "Utilization", width=85)
        self.budget_list.InsertColumn(6, "Status", width=80)
        main.Add(self.budget_list, 1, wx.EXPAND | wx.ALL, 8)

        self.budget_info = wx.StaticText(panel, label="Set a system reliability target and allocate budgets to components.")
        self.budget_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.budget_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    def _on_run_budget(self, event):
        filtered = self._get_filtered_active_data()
        if not filtered:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Running budget allocation...")
        self.btn_budget.Disable()
        wx.Yield()
        try:
            target_r = float(self.budget_target_r.GetValue())
            strategy_map = {0: "proportional", 1: "equal", 2: "complexity", 3: "criticality"}
            strategy = strategy_map.get(self.budget_strategy.GetSelection(), "proportional")
            margin = self.budget_margin.GetValue()

            self.budget_result = allocate_budget(
                filtered, self.mission_hours, target_reliability=target_r,
                strategy=strategy, active_sheets=list(filtered.keys()),
                margin_percent=margin)

            self.budget_list.DeleteAllItems()
            for sb in self.budget_result.sheet_budgets:
                for cb in sb.component_budgets:
                    idx = self.budget_list.InsertItem(self.budget_list.GetItemCount(), cb.reference)
                    self.budget_list.SetItem(idx, 1, cb.component_type[:18])
                    self.budget_list.SetItem(idx, 2, f"{cb.actual_fit:.2f}")
                    self.budget_list.SetItem(idx, 3, f"{cb.budget_fit:.2f}")
                    self.budget_list.SetItem(idx, 4, f"{cb.margin_fit:+.2f}")
                    self.budget_list.SetItem(idx, 5, f"{cb.utilization*100:.0f}%")
                    status = "PASS" if cb.within_budget else "OVER"
                    self.budget_list.SetItem(idx, 6, status)

            br = self.budget_result
            status_icon = "PASS" if br.system_within_budget else "FAIL"
            info = (f"Budget [{status_icon}]: Target {target_r:.4f} = {br.target_fit:.1f} FIT, "
                    f"Actual {br.actual_fit:.1f} FIT, "
                    f"Margin {br.system_margin_fit:+.1f} FIT ({br.system_margin_percent:+.1f}%). "
                    f"{br.components_over_budget}/{br.total_components} components over budget.")
            self.budget_info.SetLabel(info)
            self.budget_info.SetForegroundColour(Colors.SUCCESS if br.system_within_budget else Colors.DANGER)
            self.status.SetLabel(f"Budget allocation complete: {br.total_components} components")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.btn_budget.Enable()

    # =========================================================================
    # Derating Guidance Tab (Phase 4)
    # =========================================================================
    def _create_derating_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_sizer.Add(wx.StaticText(panel, label="Target FIT:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.derate_target = wx.TextCtrl(panel, value="", size=(80, -1))
        self.derate_target.SetHint("auto")
        ctrl_sizer.Add(self.derate_target, 0, wx.RIGHT, 15)

        ctrl_sizer.Add(wx.StaticText(panel, label="Top N:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.derate_top_n = wx.SpinCtrl(panel, min=3, max=30, initial=10, size=(60, -1))
        ctrl_sizer.Add(self.derate_top_n, 0, wx.RIGHT, 15)

        self.btn_derate = wx.Button(panel, label="Generate Recommendations")
        self.btn_derate.Bind(wx.EVT_BUTTON, self._on_run_derating)
        ctrl_sizer.Add(self.btn_derate, 0)
        main.Add(ctrl_sizer, 0, wx.EXPAND | wx.ALL, 12)

        self.derate_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.derate_list.SetBackgroundColour(Colors.BG_WHITE)
        self.derate_list.InsertColumn(0, "#", width=35)
        self.derate_list.InsertColumn(1, "Reference", width=80)
        self.derate_list.InsertColumn(2, "Parameter", width=110)
        self.derate_list.InsertColumn(3, "Current", width=80)
        self.derate_list.InsertColumn(4, "Required", width=80)
        self.derate_list.InsertColumn(5, "Change", width=80)
        self.derate_list.InsertColumn(6, "Sys FIT Saved", width=90)
        self.derate_list.InsertColumn(7, "Feasibility", width=80)
        main.Add(self.derate_list, 1, wx.EXPAND | wx.ALL, 8)

        self.derate_info = wx.StaticText(panel, label="Compute required parameter values to meet reliability targets.")
        self.derate_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.derate_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    def _on_run_derating(self, event):
        filtered = self._get_filtered_active_data()
        if not filtered:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Computing derating guidance...")
        self.btn_derate.Disable()
        wx.Yield()
        try:
            target_text = self.derate_target.GetValue().strip()
            if target_text:
                target_fit = float(target_text)
            else:
                from .reliability_math import lambda_from_reliability
                target_fit = lambda_from_reliability(0.999, self.mission_hours) * 1e9

            top_n = self.derate_top_n.GetValue()
            self.derating_result = compute_derating_guidance(
                filtered, self.mission_hours, target_fit,
                active_sheets=list(filtered.keys()), top_n=top_n)

            self.derate_list.DeleteAllItems()
            for rec in self.derating_result.recommendations[:30]:
                idx = self.derate_list.InsertItem(self.derate_list.GetItemCount(), str(rec.priority))
                self.derate_list.SetItem(idx, 1, rec.reference)
                self.derate_list.SetItem(idx, 2, rec.parameter)
                self.derate_list.SetItem(idx, 3, f"{rec.current_value:.2f}")
                self.derate_list.SetItem(idx, 4, f"{rec.required_value:.2f}")
                self.derate_list.SetItem(idx, 5, f"{rec.change_percent:+.1f}%")
                self.derate_list.SetItem(idx, 6, f"{rec.system_fit_reduction:.2f}")
                self.derate_list.SetItem(idx, 7, rec.feasibility)

            self.derate_info.SetLabel(self.derating_result.summary[:200])
            self.status.SetLabel(f"Derating: {len(self.derating_result.recommendations)} recommendations")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.btn_derate.Enable()

    # =========================================================================
    # Component Swap Tab (Phase 5)
    # =========================================================================
    def _create_swap_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_sizer.Add(wx.StaticText(panel, label="Max per component:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.swap_max = wx.SpinCtrl(panel, min=1, max=20, initial=5, size=(60, -1))
        ctrl_sizer.Add(self.swap_max, 0, wx.RIGHT, 15)

        self.btn_swap = wx.Button(panel, label="Rank All Swaps")
        self.btn_swap.Bind(wx.EVT_BUTTON, self._on_run_swap)
        ctrl_sizer.Add(self.btn_swap, 0)
        main.Add(ctrl_sizer, 0, wx.EXPAND | wx.ALL, 12)

        self.swap_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.swap_list.SetBackgroundColour(Colors.BG_WHITE)
        self.swap_list.InsertColumn(0, "Reference", width=80)
        self.swap_list.InsertColumn(1, "Type", width=100)
        self.swap_list.InsertColumn(2, "Swap", width=90)
        self.swap_list.InsertColumn(3, "Description", width=230)
        self.swap_list.InsertColumn(4, "Delta FIT", width=85)
        self.swap_list.InsertColumn(5, "Change %", width=80)
        self.swap_list.InsertColumn(6, "New Sys FIT", width=90)
        main.Add(self.swap_list, 1, wx.EXPAND | wx.ALL, 8)

        self.swap_info = wx.StaticText(panel, label="Find the single component change that most improves system reliability.")
        self.swap_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.swap_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    def _on_run_swap(self, event):
        filtered = self._get_filtered_active_data()
        if not filtered:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Ranking all component swaps...")
        self.btn_swap.Disable()
        wx.Yield()
        try:
            all_comps = []
            for data in filtered.values():
                all_comps.extend(data.get("components", []))

            system_fit = sum(float(d.get("lambda", 0)) for d in filtered.values()) * 1e9
            max_per = self.swap_max.GetValue()

            self.swap_results = rank_all_swaps(all_comps, system_fit, max_per)

            self.swap_list.DeleteAllItems()
            for r in self.swap_results[:50]:
                idx = self.swap_list.InsertItem(self.swap_list.GetItemCount(), r["reference"])
                self.swap_list.SetItem(idx, 1, r["component_type"][:18])
                self.swap_list.SetItem(idx, 2, r["swap_type"])
                self.swap_list.SetItem(idx, 3, r["description"][:35])
                self.swap_list.SetItem(idx, 4, f"{r['delta_fit']:.2f}")
                self.swap_list.SetItem(idx, 5, f"{r['delta_percent']:.1f}%")
                self.swap_list.SetItem(idx, 6, f"{r['new_system_fit']:.1f}")

            n = len(self.swap_results)
            if n > 0:
                best = self.swap_results[0]
                self.swap_info.SetLabel(
                    f"Found {n} improvement opportunities. Best: {best['reference']} "
                    f"({best['description'][:40]}) saves {abs(best['delta_fit']):.2f} FIT.")
                self.swap_info.SetForegroundColour(Colors.SUCCESS)
            else:
                self.swap_info.SetLabel("No improvements found from single-parameter swaps.")
            self.status.SetLabel(f"Swap analysis: {n} improvements found")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.btn_swap.Enable()

    # =========================================================================
    # Correlated Monte Carlo Tab (Phase 7)
    # =========================================================================
    def _create_correlated_mc_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_sizer.Add(wx.StaticText(panel, label="Grouping:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.corr_group_choice = wx.Choice(panel,
            choices=["By Sheet (rho=0.80)", "By Type (rho=0.60)", "All on Board (rho=0.50)"],
            size=(180, -1))
        self.corr_group_choice.SetSelection(0)
        ctrl_sizer.Add(self.corr_group_choice, 0, wx.RIGHT, 15)

        ctrl_sizer.Add(wx.StaticText(panel, label="Simulations:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.corr_sims = wx.SpinCtrl(panel, min=500, max=50000, initial=5000, size=(80, -1))
        ctrl_sizer.Add(self.corr_sims, 0, wx.RIGHT, 15)

        ctrl_sizer.Add(wx.StaticText(panel, label="Uncertainty %:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.corr_unc = wx.SpinCtrlDouble(panel, min=5, max=50, initial=20.0, inc=5.0, size=(60, -1))
        self.corr_unc.SetDigits(0)
        ctrl_sizer.Add(self.corr_unc, 0, wx.RIGHT, 15)

        self.btn_corr_mc = wx.Button(panel, label="Run Correlated MC")
        self.btn_corr_mc.Bind(wx.EVT_BUTTON, self._on_run_correlated_mc)
        ctrl_sizer.Add(self.btn_corr_mc, 0)
        main.Add(ctrl_sizer, 0, wx.EXPAND | wx.ALL, 12)

        # Results display
        self.corr_mc_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.corr_mc_list.SetBackgroundColour(Colors.BG_WHITE)
        self.corr_mc_list.InsertColumn(0, "Metric", width=180)
        self.corr_mc_list.InsertColumn(1, "Independent", width=130)
        self.corr_mc_list.InsertColumn(2, "Correlated", width=130)
        self.corr_mc_list.InsertColumn(3, "Ratio/Delta", width=130)
        main.Add(self.corr_mc_list, 1, wx.EXPAND | wx.ALL, 8)

        self.corr_mc_info = wx.StaticText(panel,
            label="Compares independent vs. correlated uncertainty to assess if ignoring correlations underestimates risk.")
        self.corr_mc_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.corr_mc_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    def _on_run_correlated_mc(self, event):
        import numpy as np
        filtered = self._get_filtered_active_data()
        if not filtered:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Running correlated Monte Carlo...")
        self.btn_corr_mc.Disable()
        wx.Yield()
        try:
            all_comps = []
            for data in filtered.values():
                all_comps.extend(data.get("components", []))
            if not all_comps:
                wx.MessageBox("No components found.", "No Data", wx.OK | wx.ICON_WARNING)
                return

            lambdas = np.array([c.get("lambda", 0) for c in all_comps])
            refs = [c.get("ref", "?") for c in all_comps]

            # Select grouping strategy
            sel = self.corr_group_choice.GetSelection()
            if sel == 0:
                groups = auto_group_by_sheet(filtered, 0.80)
            elif sel == 1:
                groups = auto_group_by_type(filtered, 0.60)
            else:
                groups = [CorrelationGroup(
                    name="All on Board", component_refs=refs,
                    rho=0.50, description="All components share board-level environment")]

            n_sims = self.corr_sims.GetValue()
            unc = self.corr_unc.GetValue()

            self.correlated_mc_result = correlated_monte_carlo(
                lambdas, refs, groups, self.mission_hours,
                n_simulations=n_sims, uncertainty_percent=unc, seed=42)

            # Display results
            r = self.correlated_mc_result
            self.corr_mc_list.DeleteAllItems()
            rows = [
                ("Mean Reliability", f"{r.independent_mean:.6f}", f"{r.correlated_mean:.6f}", f"{r.correlated_mean - r.independent_mean:+.6f}"),
                ("Std Deviation", f"{r.independent_std:.6f}", f"{r.correlated_std:.6f}", f"{r.std_ratio:.3f}x"),
                ("5th Percentile", f"{r.independent_ci[0]:.6f}", f"{r.correlated_ci[0]:.6f}", f"{r.correlated_ci[0] - r.independent_ci[0]:+.6f}"),
                ("95th Percentile", f"{r.independent_ci[1]:.6f}", f"{r.correlated_ci[1]:.6f}", f"{r.correlated_ci[1] - r.independent_ci[1]:+.6f}"),
                ("Correlation Groups", "-", f"{r.n_groups}", "-"),
                ("Simulations", f"{n_sims:,}", f"{n_sims:,}", "-"),
                ("Variance Impact", "-", f"{r.variance_impact.title()}", f"{r.std_ratio:.3f}x ratio"),
            ]
            for label, indep, corr, delta in rows:
                idx = self.corr_mc_list.InsertItem(self.corr_mc_list.GetItemCount(), label)
                self.corr_mc_list.SetItem(idx, 1, indep)
                self.corr_mc_list.SetItem(idx, 2, corr)
                self.corr_mc_list.SetItem(idx, 3, delta)

            if r.std_ratio > 1.05:
                self.corr_mc_info.SetLabel(
                    f"Correlation INCREASES uncertainty by {r.std_ratio:.2f}x. "
                    f"Independent MC underestimates tail risk.")
                self.corr_mc_info.SetForegroundColour(Colors.WARNING)
            elif r.std_ratio < 0.95:
                self.corr_mc_info.SetLabel(
                    f"Correlation REDUCES uncertainty ({r.std_ratio:.2f}x). "
                    f"Independent MC overestimates variability.")
                self.corr_mc_info.SetForegroundColour(Colors.SUCCESS)
            else:
                self.corr_mc_info.SetLabel("Correlation has minimal impact on uncertainty bounds.")
                self.corr_mc_info.SetForegroundColour(Colors.TEXT_MEDIUM)

            self.status.SetLabel(f"Correlated MC: {r.n_groups} groups, std ratio = {r.std_ratio:.3f}x ({r.variance_impact})")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.btn_corr_mc.Enable()

    # =========================================================================
    # Growth Tracking Tab (Phase 6)
    # =========================================================================
    def _create_growth_tab(self):
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ctrl_sizer.Add(wx.StaticText(panel, label="Version label:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.growth_version = wx.TextCtrl(panel, value="v1.0", size=(80, -1))
        ctrl_sizer.Add(self.growth_version, 0, wx.RIGHT, 10)

        ctrl_sizer.Add(wx.StaticText(panel, label="Notes:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.growth_notes = wx.TextCtrl(panel, value="", size=(200, -1))
        ctrl_sizer.Add(self.growth_notes, 1, wx.RIGHT, 10)

        self.btn_snapshot = wx.Button(panel, label="Take Snapshot")
        self.btn_snapshot.Bind(wx.EVT_BUTTON, self._on_take_snapshot)
        ctrl_sizer.Add(self.btn_snapshot, 0, wx.RIGHT, 8)

        self.btn_compare = wx.Button(panel, label="Compare Latest")
        self.btn_compare.Bind(wx.EVT_BUTTON, self._on_compare_snapshots)
        ctrl_sizer.Add(self.btn_compare, 0)
        main.Add(ctrl_sizer, 0, wx.EXPAND | wx.ALL, 12)

        # Snapshots list
        self.growth_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.growth_list.SetBackgroundColour(Colors.BG_WHITE)
        self.growth_list.InsertColumn(0, "Version", width=80)
        self.growth_list.InsertColumn(1, "Timestamp", width=140)
        self.growth_list.InsertColumn(2, "Components", width=80)
        self.growth_list.InsertColumn(3, "System FIT", width=90)
        self.growth_list.InsertColumn(4, "R(t)", width=120)
        self.growth_list.InsertColumn(5, "Notes", width=200)
        main.Add(self.growth_list, 1, wx.EXPAND | wx.ALL, 8)

        self.growth_info = wx.StaticText(panel,
            label="Track reliability across design revisions. Take snapshots at each milestone.")
        self.growth_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.growth_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    def _on_take_snapshot(self, event):
        filtered = self._get_filtered_active_data()
        if not filtered:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        try:
            version = self.growth_version.GetValue().strip() or f"v{len(self.growth_snapshots)+1}"
            notes = self.growth_notes.GetValue().strip()

            snap = create_snapshot(
                filtered, self.system_lambda, self.mission_hours,
                version_label=version, notes=notes)

            snap_dict = snap.to_dict()
            self.growth_snapshots.append(snap_dict)

            # Update list
            idx = self.growth_list.InsertItem(self.growth_list.GetItemCount(), version)
            self.growth_list.SetItem(idx, 1, snap_dict.get("timestamp", "")[:19])
            self.growth_list.SetItem(idx, 2, str(snap_dict.get("n_components", 0)))
            self.growth_list.SetItem(idx, 3, f"{snap_dict.get('system_fit', 0):.2f}")
            self.growth_list.SetItem(idx, 4, f"{snap_dict.get('system_reliability', 0):.6f}")
            self.growth_list.SetItem(idx, 5, notes)

            # Auto-increment version
            try:
                parts = version.replace("v", "").split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                self.growth_version.SetValue("v" + ".".join(parts))
            except (ValueError, IndexError):
                pass

            self.growth_info.SetLabel(
                f"Snapshot '{version}' captured: {snap.n_components} components, "
                f"{snap.system_fit:.2f} FIT, R = {snap.system_reliability:.6f}")
            self.growth_info.SetForegroundColour(Colors.SUCCESS)
            self.status.SetLabel(f"Growth: {len(self.growth_snapshots)} snapshots captured")
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)

    def _on_compare_snapshots(self, event):
        if len(self.growth_snapshots) < 2:
            wx.MessageBox("Need at least 2 snapshots to compare.", "Insufficient Data", wx.OK | wx.ICON_WARNING)
            return
        try:
            from .growth_tracking import ReliabilitySnapshot
            snap_a = ReliabilitySnapshot.from_dict(self.growth_snapshots[-2])
            snap_b = ReliabilitySnapshot.from_dict(self.growth_snapshots[-1])
            comp = compare_revisions(snap_a, snap_b)

            comp_dict = {
                "from_version": snap_a.version_label,
                "to_version": snap_b.version_label,
                "system_delta_fit": comp.system_delta_fit,
                "components_added": comp.components_added,
                "components_removed": comp.components_removed,
                "components_improved": comp.components_improved,
                "components_degraded": comp.components_degraded,
                "top_changes": [
                    {"ref": c.reference, "change_type": c.change_type,
                     "delta_fit": c.delta_fit}
                    for c in (comp.top_improvements + comp.top_degradations)[:10]
                ]
            }
            self.growth_comparisons.append(comp_dict)

            direction = "improved" if comp.system_delta_fit < 0 else "degraded" if comp.system_delta_fit > 0 else "unchanged"
            self.growth_info.SetLabel(
                f"Comparison {snap_a.version_label} -> {snap_b.version_label}: "
                f"{comp.system_delta_fit:+.3f} FIT ({direction}), "
                f"+{comp.components_added} added, -{comp.components_removed} removed, "
                f"{comp.components_improved} improved, {comp.components_degraded} degraded")
            color = Colors.SUCCESS if comp.system_delta_fit <= 0 else Colors.WARNING
            self.growth_info.SetForegroundColour(color)
            self._update_report()
        except Exception as e:
            import traceback; traceback.print_exc()
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)

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
        btn_pdf = wx.Button(panel, label="Export PDF")
        btn_pdf.Bind(wx.EVT_BUTTON, self._on_export_pdf)
        btn_row.Add(btn_pdf, 0)
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

    def _get_component_types_in_data(self) -> set:
        """Find which component types (reliability_math names) are in the data."""
        types = set()
        for data in self._active_data.values():
            for comp in data.get("components", []):
                cls = comp.get("class", "")
                if cls:
                    types.add(cls)
        return types

    def _get_ecss_categories_in_data(self) -> set:
        """Find which ECSS categories are present in the active data.
        Uses the improved math_type_to_ecss for proper IC mapping."""
        cats = set()
        for data in self._active_data.values():
            for comp in data.get("components", []):
                cls = comp.get("class", "")
                if cls:
                    cats.add(math_type_to_ecss(cls))
        return cats

    def _on_type_filter_change(self, event):
        """Update excluded_types set and refilter active data."""
        self.excluded_types = {
            type_name for type_name, cb in self._type_checkboxes.items()
            if not cb.GetValue()
        }
        self._update_contributions()

    def _get_filtered_active_data(self) -> Dict:
        """Return active data with excluded component types removed."""
        if not self.excluded_types:
            return self._active_data
        filtered = {}
        for path, data in self._active_data.items():
            new_comps = [c for c in data.get("components", [])
                         if c.get("class", "") not in self.excluded_types]
            if new_comps:
                new_data = dict(data)
                new_data["components"] = new_comps
                new_data["lambda"] = sum(c.get("lambda", 0) for c in new_comps)
                filtered[path] = new_data
        return filtered

    def _get_categories_in_data(self) -> set:
        """Legacy: Find which ECSS categories are present in the active data."""
        return self._get_ecss_categories_in_data()

    def _get_target_fields(self) -> Optional[Dict[str, List[str]]]:
        """Build target_fields dict from checkbox state. Returns None if all checked."""
        result = {}
        all_checked = True
        for cat_key, fields in self._field_checkboxes.items():
            selected = [fname for fname, cb in fields.items() if cb.GetValue()]
            if len(selected) < len(fields):
                all_checked = False
            result[cat_key] = selected

        # Map component class names to ECSS category keys for the target_fields
        expanded = {}
        for cat_key, field_list in result.items():
            expanded[cat_key] = field_list
        return None if all_checked else expanded

    def _extract_components_for_mc(self) -> List[Dict]:
        """Extract component data from active sheet_data (respecting type exclusion)."""
        components = []
        data_source = self._get_filtered_active_data()
        for sheet_path, data in data_source.items():
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
        filtered = self._get_filtered_active_data()
        if not filtered:
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
                    filtered, self.mission_hours, pct, list(filtered.keys()))
                self.tornado_result = result
            else:  # Parameter-level
                result = tornado_parameter_sensitivity(
                    filtered, self.mission_hours, pct, list(filtered.keys()))
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
        filtered = self._get_filtered_active_data()
        if not filtered:
            wx.MessageBox("No active data.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        self.status.SetLabel("Running design margin analysis...")
        self.btn_dm.Disable()
        wx.Yield()
        try:
            self.design_margin_result = design_margin_analysis(
                filtered, self.mission_hours, list(filtered.keys()))

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
        filtered = self._get_filtered_active_data()
        if not filtered:
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
            for data in filtered.values():
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
        data_source = self._get_filtered_active_data()
        if not data_source:
            return
        contribs = []
        total_lam = 0
        for path, data in data_source.items():
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

        if self.budget_result:
            br = self.budget_result
            lines.append(f"\nBUDGET ALLOCATION ({br.strategy.title()})")
            lines.append("-" * 50)
            lines.append(f"  Target: {br.target_fit:.1f} FIT | Actual: {br.actual_fit:.2f} FIT | Over: {br.components_over_budget}")

        if self.derating_result:
            dr = self.derating_result
            lines.append(f"\nDERATING GUIDANCE ({len(dr.recommendations)} recommendations)")
            lines.append("-" * 50)
            lines.append(f"  Gap: {dr.system_gap_fit:+.2f} FIT | Feasible: {dr.n_feasible}")

        if self.correlated_mc_result:
            cm = self.correlated_mc_result
            lines.append(f"\nCORRELATED MONTE CARLO ({cm.n_groups} groups)")
            lines.append("-" * 50)
            lines.append(f"  Std ratio: {cm.std_ratio:.3f}x ({cm.variance_impact})")
            lines.append(f"  Independent: mean={cm.independent_mean:.6f}, std={cm.independent_std:.6f}")
            lines.append(f"  Correlated:  mean={cm.correlated_mean:.6f}, std={cm.correlated_std:.6f}")

        if self.growth_snapshots:
            lines.append(f"\nGROWTH TRACKING ({len(self.growth_snapshots)} snapshots)")
            lines.append("-" * 50)
            for s in self.growth_snapshots:
                lines.append(f"  {s.get('version_label','?'):<10} {s.get('system_fit',0):.2f} FIT  R={s.get('system_reliability',0):.6f}")

        lines.append("\n" + "=" * 70)
        lines.append("  Eliot Abramo | KiCad Reliability Plugin v3.1.0 | IEC TR 62380")
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

    def _on_export_pdf(self, event):
        dlg = wx.FileDialog(self, "Export PDF", wildcard="PDF (*.pdf)|*.pdf",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            try:
                html = self._generate_html()
                from .report_generator import ReportGenerator
                ReportGenerator.html_to_pdf(html, dlg.GetPath())
                self.status.SetLabel(f"PDF exported: {dlg.GetPath()}")
            except Exception as e:
                import traceback; traceback.print_exc()
                wx.MessageBox(f"PDF export failed: {e}\n\nFalling back: saving HTML for manual conversion.",
                              "PDF Error", wx.OK | wx.ICON_WARNING)
                # Fallback: save HTML
                fallback = dlg.GetPath().replace('.pdf', '.html')
                with open(fallback, "w", encoding="utf-8") as f:
                    f.write(html)
                self.status.SetLabel(f"HTML fallback exported: {fallback}")
        dlg.Destroy()

    def _build_mission_profile_dict(self) -> Optional[Dict]:
        """Build mission profile dict for report from parent dialog's settings."""
        try:
            parent = self.GetParent()
            if hasattr(parent, 'settings_panel') and hasattr(parent.settings_panel, 'get_mission_profile'):
                mp = parent.settings_panel.get_mission_profile()
                if mp and not mp.is_single_phase:
                    d = mp.to_dict()
                    # Try to add phasing impact
                    try:
                        from .mission_profile import estimate_phasing_impact
                        filtered = self._get_filtered_active_data()
                        for data in filtered.values():
                            for comp in data.get("components", [])[:1]:
                                impact = estimate_phasing_impact(
                                    comp.get("class", "Resistor"),
                                    comp.get("params", {}), mp.phases)
                                d["phased_lambda_fit"] = impact.get("phased_lambda", 0) * 1e9
                                d["single_phase_lambda_fit"] = impact.get("single_lambda", 0) * 1e9
                                d["delta_percent"] = impact.get("delta_percent", 0)
                                break
                            break
                    except Exception:
                        pass
                    return d
        except Exception:
            pass
        return None

    def _build_budget_dict(self) -> Optional[Dict]:
        """Build budget allocation dict for report."""
        if not self.budget_result:
            return None
        br = self.budget_result
        return {
            "strategy": br.strategy,
            "target_fit": br.target_fit,
            "actual_fit": br.actual_fit,
            "margin_fit": br.system_margin_fit,
            "target_reliability": br.target_reliability,
            "design_margin_pct": br.system_margin_percent,
            "components_over_budget": br.components_over_budget,
            "total_components": br.total_components,
            "sheet_budgets": [
                {
                    "sheet_path": sb.sheet_path,
                    "component_budgets": [
                        {
                            "ref": cb.reference,
                            "component_type": cb.component_type,
                            "actual_fit": cb.actual_fit,
                            "budget_fit": cb.budget_fit,
                            "margin_fit": cb.margin_fit,
                            "utilization_pct": cb.utilization * 100.0,
                            "passed": cb.within_budget,
                        } for cb in sb.component_budgets
                    ]
                } for sb in br.sheet_budgets
            ],
            "recommendations": br.recommendations,
        }

    def _build_derating_dict(self) -> Optional[Dict]:
        """Build derating guidance dict for report."""
        if not self.derating_result:
            return None
        dr = self.derating_result
        n_feasible = sum(1 for r in dr.recommendations if r.feasibility in ("easy", "moderate"))
        return {
            "system_actual_fit": dr.system_actual_fit,
            "system_target_fit": dr.system_target_fit,
            "system_gap_fit": dr.system_gap_fit,
            "n_feasible": n_feasible,
            "recommendations": [
                {
                    "reference": r.reference,
                    "parameter": r.parameter,
                    "current_value": str(r.current_value),
                    "required_value": str(r.required_value),
                    "change_pct": r.change_percent,
                    "fit_saved": r.system_fit_reduction,
                    "feasibility": r.feasibility,
                    "actions": r.actions,
                } for r in dr.recommendations
            ]
        }

    def _build_swap_dict(self) -> Optional[Dict]:
        """Build swap analysis dict for report."""
        if not self.swap_results:
            return None
        improvements = []
        for r in self.swap_results:
            improvements.append({
                "ref": r.get("reference", "?"),
                "component_type": r.get("component_type", ""),
                "param_name": r.get("swap_type", ""),
                "new_value": r.get("description", ""),
                "fit_before": r.get("fit_before", 0),
                "fit_after": r.get("fit_after", 0),
                "delta_percent": r.get("delta_percent", 0),
                "delta_system_fit": r.get("delta_fit", 0),
            })
        best = improvements[0] if improvements else None
        return {
            "improvements": improvements,
            "total_analyzed": len(self.swap_results),
            "best_single_improvement": {
                "ref": best["ref"],
                "delta_system_fit": best["delta_system_fit"],
                "swap_desc": best["new_value"][:30],
            } if best else None,
        }

    def _build_growth_dict(self) -> Optional[Dict]:
        """Build growth timeline dict for report."""
        if not self.growth_snapshots:
            return None
        target_fit = self.budget_result.target_fit if self.budget_result else None
        return {
            "snapshots": self.growth_snapshots,
            "target_fit": target_fit,
            "comparisons": self.growth_comparisons,
        }

    def _build_correlated_mc_dict(self) -> Optional[Dict]:
        """Build correlated MC dict for report."""
        if not self.correlated_mc_result:
            return None
        r = self.correlated_mc_result
        return {
            "n_groups": r.n_groups,
            "n_simulations": r.n_simulations,
            "independent_mean": r.independent_mean,
            "independent_std": r.independent_std,
            "correlated_mean": r.correlated_mean,
            "correlated_std": r.correlated_std,
            "std_ratio": r.std_ratio,
            "variance_impact": r.variance_impact,
            "independent_ci_lower": r.independent_ci[0],
            "independent_ci_upper": r.independent_ci[1],
            "correlated_ci_lower": r.correlated_ci[0],
            "correlated_ci_upper": r.correlated_ci[1],
            "groups": [
                {"name": g.name, "component_refs": g.component_refs,
                 "rho": g.correlation, "description": g.description}
                for g in (r.groups if hasattr(r, 'groups') else [])
            ]
        }

    def _generate_html(self) -> str:
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        filtered = self._get_filtered_active_data()

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
            sheets=filtered,
            blocks=[],
            monte_carlo=mc_dict,
            sensitivity=sens_dict,
            sheet_mc=sheet_mc_dict,
            criticality=self.criticality_results if self.criticality_results else None,
            tornado=tornado_dict,
            design_margin=dm_dict,
            # v3.1.0 co-design data
            mission_profile=self._build_mission_profile_dict(),
            budget=self._build_budget_dict(),
            derating=self._build_derating_dict(),
            swap_analysis=self._build_swap_dict(),
            growth_timeline=self._build_growth_dict(),
            correlated_mc=self._build_correlated_mc_dict(),
        )

        generator = ReportGenerator(logo_path=self.logo_path, logo_mime=self.logo_mime)
        return generator.generate_html(report_data)
