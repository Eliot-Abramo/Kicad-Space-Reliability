"""
Analysis Dialog - Reliability Analysis Suite v5
=============================================================
Four-tab streamlined interface for reliability-driven PCB co-design.

  Tab 1 - Overview:      System summary, contributions, type filter
  Tab 2 - Analysis:      Guided MC uncertainty + Tornado + Criticality
  Tab 3 - Design Actions: Budget + Improvements + What-If scenarios + History
  Tab 4 - Report:        PDF/HTML generation

All heavy computations run in background threads to keep KiCad responsive.
Errors are collected and reported -- never silently swallowed.

Author:  Eliot Abramo
"""

import wx
import wx.lib.scrolledpanel as scrolled
import math
import threading
import traceback
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .monte_carlo import (
    UncertaintyResult, ParameterSpec, ComponentInput,
    run_uncertainty_analysis, build_component_inputs,
    build_default_param_specs,
)
from .sensitivity_analysis import (
    TornadoResult, ScenarioResult, CriticalityEntry,
    TornadoPerturbation, DEFAULT_PERTURBATIONS,
    tornado_analysis, scenario_analysis, component_criticality,
    single_param_whatif, get_active_sheet_paths,
)
from .reliability_math import (
    reliability_from_lambda, lambda_from_reliability,
    calculate_component_lambda,
)
from .report_generator import ReportGenerator, ReportData
from .ecss_fields import (
    get_category_fields, infer_category_from_class, get_display_group,
    get_ordered_categories_present, math_type_to_ecss,
)
from .budget_allocation import allocate_budget, BudgetAllocationResult
from .derating_engine import compute_derating_guidance, DeratingResult
from .component_swap import rank_all_swaps
from .growth_tracking import (
    create_snapshot, save_snapshot, load_snapshots,
    compare_revisions, build_growth_timeline,
    ReliabilitySnapshot,
)


# =====================================================================
# Colour Scheme
# =====================================================================
class C:
    """Compact colour palette."""
    BG       = wx.Colour(248, 249, 250)
    WHITE    = wx.Colour(255, 255, 255)
    HEADER   = wx.Colour(30, 64, 120)
    TXT      = wx.Colour(33, 37, 41)
    TXT_M    = wx.Colour(108, 117, 125)
    TXT_L    = wx.Colour(173, 181, 189)
    PRI      = wx.Colour(37, 99, 235)
    OK       = wx.Colour(25, 135, 84)
    WARN     = wx.Colour(255, 193, 7)
    FAIL     = wx.Colour(220, 53, 69)
    BORDER   = wx.Colour(222, 226, 230)
    GRID     = wx.Colour(233, 236, 239)
    ROW_ALT  = wx.Colour(248, 249, 250)
    BAR = [wx.Colour(59,130,246), wx.Colour(16,185,129), wx.Colour(245,158,11),
           wx.Colour(220,53,69), wx.Colour(139,92,246), wx.Colour(236,72,153),
           wx.Colour(20,184,166), wx.Colour(249,115,22)]


# =====================================================================
# Chart Panels
# =====================================================================

def _trunc(s, maxlen=22):
    return (s[:maxlen-1] + "\u2026") if len(s) > maxlen else s


def _adaptive_font(dc, base_size, w, h, min_size=8):
    scale = min(w / 600.0, h / 400.0)
    sz = max(min_size, int(base_size * max(0.7, min(1.3, scale))))
    return sz


class HistogramPanel(wx.Panel):
    """Reliability distribution histogram with CI shading."""

    def __init__(self, parent, title="Distribution"):
        super().__init__(parent, size=(-1, 300))
        self.SetMinSize((400, 250))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.samples = None
        self.mean = self.p5 = self.p95 = None
        self.ci_lo = self.ci_hi = None
        self.ci_label = "90% CI"
        self.nominal = None
        self.jensen_note = ""
        self.Bind(wx.EVT_PAINT, self._paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, samples, mean, p5, p95, ci_lo=None, ci_hi=None,
                 ci_label="90% CI", nominal=None, jensen_note=""):
        self.samples = samples
        self.mean, self.p5, self.p95 = mean, p5, p95
        self.ci_lo, self.ci_hi = ci_lo, ci_hi
        self.ci_label = ci_label
        self.nominal = nominal
        self.jensen_note = jensen_note
        self.Refresh()

    def _paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()
        dc.SetBackground(wx.Brush(C.WHITE))
        dc.Clear()
        dc.SetPen(wx.Pen(C.BORDER, 1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(0, 0, w, h)

        fsz = _adaptive_font(dc, 11, w, h)
        dc.SetFont(wx.Font(fsz, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(C.TXT)
        dc.DrawText(self.title, 14, 10)

        if self.samples is None or len(self.samples) < 10:
            dc.SetFont(wx.Font(fsz - 1, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            dc.SetTextForeground(C.TXT_L)
            dc.DrawText("Run analysis to see distribution", w // 2 - 110, h // 2 - 10)
            return

        ml = max(55, int(w * 0.10))
        mr = max(20, int(w * 0.04))
        mt = max(40, int(h * 0.14))
        mb = max(50, int(h * 0.17))
        cw = w - ml - mr
        ch = h - mt - mb
        if cw < 50 or ch < 30:
            return

        n_bins = min(40, max(15, cw // 12))
        hist, edges = np.histogram(self.samples, bins=n_bins)
        max_count = max(hist) if max(hist) > 0 else 1
        bar_w = max(1, cw // n_bins - 1)
        min_val, max_val = edges[0], edges[-1]
        val_range = max_val - min_val
        if val_range <= 0:
            val_range = 1e-6

        def v2x(v):
            return ml + (v - min_val) / val_range * cw

        dc.SetPen(wx.Pen(C.GRID, 1))
        for i in range(5):
            y = mt + ch * i // 4
            dc.DrawLine(ml, y, w - mr, y)

        if self.ci_lo is not None and self.ci_hi is not None:
            x1, x2 = int(v2x(self.ci_lo)), int(v2x(self.ci_hi))
            dc.SetBrush(wx.Brush(wx.Colour(25, 135, 84, 25)))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(x1, mt, max(1, x2 - x1), ch)

        dc.SetBrush(wx.Brush(C.PRI))
        dc.SetPen(wx.Pen(C.PRI.ChangeLightness(80), 1))
        for i, count in enumerate(hist):
            if count > 0:
                x = ml + i * cw // n_bins
                bh = max(1, int((count / max_count) * ch))
                dc.DrawRectangle(int(x), mt + ch - bh, int(bar_w), bh)

        line_specs = []
        if self.nominal is not None:
            line_specs.append((self.nominal, wx.Colour(0, 150, 0), 2, wx.PENSTYLE_SOLID, "Nominal"))
        if self.mean is not None:
            line_specs.append((self.mean, C.FAIL, 2, wx.PENSTYLE_SOLID, "Mean"))
        for pv, label in [(self.p5, "P5"), (self.p95, "P95")]:
            if pv is not None:
                line_specs.append((pv, C.WARN, 2, wx.PENSTYLE_SHORT_DASH, label))
        if self.ci_lo is not None:
            line_specs.append((self.ci_lo, C.OK, 2, wx.PENSTYLE_SHORT_DASH, "CI Low"))
            line_specs.append((self.ci_hi, C.OK, 2, wx.PENSTYLE_SHORT_DASH, "CI High"))

        for val, col, width, style, _ in line_specs:
            if min_val <= val <= max_val:
                dc.SetPen(wx.Pen(col, width, style))
                xv = int(v2x(val))
                dc.DrawLine(xv, mt, xv, mt + ch)

        dc.SetPen(wx.Pen(C.TXT_M, 1))
        dc.DrawLine(ml, mt + ch, w - mr, mt + ch)
        dc.DrawLine(ml, mt, ml, mt + ch)

        fsm = max(8, fsz - 2)
        dc.SetFont(wx.Font(fsm, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        dc.SetTextForeground(C.TXT_M)
        n_labels = min(5, max(3, cw // 80))
        for i in range(n_labels):
            val = min_val + val_range * i / (n_labels - 1)
            x = int(v2x(val))
            lbl = f"{val:.4f}" if val_range < 0.1 else f"{val:.3f}"
            tw, _ = dc.GetTextExtent(lbl)
            dc.DrawText(lbl, x - tw // 2, mt + ch + 6)

        dc.DrawText("R(t)", ml + cw // 2 - 12, h - mb + 30)

        lx = w - mr - 130
        ly = mt + 5
        dc.SetFont(wx.Font(max(7, fsm - 1), wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        row_h = 14
        for val, col, width, style, label in line_specs:
            dc.SetPen(wx.Pen(col, width, style))
            dc.DrawLine(lx, ly + 6, lx + 18, ly + 6)
            dc.SetTextForeground(C.TXT_M)
            dc.DrawText(label, lx + 22, ly)
            ly += row_h


class HBarPanel(wx.Panel):
    """Horizontal bar chart with proper label handling."""

    def __init__(self, parent, title="Chart"):
        super().__init__(parent, size=(-1, 350))
        self.SetMinSize((350, 200))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.data = []
        self.max_value = 1.0
        self.x_label = "FIT"
        self.Bind(wx.EVT_PAINT, self._paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, data, max_value=None, x_label="FIT"):
        self.data = [(n, v, i % len(C.BAR)) for i, (n, v) in enumerate(data)]
        self.max_value = max_value or (max(d[1] for d in self.data) if self.data else 1.0)
        self.max_value = max(self.max_value, 1e-6)
        self.x_label = x_label
        self.Refresh()

    def _paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()
        dc.SetBackground(wx.Brush(C.WHITE))
        dc.Clear()
        dc.SetPen(wx.Pen(C.BORDER, 1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(0, 0, w, h)

        fsz = _adaptive_font(dc, 11, w, h)
        dc.SetFont(wx.Font(fsz, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(C.TXT)
        dc.DrawText(self.title, 14, 10)

        if not self.data:
            return

        fsm = max(8, fsz - 1)
        dc.SetFont(wx.Font(fsm, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        max_label_w = 0
        for name, _, _ in self.data[:20]:
            tw, _ = dc.GetTextExtent(_trunc(name, 24))
            max_label_w = max(max_label_w, tw)

        ml = max(80, min(int(w * 0.35), max_label_w + 20))
        mr = max(20, int(w * 0.04))
        mt = max(38, int(h * 0.12))
        mb = max(30, int(h * 0.10))
        cw = w - ml - mr
        ch = h - mt - mb
        if cw < 30 or ch < 20:
            return

        n = min(len(self.data), max(5, ch // 20))
        bh = max(12, min(24, (ch - 8) // n))
        sp = max(2, (ch - n * bh) // (n + 1))

        dc.SetPen(wx.Pen(C.GRID, 1))
        for i in range(5):
            x = ml + cw * i // 4
            dc.DrawLine(x, mt, x, mt + ch)

        for i, (name, value, ci) in enumerate(self.data[:n]):
            y = mt + sp + i * (bh + sp)
            bw = max(2, int((value / self.max_value) * cw))
            color = C.BAR[ci]

            if i % 2 == 1:
                dc.SetBrush(wx.Brush(C.ROW_ALT))
                dc.SetPen(wx.TRANSPARENT_PEN)
                dc.DrawRectangle(0, y - sp // 2, w, bh + sp)

            dc.SetBrush(wx.Brush(color))
            dc.SetPen(wx.Pen(color.ChangeLightness(80), 1))
            dc.DrawRoundedRectangle(ml, y, bw, bh, 2)

            dn = _trunc(name, 24)
            dc.SetTextForeground(C.TXT)
            dc.SetFont(wx.Font(fsm, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            tw, th = dc.GetTextExtent(dn)
            dc.DrawText(dn, ml - tw - 8, y + (bh - th) // 2)

            vt = f"{value:.2f}" if value < 100 else f"{value:.1f}" if value < 10000 else f"{value:.0f}"
            vw, vh = dc.GetTextExtent(vt)
            if bw > vw + 14:
                dc.SetTextForeground(C.WHITE)
                dc.DrawText(vt, ml + 6, y + (bh - vh) // 2)
            else:
                dc.SetTextForeground(C.TXT)
                dc.DrawText(vt, ml + bw + 6, y + (bh - vh) // 2)


class ConvergencePanel(wx.Panel):
    """Running mean convergence plot."""

    def __init__(self, parent, title="Mean Convergence"):
        super().__init__(parent, size=(-1, 180))
        self.SetMinSize((300, 140))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.history = []
        self.Bind(wx.EVT_PAINT, self._paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, history):
        self.history = history or []
        self.Refresh()

    def _paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()
        dc.SetBackground(wx.Brush(C.WHITE))
        dc.Clear()
        dc.SetPen(wx.Pen(C.BORDER, 1))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(0, 0, w, h)

        fsz = _adaptive_font(dc, 10, w, h, min_size=8)
        dc.SetFont(wx.Font(fsz, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(C.TXT)
        dc.DrawText(self.title, 12, 8)

        if len(self.history) < 3:
            return

        ml = max(55, int(w * 0.12))
        mr = max(15, int(w * 0.04))
        mt = max(30, int(h * 0.20))
        mb = max(25, int(h * 0.16))
        cw = w - ml - mr
        ch = h - mt - mb
        if cw < 30 or ch < 20:
            return

        vals = [v for _, v in self.history]
        n_pts = [n for n, _ in self.history]
        v_min, v_max = min(vals), max(vals)
        vr = v_max - v_min
        if vr < 1e-12:
            vr = abs(v_max) * 0.05 or 0.01
        v_min -= vr * 0.15
        v_max += vr * 0.15
        vr = v_max - v_min
        n_max = max(n_pts)

        dc.SetPen(wx.Pen(C.GRID, 1))
        for i in range(5):
            y = mt + ch * i // 4
            dc.DrawLine(ml, y, w - mr, y)

        dc.SetPen(wx.Pen(C.PRI, 2))
        prev = None
        for nn, v in self.history:
            x = ml + (nn / n_max) * cw
            y = mt + ch - ((v - v_min) / vr) * ch
            if prev:
                dc.DrawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)

        dc.SetFont(wx.Font(max(7, fsz - 2), wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        dc.SetTextForeground(C.TXT_M)
        for i in range(3):
            v = v_min + vr * i / 2
            y = mt + ch - (i / 2) * ch
            lbl = f"{v:.5f}"
            tw, _ = dc.GetTextExtent(lbl)
            dc.DrawText(lbl, ml - tw - 4, int(y) - 6)


# =====================================================================
# Utility helpers
# =====================================================================

def _make_list(parent, columns, col_widths=None):
    lc = wx.ListCtrl(parent, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
    lc.SetBackgroundColour(C.WHITE)
    lc.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
    for i, name in enumerate(columns):
        w = col_widths[i] if col_widths and i < len(col_widths) else 100
        lc.InsertColumn(i, name, width=w)
    return lc


def _add_row(lc, values, color=None):
    idx = lc.InsertItem(lc.GetItemCount(), str(values[0]))
    for i, v in enumerate(values[1:], 1):
        lc.SetItem(idx, i, str(v))
    if color:
        lc.SetItemTextColour(idx, color)
    return idx


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# =====================================================================
# Background thread infrastructure
# =====================================================================

class _AnalysisRunner:
    """Run a callable in a background thread, report results via wx.CallAfter.

    Usage::

        runner = _AnalysisRunner(
            work_fn=lambda cancel: slow_computation(),
            on_done=lambda result: update_ui(result),
            on_error=lambda exc, tb_str: show_error(exc, tb_str),
            on_progress=lambda msg: status_bar.SetLabel(msg),
        )
        runner.start()
        # runner.cancel()   -- sets cancel flag; work_fn should check it
    """

    def __init__(self, work_fn, on_done, on_error, on_progress=None):
        self._work_fn = work_fn
        self._on_done = on_done
        self._on_error = on_error
        self._on_progress = on_progress
        self._cancel_event = threading.Event()
        self._thread = None

    @property
    def cancelled(self):
        return self._cancel_event.is_set()

    def cancel(self):
        self._cancel_event.set()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            result = self._work_fn(self._cancel_event)
            if not self._cancel_event.is_set():
                wx.CallAfter(self._on_done, result)
        except Exception as exc:
            tb_str = traceback.format_exc()
            if not self._cancel_event.is_set():
                wx.CallAfter(self._on_error, exc, tb_str)

    def progress(self, msg):
        if self._on_progress and not self._cancel_event.is_set():
            wx.CallAfter(self._on_progress, msg)


# =====================================================================
# Main Analysis Dialog
# =====================================================================

class AnalysisDialog(wx.Dialog):
    """Streamlined 4-tab reliability analysis dialog.

    Tab 1 - Overview:       System summary, contributions, type filter
    Tab 2 - Analysis:       MC uncertainty + Tornado + Criticality
    Tab 3 - Design Actions: Budget + Improvements + What-If + History
    Tab 4 - Report:         HTML/PDF generation
    """

    def __init__(self, parent, system_lambda, mission_hours, sheet_data=None,
                 block_structure=None, blocks=None, root_id=None, project_path=None,
                 logo_path=None, logo_mime=None, n_cycles=5256, delta_t=3.0,
                 title="Reliability Analysis Suite"):
        display = wx.Display(0)
        rect = display.GetClientArea()
        w = min(1400, int(rect.Width * 0.88))
        h = min(950, int(rect.Height * 0.90))
        super().__init__(parent, title=title, size=(w, h),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        self.SetMinSize((1000, 700))
        self.SetBackgroundColour(C.BG)

        self.system_lambda = system_lambda
        self.mission_hours = mission_hours
        self.sheet_data = sheet_data or {}
        self.block_structure = block_structure or {}
        self.blocks = blocks
        self.root_id = root_id
        self.project_path = project_path
        self.logo_path = logo_path
        self.logo_mime = logo_mime or "image/png"
        self.n_cycles = n_cycles
        self.delta_t = delta_t

        self.active_sheets = get_active_sheet_paths(self.blocks)
        if self.active_sheets:
            self._active_data = {k: v for k, v in self.sheet_data.items()
                                 if k in self.active_sheets}
        else:
            self._active_data = self.sheet_data

        # State
        self.excluded_types = set()
        self._type_cbs = {}
        self.mc_result: Optional[UncertaintyResult] = None
        self.tornado_result: Optional[TornadoResult] = None
        self.scenario_result: Optional[ScenarioResult] = None
        self.criticality_results: List[CriticalityEntry] = []
        self.budget_result: Optional[BudgetAllocationResult] = None
        self.derating_result: Optional[DeratingResult] = None
        self.swap_results: List[Dict] = []
        self.param_specs: List[ParameterSpec] = []

        # Active background runner (at most one at a time)
        self._runner: Optional[_AnalysisRunner] = None
        # Collected warnings from last analysis run
        self._warnings: List[str] = []

        self._build_ui()
        self.Centre()

    # =================================================================
    # Data helpers
    # =================================================================

    def _filtered(self):
        result = {}
        for path, data in self._active_data.items():
            comps = [c for c in data.get("components", [])
                     if c.get("class", "Unknown") not in self.excluded_types]
            new_lam = sum(_safe_float(c.get("lambda", 0)) for c in comps)
            result[path] = {**data, "components": comps, "lambda": new_lam}
        return result

    def _all_components(self):
        comps = []
        for data in self._filtered().values():
            comps.extend(data.get("components", []))
        return comps

    def _sys_fit(self):
        return self.system_lambda * 1e9

    def _sys_r(self):
        return reliability_from_lambda(self.system_lambda, self.mission_hours)

    def _component_types(self):
        types = set()
        for data in self._active_data.values():
            for c in data.get("components", []):
                types.add(c.get("class", "Unknown"))
        return types

    # =================================================================
    # Threading helpers
    # =================================================================

    def _is_busy(self):
        return self._runner is not None and self._runner._thread is not None and self._runner._thread.is_alive()

    def _start_analysis(self, work_fn, on_done, label="Running analysis...",
                        buttons_to_disable=None):
        """Start a background analysis. Blocks re-entry; disables given buttons."""
        if self._is_busy():
            wx.MessageBox(
                "An analysis is already running. Please wait or cancel it first.",
                "Busy", wx.OK | wx.ICON_INFORMATION)
            return
        self._warnings = []
        self.status.SetLabel(label)
        self._disabled_buttons = buttons_to_disable or []
        for btn in self._disabled_buttons:
            btn.Disable()

        def _on_error(exc, tb_str):
            for btn in self._disabled_buttons:
                btn.Enable()
            self.status.SetLabel("Analysis failed.")
            wx.MessageBox(
                f"Analysis error:\n\n{exc}\n\nDetails:\n{tb_str[:800]}",
                "Analysis Error", wx.OK | wx.ICON_ERROR)

        def _on_done_wrapper(result):
            for btn in self._disabled_buttons:
                btn.Enable()
            on_done(result)
            if self._warnings:
                n = len(self._warnings)
                summary = "\n".join(self._warnings[:20])
                if n > 20:
                    summary += f"\n... and {n - 20} more warnings"
                wx.MessageBox(
                    f"{n} warning(s) during analysis:\n\n{summary}",
                    "Analysis Warnings", wx.OK | wx.ICON_WARNING)

        def _on_progress(msg):
            self.status.SetLabel(msg)

        self._runner = _AnalysisRunner(
            work_fn=work_fn,
            on_done=_on_done_wrapper,
            on_error=_on_error,
            on_progress=_on_progress,
        )
        self._runner.start()

    def _cancel_analysis(self):
        if self._runner:
            self._runner.cancel()
            for btn in getattr(self, '_disabled_buttons', []):
                btn.Enable()
            self.status.SetLabel("Cancelled.")

    # =================================================================
    # UI Construction
    # =================================================================

    def _build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        hdr = wx.Panel(self)
        hdr.SetBackgroundColour(C.HEADER)
        hs = wx.BoxSizer(wx.HORIZONTAL)
        t = wx.StaticText(hdr, label="Reliability Analysis Suite")
        t.SetFont(wx.Font(13, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        t.SetForegroundColour(wx.WHITE)
        hs.Add(t, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)
        r = self._sys_r()
        yrs = self.mission_hours / 8760
        info = (f"\u03BB = {self._sys_fit():.2f} FIT  |  "
                f"R(t) = {r:.6f}  |  {yrs:.1f}y  |  "
                f"{len(self._active_data)} sheets")
        il = wx.StaticText(hdr, label=info)
        il.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        il.SetForegroundColour(wx.Colour(200, 215, 240))
        hs.Add(il, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)

        self.btn_cancel = wx.Button(hdr, label="Cancel")
        self.btn_cancel.SetForegroundColour(wx.WHITE)
        self.btn_cancel.SetBackgroundColour(C.FAIL)
        self.btn_cancel.Bind(wx.EVT_BUTTON, lambda e: self._cancel_analysis())
        self.btn_cancel.SetToolTip("Cancel running analysis")
        hs.Add(self.btn_cancel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)

        hdr.SetSizer(hs)
        sizer.Add(hdr, 0, wx.EXPAND)

        # Notebook
        self.nb = wx.Notebook(self)
        self.nb.SetBackgroundColour(C.BG)
        self.nb.AddPage(self._tab_overview(), "Overview")
        self.nb.AddPage(self._tab_analysis(), "Analysis")
        self.nb.AddPage(self._tab_design_actions(), "Design Actions")
        self.nb.AddPage(self._tab_report(), "Report")
        sizer.Add(self.nb, 1, wx.EXPAND | wx.ALL, 8)

        # Status bar
        self.status = wx.StaticText(self, label="Ready")
        self.status.SetForegroundColour(C.TXT_M)
        sizer.Add(self.status, 0, wx.EXPAND | wx.LEFT | wx.BOTTOM, 12)

        self.SetSizer(sizer)

    # =================================================================
    # Tab 1: Overview (was Dashboard)
    # =================================================================

    def _tab_overview(self):
        panel = wx.Panel(self.nb)
        panel.SetBackgroundColour(C.BG)
        main = wx.BoxSizer(wx.VERTICAL)

        # Component type filter
        fp = wx.Panel(panel)
        fp.SetBackgroundColour(C.WHITE)
        fs = wx.BoxSizer(wx.VERTICAL)
        fl = wx.StaticText(fp, label="Component Types (uncheck to exclude from all analyses):")
        fl.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        fs.Add(fl, 0, wx.ALL, 8)
        tr = wx.WrapSizer(wx.HORIZONTAL)
        for tn in sorted(self._component_types()):
            cb = wx.CheckBox(fp, label=tn)
            cb.SetValue(True)
            cb.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            cb.Bind(wx.EVT_CHECKBOX, self._on_type_filter)
            tr.Add(cb, 0, wx.ALL, 4)
            self._type_cbs[tn] = cb
        fs.Add(tr, 0, wx.LEFT, 10)
        fp.SetSizer(fs)
        main.Add(fp, 0, wx.EXPAND | wx.ALL, 6)

        # Charts row
        charts = wx.BoxSizer(wx.HORIZONTAL)
        self.contrib_chart = HBarPanel(panel, "FIT Contributions (Top 15)")
        charts.Add(self.contrib_chart, 2, wx.EXPAND | wx.RIGHT, 6)

        card = wx.Panel(panel)
        card.SetBackgroundColour(C.WHITE)
        cs = wx.BoxSizer(wx.VERTICAL)
        cl = wx.StaticText(card, label="System Summary")
        cl.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        cs.Add(cl, 0, wx.ALL, 12)
        self.dash_summary = wx.StaticText(card, label="")
        self.dash_summary.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        cs.Add(self.dash_summary, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 12)
        card.SetSizer(cs)
        charts.Add(card, 1, wx.EXPAND)
        main.Add(charts, 1, wx.EXPAND | wx.ALL, 6)

        # Contribution table
        self.contrib_list = _make_list(panel,
            ["Component", "Type", "\u03BB (FIT)", "Contribution %", "Cumulative %"],
            [160, 130, 100, 100, 100])
        main.Add(self.contrib_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        panel.SetSizer(main)
        self._refresh_overview()
        return panel

    def _refresh_overview(self):
        filtered = self._filtered()
        total_lam = sum(_safe_float(d.get("lambda", 0)) for d in filtered.values())
        total_fit = total_lam * 1e9

        comps = []
        for data in filtered.values():
            for c in data.get("components", []):
                lam = _safe_float(c.get("lambda", 0))
                comps.append((c.get("ref", "?"), c.get("class", "?"), lam * 1e9))
        comps.sort(key=lambda x: -x[2])

        chart_data = [(ref, fit) for ref, _, fit in comps[:15]]
        self.contrib_chart.set_data(chart_data, x_label="FIT")

        self.contrib_list.DeleteAllItems()
        cum = 0.0
        for ref, ctype, fit in comps:
            pct = (fit / total_fit * 100) if total_fit > 0 else 0
            cum += pct
            _add_row(self.contrib_list, [
                ref, _trunc(ctype, 18), f"{fit:.2f}",
                f"{pct:.1f}%", f"{cum:.1f}%",
            ])

        r = reliability_from_lambda(total_lam, self.mission_hours)
        mttf = 1.0 / total_lam if total_lam > 0 else float('inf')
        mttf_yr = mttf / 8760
        lines = [
            f"System \u03BB:    {total_fit:.2f} FIT",
            f"R(t):         {r:.6f}",
            f"MTTF:         {mttf_yr:.1f} years",
            f"Mission:      {self.mission_hours/8760:.1f} years",
            f"Components:   {len(comps)}",
            f"Sheets:       {len(filtered)}",
        ]
        self.dash_summary.SetLabel("\n".join(lines))

    # =================================================================
    # Tab 2: Analysis (MC + Tornado + Criticality -- guided workflow)
    # =================================================================

    def _tab_analysis(self):
        panel = scrolled.ScrolledPanel(self.nb)
        panel.SetBackgroundColour(C.BG)
        panel.SetupScrolling(scroll_x=False, scrollToTop=True)
        main = wx.BoxSizer(wx.VERTICAL)

        # Workflow hint
        hint_panel = wx.Panel(panel)
        hint_panel.SetBackgroundColour(wx.Colour(230, 240, 255))
        hs = wx.BoxSizer(wx.VERTICAL)
        hint = wx.StaticText(hint_panel,
            label="Workflow: 1) Run Uncertainty Analysis to quantify confidence bounds  "
                  "2) Run Tornado to find highest-leverage parameters  "
                  "3) Run Criticality to identify which component specs to tighten")
        hint.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
        hint.SetForegroundColour(C.HEADER)
        hint.Wrap(1300)
        hs.Add(hint, 0, wx.ALL, 10)
        hint_panel.SetSizer(hs)
        main.Add(hint_panel, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 1: Uncertainty (Monte Carlo) ---
        main.Add(self._section_label(panel, "Step 1: Uncertainty Analysis (Monte Carlo)"), 0, wx.EXPAND | wx.ALL, 6)

        # Quick setup bar
        qp = wx.Panel(panel)
        qp.SetBackgroundColour(C.WHITE)
        qs = wx.BoxSizer(wx.HORIZONTAL)
        qs.Add(wx.StaticText(qp, label="Global \u00b1%:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.unc_pct = wx.SpinCtrlDouble(qp, min=1, max=50, initial=10, inc=1, size=(70, -1))
        qs.Add(self.unc_pct, 0, wx.ALL, 6)

        qs.Add(wx.StaticText(qp, label="Dist:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.unc_dist = wx.Choice(qp, choices=["PERT", "Uniform"])
        self.unc_dist.SetSelection(0)
        qs.Add(self.unc_dist, 0, wx.ALL, 6)

        btn_apply = wx.Button(qp, label="Apply to All")
        btn_apply.Bind(wx.EVT_BUTTON, self._on_apply_quick_unc)
        qs.Add(btn_apply, 0, wx.ALL, 6)

        qs.AddStretchSpacer()

        qs.Add(wx.StaticText(qp, label="Simulations:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.unc_n = wx.SpinCtrl(qp, min=500, max=50000, initial=3000, size=(85, -1))
        qs.Add(self.unc_n, 0, wx.ALL, 6)

        qs.Add(wx.StaticText(qp, label="CI:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.unc_ci = wx.Choice(qp, choices=["80%", "90%", "95%", "99%"])
        self.unc_ci.SetSelection(1)
        qs.Add(self.unc_ci, 0, wx.ALL, 6)

        self.btn_run_mc = wx.Button(qp, label="\u25B6  Run Uncertainty")
        self.btn_run_mc.SetBackgroundColour(C.PRI)
        self.btn_run_mc.SetForegroundColour(wx.WHITE)
        self.btn_run_mc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_run_mc.Bind(wx.EVT_BUTTON, self._on_run_uncertainty)
        qs.Add(self.btn_run_mc, 0, wx.ALL, 6)

        qp.SetSizer(qs)
        main.Add(qp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        # Parameter table
        param_panel = wx.Panel(panel)
        param_panel.SetBackgroundColour(C.WHITE)
        ps = wx.BoxSizer(wx.VERTICAL)
        pl = wx.StaticText(param_panel,
            label="Parameter Uncertainty Bounds (double-click to edit):")
        pl.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        ps.Add(pl, 0, wx.ALL, 8)

        self.param_list = _make_list(param_panel,
            ["Parameter", "Components", "Mode", "Low Bound", "High Bound", "Distribution", "Shared"],
            [140, 80, 80, 95, 95, 90, 65])
        ps.Add(self.param_list, 1, wx.EXPAND | wx.ALL, 4)

        param_panel.SetSizer(ps)
        main.Add(param_panel, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        # MC results area
        res_panel = wx.Panel(panel)
        res_panel.SetBackgroundColour(C.BG)
        rs = wx.BoxSizer(wx.HORIZONTAL)

        left = wx.BoxSizer(wx.VERTICAL)
        self.mc_histogram = HistogramPanel(res_panel, "R(t) Distribution")
        left.Add(self.mc_histogram, 3, wx.EXPAND)
        self.mc_convergence = ConvergencePanel(res_panel, "Mean R(t) Convergence")
        left.Add(self.mc_convergence, 1, wx.EXPAND | wx.TOP, 6)
        rs.Add(left, 3, wx.EXPAND | wx.RIGHT, 6)

        right = wx.BoxSizer(wx.VERTICAL)
        sp = wx.Panel(res_panel)
        sp.SetBackgroundColour(C.WHITE)
        ss = wx.BoxSizer(wx.VERTICAL)
        sl = wx.StaticText(sp, label="Statistics")
        sl.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        ss.Add(sl, 0, wx.ALL, 8)
        self.mc_stats_text = wx.StaticText(sp, label="Run analysis to see results.")
        self.mc_stats_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        ss.Add(self.mc_stats_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        sp.SetSizer(ss)
        right.Add(sp, 1, wx.EXPAND)

        self.mc_importance = HBarPanel(res_panel, "Parameter Importance (SRRC\u00B2)")
        right.Add(self.mc_importance, 2, wx.EXPAND | wx.TOP, 6)

        rs.Add(right, 2, wx.EXPAND)
        res_panel.SetSizer(rs)
        main.Add(res_panel, 2, wx.EXPAND | wx.ALL, 6)

        self.jensen_label = wx.StaticText(panel, label="")
        self.jensen_label.SetForegroundColour(C.TXT_M)
        self.jensen_label.SetFont(wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL))
        main.Add(self.jensen_label, 0, wx.LEFT | wx.BOTTOM, 8)

        # --- Section 2: Tornado Sensitivity ---
        main.Add(self._section_label(panel, "Step 2: Tornado Sensitivity (OAT)"), 0, wx.EXPAND | wx.ALL, 6)

        tp = wx.Panel(panel)
        tp.SetBackgroundColour(C.WHITE)
        ts = wx.BoxSizer(wx.HORIZONTAL)
        ts.Add(wx.StaticText(tp, label="Mode:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.tornado_mode = wx.Choice(tp, choices=["Parameter-level", "Sheet-level"])
        self.tornado_mode.SetSelection(0)
        ts.Add(self.tornado_mode, 0, wx.ALL, 6)
        self.btn_tornado = wx.Button(tp, label="\u25B6  Run Tornado")
        self.btn_tornado.SetBackgroundColour(C.OK)
        self.btn_tornado.SetForegroundColour(wx.WHITE)
        self.btn_tornado.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_tornado.Bind(wx.EVT_BUTTON, self._on_run_tornado)
        ts.Add(self.btn_tornado, 0, wx.ALL, 6)
        ts.Add(wx.StaticText(tp,
            label="Perturbations use physical units. Double-click table to edit."),
            0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        tp.SetSizer(ts)
        main.Add(tp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.pert_list = _make_list(panel,
            ["Parameter", "Low (-)", "High (+)", "Unit", "Enabled"],
            [150, 85, 85, 85, 65])
        self.pert_list.SetMinSize((-1, 160))
        self._populate_pert_table()
        self.pert_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit_pert)
        main.Add(self.pert_list, 0, wx.EXPAND | wx.ALL, 6)

        self.tornado_chart = HBarPanel(panel, "Tornado: System FIT Swing")
        self.tornado_chart.SetMinSize((-1, 280))
        main.Add(self.tornado_chart, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.tornado_table = _make_list(panel,
            ["Parameter", "Low FIT", "Base FIT", "High FIT", "Swing", "Perturbation"],
            [170, 95, 95, 95, 95, 140])
        self.tornado_table.SetMinSize((-1, 160))
        main.Add(self.tornado_table, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 3: Component Criticality ---
        main.Add(self._section_label(panel, "Step 3: Component Criticality (Elasticity)"), 0, wx.EXPAND | wx.ALL, 6)

        cp = wx.Panel(panel)
        cp.SetBackgroundColour(C.WHITE)
        css = wx.BoxSizer(wx.HORIZONTAL)
        css.Add(wx.StaticText(cp, label="Top N (0 = all):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.crit_n = wx.SpinCtrl(cp, min=0, max=500, initial=0, size=(70, -1))
        css.Add(self.crit_n, 0, wx.ALL, 6)
        self.btn_crit = wx.Button(cp, label="\u25B6  Run Criticality")
        self.btn_crit.SetBackgroundColour(C.PRI)
        self.btn_crit.SetForegroundColour(wx.WHITE)
        self.btn_crit.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_crit.Bind(wx.EVT_BUTTON, self._on_run_criticality)
        css.Add(self.btn_crit, 0, wx.ALL, 6)
        cp.SetSizer(css)
        main.Add(cp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.crit_table = _make_list(panel,
            ["Reference", "Type", "\u03BB (FIT)", "Top Parameter", "Elasticity", "Impact %"],
            [100, 130, 90, 150, 90, 90])
        self.crit_table.SetMinSize((-1, 240))
        main.Add(self.crit_table, 0, wx.EXPAND | wx.ALL, 6)

        # Bind param table editing
        self.param_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit_param)
        self._populate_param_table()

        panel.SetSizer(main)
        return panel

    def _section_label(self, parent, text):
        p = wx.Panel(parent)
        p.SetBackgroundColour(wx.Colour(230, 240, 255))
        s = wx.BoxSizer(wx.VERTICAL)
        l = wx.StaticText(p, label=text)
        l.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        l.SetForegroundColour(C.HEADER)
        s.Add(l, 0, wx.ALL, 10)
        p.SetSizer(s)
        return p

    # -- Param table management --

    def _populate_param_table(self, pct=10.0, dist="pert"):
        comps = build_component_inputs(self._active_data, excluded_types=self.excluded_types)
        self.param_specs = build_default_param_specs(comps, pct, dist)
        self.param_list.DeleteAllItems()
        for spec in self.param_specs:
            mode = "Shared" if spec.shared else "Independent"
            if spec.shared:
                lo = f"{spec.delta_low:+.2f}"
                hi = f"{spec.delta_high:+.2f}"
            else:
                lo = f"-{spec.rel_low:.1f}%"
                hi = f"+{spec.rel_high:.1f}%"
            _add_row(self.param_list, [
                spec.name, str(spec.n_components), mode,
                lo, hi, spec.distribution.upper(),
                "Yes" if spec.shared else "No",
            ])

    def _on_apply_quick_unc(self, event):
        pct = self.unc_pct.GetValue()
        dist = "pert" if self.unc_dist.GetSelection() == 0 else "uniform"
        self._populate_param_table(pct, dist)

    def _on_edit_param(self, event):
        idx = event.GetIndex()
        if idx < 0 or idx >= len(self.param_specs):
            return
        spec = self.param_specs[idx]

        dlg = wx.Dialog(self, title=f"Edit: {spec.name}", size=(400, 300))
        ds = wx.BoxSizer(wx.VERTICAL)
        ds.Add(wx.StaticText(dlg, label=f"Parameter: {spec.name} ({spec.n_components} components)"),
               0, wx.ALL, 12)

        gs = wx.FlexGridSizer(5, 2, 8, 12)
        gs.AddGrowableCol(1)

        gs.Add(wx.StaticText(dlg, label="Mode:"), 0, wx.ALIGN_CENTER_VERTICAL)
        mode_ch = wx.Choice(dlg, choices=["Shared (additive delta)", "Independent (relative %)"])
        mode_ch.SetSelection(0 if spec.shared else 1)
        gs.Add(mode_ch, 1, wx.EXPAND)

        gs.Add(wx.StaticText(dlg, label="Low bound:"), 0, wx.ALIGN_CENTER_VERTICAL)
        lo_val = spec.delta_low if spec.shared else -spec.rel_low
        lo_ctrl = wx.TextCtrl(dlg, value=f"{lo_val:.2f}")
        gs.Add(lo_ctrl, 1, wx.EXPAND)

        gs.Add(wx.StaticText(dlg, label="High bound:"), 0, wx.ALIGN_CENTER_VERTICAL)
        hi_val = spec.delta_high if spec.shared else spec.rel_high
        hi_ctrl = wx.TextCtrl(dlg, value=f"{hi_val:.2f}")
        gs.Add(hi_ctrl, 1, wx.EXPAND)

        gs.Add(wx.StaticText(dlg, label="Distribution:"), 0, wx.ALIGN_CENTER_VERTICAL)
        dist_ch = wx.Choice(dlg, choices=["PERT", "Uniform"])
        dist_ch.SetSelection(0 if spec.distribution == "pert" else 1)
        gs.Add(dist_ch, 1, wx.EXPAND)

        ds.Add(gs, 0, wx.EXPAND | wx.ALL, 12)
        ds.Add(dlg.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 12)
        dlg.SetSizer(ds)

        if dlg.ShowModal() == wx.ID_OK:
            spec.shared = (mode_ch.GetSelection() == 0)
            spec.distribution = "pert" if dist_ch.GetSelection() == 0 else "uniform"
            try:
                lo = float(lo_ctrl.GetValue())
                hi = float(hi_ctrl.GetValue())
            except ValueError:
                dlg.Destroy()
                return
            if spec.shared:
                spec.delta_low = lo
                spec.delta_high = hi
            else:
                spec.rel_low = abs(lo)
                spec.rel_high = abs(hi)
            self.param_list.DeleteAllItems()
            for s in self.param_specs:
                mode = "Shared" if s.shared else "Independent"
                if s.shared:
                    lo_s = f"{s.delta_low:+.2f}"
                    hi_s = f"{s.delta_high:+.2f}"
                else:
                    lo_s = f"-{s.rel_low:.1f}%"
                    hi_s = f"+{s.rel_high:.1f}%"
                _add_row(self.param_list, [
                    s.name, str(s.n_components), mode,
                    lo_s, hi_s, s.distribution.upper(),
                    "Yes" if s.shared else "No",
                ])
        dlg.Destroy()

    # -- Perturbation table --

    def _populate_pert_table(self):
        self._perturbations = list(DEFAULT_PERTURBATIONS)
        self.pert_list.DeleteAllItems()
        for p in self._perturbations:
            _add_row(self.pert_list, [
                p.param_name, f"{p.delta_low}", f"{p.delta_high}",
                p.unit, "Yes" if p.enabled else "No",
            ])

    def _on_edit_pert(self, event):
        idx = event.GetIndex()
        if idx < 0 or idx >= len(self._perturbations):
            return
        p = self._perturbations[idx]
        dlg = wx.TextEntryDialog(self,
            f"Enter low,high for {p.param_name} (e.g. '10,15'):",
            "Edit Perturbation", f"{p.delta_low},{p.delta_high}")
        if dlg.ShowModal() == wx.ID_OK:
            try:
                parts = dlg.GetValue().split(",")
                p.delta_low = abs(float(parts[0]))
                p.delta_high = abs(float(parts[1])) if len(parts) > 1 else p.delta_low
                p.enabled = True
            except (ValueError, IndexError):
                wx.MessageBox("Invalid format. Use 'low,high' (e.g. '10,15').",
                              "Input Error", wx.OK | wx.ICON_WARNING)
            self._populate_pert_table()
        dlg.Destroy()

    # -- Uncertainty analysis (threaded) --

    def _on_run_uncertainty(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return

        n_sims = self.unc_n.GetValue()
        ci_map = {0: 0.80, 1: 0.90, 2: 0.95, 3: 0.99}
        ci = ci_map.get(self.unc_ci.GetSelection(), 0.90)

        comps = build_component_inputs(filtered, excluded_types=self.excluded_types)

        ref_params = {}
        for c in comps:
            if c.override_lambda is not None:
                continue
            for pname, pval in c.base_params.items():
                try:
                    v = float(pval)
                except (TypeError, ValueError):
                    continue
                if v == 0:
                    continue
                ref_params.setdefault(pname, {})[c.reference] = v

        specs = list(self.param_specs)
        for spec in specs:
            if spec.name in ref_params:
                spec.nominal_by_ref = ref_params[spec.name]

        runner_ref = [None]

        def work(cancel_event):
            def progress(cur, tot, msg):
                if cancel_event.is_set():
                    raise InterruptedError("Cancelled by user")
                if runner_ref[0]:
                    runner_ref[0].progress(f"MC: {cur}/{tot} samples...")

            return run_uncertainty_analysis(
                comps, specs, self.mission_hours,
                n_simulations=n_sims, confidence_level=ci,
                seed=42, progress_callback=progress,
            )

        def on_done(result):
            self.mc_result = result
            self._display_mc_results()
            self.status.SetLabel(
                f"Uncertainty analysis complete: {n_sims} samples in "
                f"{self.mc_result.runtime_seconds:.1f}s")

        self._start_analysis(work, on_done,
                             label=f"Running uncertainty analysis ({n_sims} samples)...",
                             buttons_to_disable=[self.btn_run_mc])
        runner_ref[0] = self._runner

    def _display_mc_results(self):
        r = self.mc_result
        ci_label = f"{r.confidence_level*100:.0f}% CI"

        self.mc_histogram.set_data(
            r.reliability_samples, r.mean_reliability,
            float(np.percentile(r.reliability_samples, 5)),
            float(np.percentile(r.reliability_samples, 95)),
            ci_lo=r.ci_lower, ci_hi=r.ci_upper,
            ci_label=ci_label,
            nominal=r.nominal_reliability,
            jensen_note=r.jensen_note,
        )
        self.mc_convergence.set_data(r.convergence_history)

        lines = [
            f"Nominal R(t):   {r.nominal_reliability:.6f}",
            f"Mean R(t):      {r.mean_reliability:.6f}",
            f"Median R(t):    {r.median_reliability:.6f}",
            f"Std Dev:        {r.std_reliability:.6f}",
            f"CI [{r.confidence_level*100:.0f}%]:      [{r.ci_lower:.6f}, {r.ci_upper:.6f}]",
            f"CI Width:       {r.ci_upper - r.ci_lower:.6f}",
            f"Mean HW CI:     \u00b1{r.mean_ci_halfwidth:.2e}",
            f"",
            f"Mean \u03BB:        {r.mean_lambda_fit:.2f} FIT",
            f"Std \u03BB:         {r.std_lambda_fit:.2f} FIT",
            f"CI \u03BB:          [{r.ci_lower_lambda_fit:.2f}, {r.ci_upper_lambda_fit:.2f}]",
            f"",
            f"Simulations:    {r.n_simulations:,}",
            f"Runtime:        {r.runtime_seconds:.1f}s",
            f"Uncertain params: {r.n_uncertain_params}",
            f"Uncertain comps:  {r.n_uncertain_components}/{r.n_total_components}",
        ]
        self.mc_stats_text.SetLabel("\n".join(lines))

        if r.parameter_importance:
            chart_data = [
                (f"{p['name']} ({'S' if p['shared'] else 'I'})",
                 p["srrc_sq"])
                for p in r.parameter_importance[:12]
                if p["srrc_sq"] > 0.001
            ]
            if chart_data:
                self.mc_importance.set_data(chart_data, x_label="SRRC\u00B2")

        self.jensen_label.SetLabel(r.jensen_note)
        self.jensen_label.Wrap(self.GetSize().Width - 40)

    # -- Tornado analysis (threaded) --

    def _on_run_tornado(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return

        mode = "parameter" if self.tornado_mode.GetSelection() == 0 else "sheet"
        perturbations = list(self._perturbations)
        active = list(filtered.keys())
        excluded = set(self.excluded_types)
        mh = self.mission_hours

        def work(cancel_event):
            return tornado_analysis(
                filtered, mh,
                perturbations=perturbations,
                active_sheets=active,
                excluded_types=excluded,
                mode=mode,
            )

        def on_done(result):
            self.tornado_result = result
            self.tornado_chart.set_data(
                [(e.name, e.swing) for e in result.entries[:15]],
                x_label="FIT Swing",
            )
            self.tornado_table.DeleteAllItems()
            for e in result.entries:
                _add_row(self.tornado_table, [
                    e.name, f"{e.low_value:.2f}", f"{e.base_value:.2f}",
                    f"{e.high_value:.2f}", f"{e.swing:.2f}", e.perturbation_desc,
                ])
            self.status.SetLabel(f"Tornado: {len(result.entries)} parameters analyzed")

        self._start_analysis(work, on_done,
                             label="Running tornado analysis...",
                             buttons_to_disable=[self.btn_tornado])

    # -- Criticality analysis (threaded) --

    def _on_run_criticality(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return

        top_n = self.crit_n.GetValue()
        mh = self.mission_hours
        active = list(filtered.keys())
        excluded = set(self.excluded_types)

        def work(cancel_event):
            return component_criticality(
                filtered, mh, perturbation=0.10,
                active_sheets=active,
                excluded_types=excluded,
                max_components=top_n,
            )

        def on_done(results):
            self.criticality_results = results
            self.crit_table.DeleteAllItems()
            for entry in results:
                top_field = entry.fields[0] if entry.fields else {}
                _add_row(self.crit_table, [
                    entry.reference, _trunc(entry.component_type, 18),
                    f"{entry.base_lambda_fit:.2f}",
                    top_field.get("name", "-"),
                    f"{top_field.get('elasticity', 0):.3f}",
                    f"{top_field.get('impact_pct', 0):.1f}%",
                ])
            self.status.SetLabel(
                f"Criticality: {len(results)} components analyzed")

        self._start_analysis(work, on_done,
                             label="Running criticality analysis...",
                             buttons_to_disable=[self.btn_crit])

    # =================================================================
    # Tab 3: Design Actions (Budget + Improvements + Scenarios + History)
    # =================================================================

    def _tab_design_actions(self):
        panel = scrolled.ScrolledPanel(self.nb)
        panel.SetBackgroundColour(C.BG)
        panel.SetupScrolling(scroll_x=False, scrollToTop=True)
        main = wx.BoxSizer(wx.VERTICAL)

        # --- Section 1: What-If Scenarios ---
        main.Add(self._section_label(panel, "1. What-If / Design Margin Scenarios"), 0, wx.EXPAND | wx.ALL, 6)

        wp = wx.Panel(panel)
        wp.SetBackgroundColour(C.WHITE)
        ws = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_whatif = wx.Button(wp, label="\u25B6  Run Scenarios")
        self.btn_whatif.SetBackgroundColour(C.PRI)
        self.btn_whatif.SetForegroundColour(wx.WHITE)
        self.btn_whatif.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_whatif.Bind(wx.EVT_BUTTON, self._on_run_whatif)
        ws.Add(self.btn_whatif, 0, wx.ALL, 6)
        ws.Add(wx.StaticText(wp,
            label="Evaluates environmental scenarios (Temp, Cycling, Duty). "
                  "Each recomputes every component through IEC TR 62380."),
            0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        wp.SetSizer(ws)
        main.Add(wp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.whatif_table = _make_list(panel,
            ["Scenario", "Description", "\u03BB (FIT)", "R(t)", "\u0394\u03BB %", "\u0394R"],
            [140, 280, 95, 100, 85, 95])
        self.whatif_table.SetMinSize((-1, 240))
        main.Add(self.whatif_table, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 2: Budget allocation ---
        main.Add(self._section_label(panel, "2. Reliability Target & Budget Allocation"), 0, wx.EXPAND | wx.ALL, 6)

        tp = wx.Panel(panel)
        tp.SetBackgroundColour(C.WHITE)
        ts = wx.BoxSizer(wx.HORIZONTAL)
        ts.Add(wx.StaticText(tp, label="R target:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.opt_target = wx.TextCtrl(tp, value="0.999", size=(80, -1))
        ts.Add(self.opt_target, 0, wx.ALL, 6)
        ts.Add(wx.StaticText(tp, label="Strategy:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.opt_strategy = wx.Choice(tp,
            choices=["Proportional", "Equal", "Complexity", "Criticality"])
        self.opt_strategy.SetSelection(0)
        ts.Add(self.opt_strategy, 0, wx.ALL, 6)
        ts.Add(wx.StaticText(tp, label="Margin %:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.opt_margin = wx.SpinCtrlDouble(tp, min=0, max=50, initial=10,
                                             inc=1, size=(65, -1))
        ts.Add(self.opt_margin, 0, wx.ALL, 6)
        self.btn_budget = wx.Button(tp, label="\u25B6  Allocate Budget")
        self.btn_budget.SetBackgroundColour(C.PRI)
        self.btn_budget.SetForegroundColour(wx.WHITE)
        self.btn_budget.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_budget.Bind(wx.EVT_BUTTON, self._on_run_budget)
        ts.Add(self.btn_budget, 0, wx.ALL, 6)
        tp.SetSizer(ts)
        main.Add(tp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.budget_info = wx.StaticText(panel, label="Set target and run budget allocation.")
        self.budget_info.SetForegroundColour(C.TXT_M)
        self.budget_info.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main.Add(self.budget_info, 0, wx.ALL, 10)

        self.budget_list = _make_list(panel,
            ["Reference", "Type", "Actual FIT", "Budget FIT", "Margin", "Util %", "Status"],
            [90, 120, 85, 85, 80, 70, 60])
        self.budget_list.SetMinSize((-1, 260))
        main.Add(self.budget_list, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 3: Improvements ---
        main.Add(self._section_label(panel, "3. Improvement Recommendations (Derating + Swap)"), 0, wx.EXPAND | wx.ALL, 6)

        rp = wx.Panel(panel)
        rp.SetBackgroundColour(C.WHITE)
        rps = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_improve = wx.Button(rp, label="\u25B6  Generate Recommendations")
        self.btn_improve.SetBackgroundColour(C.OK)
        self.btn_improve.SetForegroundColour(wx.WHITE)
        self.btn_improve.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_improve.Bind(wx.EVT_BUTTON, self._on_run_improvements)
        rps.Add(self.btn_improve, 0, wx.ALL, 6)
        self.improve_info = wx.StaticText(rp, label="Run budget first, then generate improvements.")
        self.improve_info.SetForegroundColour(C.TXT_M)
        rps.Add(self.improve_info, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        rp.SetSizer(rps)
        main.Add(rp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.improve_list = _make_list(panel,
            ["#", "Reference", "Type", "Action", "Current", "Proposed",
             "FIT Saved", "Feasibility"],
            [40, 80, 110, 120, 80, 80, 80, 75])
        self.improve_list.SetMinSize((-1, 280))
        main.Add(self.improve_list, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 4: Parameter What-If (bidirectional) ---
        main.Add(self._section_label(panel, "4. Parameter What-If (single component)"), 0, wx.EXPAND | wx.ALL, 6)

        bip = wx.Panel(panel)
        bip.SetBackgroundColour(C.WHITE)
        bis = wx.BoxSizer(wx.HORIZONTAL)
        bis.Add(wx.StaticText(bip, label="Component:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.wi_ref = wx.TextCtrl(bip, size=(75, -1))
        bis.Add(self.wi_ref, 0, wx.ALL, 6)
        bis.Add(wx.StaticText(bip, label="Param:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.wi_param = wx.TextCtrl(bip, size=(110, -1))
        bis.Add(self.wi_param, 0, wx.ALL, 6)
        bis.Add(wx.StaticText(bip, label="New value:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.wi_val = wx.TextCtrl(bip, size=(85, -1))
        bis.Add(self.wi_val, 0, wx.ALL, 6)
        self.btn_wi = wx.Button(bip, label="Evaluate")
        self.btn_wi.Bind(wx.EVT_BUTTON, self._on_whatif_single)
        bis.Add(self.btn_wi, 0, wx.ALL, 6)
        bip.SetSizer(bis)
        main.Add(bip, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.wi_result = wx.StaticText(panel, label="")
        self.wi_result.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main.Add(self.wi_result, 0, wx.ALL, 10)

        # --- Section 5: History ---
        main.Add(self._section_label(panel, "5. Reliability History"), 0, wx.EXPAND | wx.ALL, 6)

        snap_panel = wx.Panel(panel)
        snap_panel.SetBackgroundColour(C.WHITE)
        snap_s = wx.BoxSizer(wx.HORIZONTAL)
        snap_s.Add(wx.StaticText(snap_panel, label="Version:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.snap_label = wx.TextCtrl(snap_panel, size=(120, -1))
        snap_s.Add(self.snap_label, 0, wx.ALL, 6)
        snap_s.Add(wx.StaticText(snap_panel, label="Notes:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.snap_notes = wx.TextCtrl(snap_panel, size=(200, -1))
        snap_s.Add(self.snap_notes, 0, wx.ALL, 6)
        btn_snap = wx.Button(snap_panel, label="Save Snapshot")
        btn_snap.Bind(wx.EVT_BUTTON, self._on_save_snapshot)
        snap_s.Add(btn_snap, 0, wx.ALL, 6)
        btn_load = wx.Button(snap_panel, label="Load History")
        btn_load.Bind(wx.EVT_BUTTON, self._on_load_history)
        snap_s.Add(btn_load, 0, wx.ALL, 6)
        snap_panel.SetSizer(snap_s)
        main.Add(snap_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.history_list = _make_list(panel,
            ["Version", "Date", "System FIT", "R(t)", "Components", "Notes"],
            [110, 150, 95, 105, 85, 210])
        self.history_list.SetMinSize((-1, 180))
        main.Add(self.history_list, 0, wx.EXPAND | wx.ALL, 6)

        panel.SetSizer(main)
        return panel

    # -- What-If Scenarios (threaded) --

    def _on_run_whatif(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return
        active = list(filtered.keys())
        excluded = set(self.excluded_types)
        mh = self.mission_hours

        def work(cancel_event):
            return scenario_analysis(
                filtered, mh,
                active_sheets=active,
                excluded_types=excluded,
            )

        def on_done(result):
            self.scenario_result = result
            self.whatif_table.DeleteAllItems()
            for s in result.scenarios:
                color = C.OK if s.delta_lambda_pct < 0 else (
                    C.FAIL if s.delta_lambda_pct > 5 else C.TXT)
                _add_row(self.whatif_table, [
                    s.name, s.description, f"{s.lambda_fit:.2f}",
                    f"{s.reliability:.6f}", f"{s.delta_lambda_pct:+.1f}%",
                    f"{s.delta_reliability:+.6f}",
                ], color=color)
            self.status.SetLabel(f"Scenarios: {len(result.scenarios)} evaluated")

        self._start_analysis(work, on_done,
                             label="Running what-if scenarios...",
                             buttons_to_disable=[self.btn_whatif])

    # -- Budget allocation (threaded) --

    def _on_run_budget(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return

        try:
            target_r = float(self.opt_target.GetValue())
            if not (0.0 < target_r < 1.0):
                raise ValueError("Target must be between 0 and 1")
        except ValueError as e:
            wx.MessageBox(f"Invalid target R: {e}", "Input Error", wx.OK | wx.ICON_ERROR)
            return

        strat_map = {0: "proportional", 1: "equal", 2: "complexity", 3: "criticality"}
        strategy = strat_map.get(self.opt_strategy.GetSelection(), "proportional")
        margin = self.opt_margin.GetValue()
        active = list(filtered.keys())
        mh = self.mission_hours

        def work(cancel_event):
            return allocate_budget(
                filtered, mh, target_reliability=target_r,
                strategy=strategy, active_sheets=active,
                margin_percent=margin)

        def on_done(result):
            self.budget_result = result
            self.budget_list.DeleteAllItems()
            for sb in result.sheet_budgets:
                for cb in sb.component_budgets:
                    color = C.OK if cb.within_budget else C.FAIL
                    _add_row(self.budget_list, [
                        cb.reference, _trunc(cb.component_type, 16),
                        f"{cb.actual_fit:.2f}", f"{cb.budget_fit:.2f}",
                        f"{cb.margin_fit:+.2f}", f"{cb.utilization*100:.0f}%",
                        "PASS" if cb.within_budget else "OVER",
                    ], color=color)
            icon = "\u2705" if result.system_within_budget else "\u274C"
            self.budget_info.SetLabel(
                f"{icon}  Target R={target_r} => {result.target_fit:.1f} FIT, "
                f"Actual {result.actual_fit:.1f} FIT, "
                f"Margin {result.system_margin_fit:+.1f} FIT. "
                f"{result.components_over_budget}/{result.total_components} over budget.")
            self.budget_info.SetForegroundColour(
                C.OK if result.system_within_budget else C.FAIL)
            self.status.SetLabel(f"Budget: {result.total_components} components allocated")

        self._start_analysis(work, on_done,
                             label="Allocating budget...",
                             buttons_to_disable=[self.btn_budget])

    # -- Improvements (threaded) --

    def _on_run_improvements(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return

        active = list(filtered.keys())
        mh = self.mission_hours
        target_fit = self.budget_result.target_fit if self.budget_result else (
            lambda_from_reliability(0.999, mh) * 1e9)
        all_comps = self._all_components()
        sys_fit = sum(_safe_float(c.get("lambda", 0)) for c in all_comps) * 1e9

        def work(cancel_event):
            derating = compute_derating_guidance(
                filtered, mh, target_fit,
                active_sheets=active, top_n=500)
            swaps = rank_all_swaps(all_comps, sys_fit, max_per_component=3)
            return derating, swaps, sys_fit, target_fit

        def on_done(result):
            derating, swaps, sf, tf = result
            self.derating_result = derating
            self.swap_results = swaps

            merged = []
            if derating:
                for rec in derating.recommendations:
                    merged.append({
                        "ref": rec.reference, "type": rec.component_type,
                        "action": f"Derate {rec.parameter}",
                        "current": f"{rec.current_value:.2f}",
                        "proposed": f"{rec.required_value:.2f}",
                        "fit_saved": rec.system_fit_reduction,
                        "feasibility": rec.feasibility,
                    })
            for sw in swaps:
                if sw.get("delta_fit", 0) < 0:
                    merged.append({
                        "ref": sw.get("reference", "?"),
                        "type": sw.get("component_type", "?"),
                        "action": sw.get("swap_type", "swap"),
                        "current": "",
                        "proposed": _trunc(sw.get("description", ""), 18),
                        "fit_saved": abs(sw.get("delta_fit", 0)),
                        "feasibility": "swap",
                    })
            merged.sort(key=lambda x: -x["fit_saved"])

            self.improve_list.DeleteAllItems()
            total_saved = 0.0
            for i, rec in enumerate(merged, 1):
                total_saved += rec["fit_saved"]
                _add_row(self.improve_list, [
                    str(i), rec["ref"], _trunc(rec["type"], 14),
                    rec["action"], rec["current"], rec["proposed"],
                    f"{rec['fit_saved']:.2f}", rec["feasibility"],
                ])

            gap = (sf - tf) if sf > tf else 0
            self.improve_info.SetLabel(
                f"{len(merged)} recommendations. "
                f"Total potential savings: {total_saved:.1f} FIT"
                + (f" (covers {total_saved/gap*100:.0f}% of gap)" if gap > 0 else "")
            )
            self.status.SetLabel(f"Improvements: {len(merged)} recommendations generated")

        self._start_analysis(work, on_done,
                             label="Generating improvement recommendations...",
                             buttons_to_disable=[self.btn_improve])

    # -- Single-param what-if (instant, no thread needed) --

    def _on_whatif_single(self, event):
        ref = self.wi_ref.GetValue().strip()
        param = self.wi_param.GetValue().strip()
        val_str = self.wi_val.GetValue().strip()
        if not ref or not param or not val_str:
            wx.MessageBox("Fill in component reference, parameter name, and new value.",
                          "Missing Input", wx.OK | wx.ICON_WARNING)
            return
        try:
            new_val = float(val_str)
        except ValueError:
            wx.MessageBox(f"'{val_str}' is not a valid number.", "Input Error", wx.OK | wx.ICON_ERROR)
            return

        comp = None
        for c in self._all_components():
            if c.get("ref", "") == ref:
                comp = c
                break
        if comp is None:
            self.wi_result.SetLabel(f"Component '{ref}' not found in active sheets.")
            self.wi_result.SetForegroundColour(C.FAIL)
            return

        try:
            result = single_param_whatif(
                comp, param, new_val, self.system_lambda, self.mission_hours)
        except Exception as e:
            self.wi_result.SetLabel(f"Error: {e}")
            self.wi_result.SetForegroundColour(C.FAIL)
            return

        self.wi_result.SetForegroundColour(C.TXT)
        self.wi_result.SetLabel(
            f"Component {ref}: {param} = {result['old_value']} -> {new_val}\n"
            f"  Component FIT: {result['comp_fit_before']:.2f} -> {result['comp_fit_after']:.2f} "
            f"({result['comp_delta_fit']:+.2f})\n"
            f"  System FIT:    {result['sys_fit_before']:.2f} -> {result['sys_fit_after']:.2f}\n"
            f"  R(t):          {result['r_before']:.6f} -> {result['r_after']:.6f} "
            f"({result['delta_r']:+.6f})"
        )

    # -- History --

    def _on_save_snapshot(self, event):
        if not self.project_path:
            wx.MessageBox("No project path set.", "Error", wx.OK | wx.ICON_WARNING)
            return
        label = self.snap_label.GetValue().strip() or "v1"
        notes = self.snap_notes.GetValue().strip()
        try:
            snap = create_snapshot(self._active_data, self.system_lambda,
                                   self.mission_hours, label, notes)
            path = save_snapshot(snap, self.project_path)
            self.status.SetLabel(f"Snapshot '{label}' saved to {path}")
            self._on_load_history(None)
        except Exception as e:
            wx.MessageBox(f"Failed to save snapshot: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def _on_load_history(self, event):
        if not self.project_path:
            return
        try:
            snapshots = load_snapshots(self.project_path)
            self.history_list.DeleteAllItems()
            for s in reversed(snapshots):
                _add_row(self.history_list, [
                    s.version_label, s.timestamp[:19],
                    f"{s.system_fit:.2f}", f"{s.system_reliability:.6f}",
                    str(s.n_components), s.notes[:40],
                ])
        except Exception as e:
            wx.MessageBox(f"Failed to load history: {e}", "Error", wx.OK | wx.ICON_ERROR)

    # =================================================================
    # Tab 4: Report
    # =================================================================

    def _tab_report(self):
        panel = wx.Panel(self.nb)
        panel.SetBackgroundColour(C.BG)
        main = wx.BoxSizer(wx.VERTICAL)

        rp = wx.Panel(panel)
        rp.SetBackgroundColour(C.WHITE)
        rs = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_html = wx.Button(rp, label="Generate HTML Report")
        self.btn_html.SetBackgroundColour(C.PRI)
        self.btn_html.SetForegroundColour(wx.WHITE)
        self.btn_html.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_html.Bind(wx.EVT_BUTTON, self._on_gen_html)
        rs.Add(self.btn_html, 0, wx.ALL, 10)
        self.btn_pdf = wx.Button(rp, label="Generate PDF Report")
        self.btn_pdf.SetBackgroundColour(C.OK)
        self.btn_pdf.SetForegroundColour(wx.WHITE)
        self.btn_pdf.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.btn_pdf.Bind(wx.EVT_BUTTON, self._on_gen_pdf)
        rs.Add(self.btn_pdf, 0, wx.ALL, 10)
        rs.Add(wx.StaticText(rp,
            label="Reports include all analyses run in this session. "
                  "Run analyses first, then generate."),
            0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        rp.SetSizer(rs)
        main.Add(rp, 0, wx.EXPAND | wx.ALL, 6)

        self.report_status = wx.StaticText(panel, label="")
        self.report_status.SetForegroundColour(C.TXT_M)
        self.report_status.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main.Add(self.report_status, 0, wx.ALL, 14)

        panel.SetSizer(main)
        return panel

    def _build_report_data(self):
        filtered = self._filtered()
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / 8760

        mc_dict = self.mc_result.to_dict() if self.mc_result else None
        tornado_dict = self.tornado_result.to_dict() if self.tornado_result else None
        scenario_dict = self.scenario_result.to_dict() if self.scenario_result else None

        crit_list = None
        if self.criticality_results:
            crit_list = [
                {"reference": e.reference, "component_type": e.component_type,
                 "base_lambda_fit": e.base_lambda_fit, "fields": e.fields}
                for e in self.criticality_results
            ]

        budget_dict = None
        if self.budget_result:
            budget_dict = self.budget_result.to_dict()

        derating_dict = None
        if self.derating_result:
            derating_dict = self.derating_result.to_dict()

        project_name = Path(self.project_path).name if self.project_path else "Reliability Report"

        return ReportData(
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
            sensitivity=None,
            sheet_mc=None,
            criticality=crit_list,
            tornado=tornado_dict,
            design_margin=scenario_dict,
        )

    def _on_gen_html(self, event):
        try:
            report_data = self._build_report_data()
            gen = ReportGenerator(logo_path=self.logo_path, logo_mime=self.logo_mime)
            html = gen.generate_html(report_data)
        except Exception as e:
            wx.MessageBox(f"Failed to build report: {e}", "Report Error", wx.OK | wx.ICON_ERROR)
            return

        dlg = wx.FileDialog(self, "Save HTML Report",
                            wildcard="HTML files (*.html)|*.html",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
                self.report_status.SetLabel(f"HTML report saved: {path}")
            except Exception as e:
                wx.MessageBox(f"Failed to write report: {e}", "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _on_gen_pdf(self, event):
        try:
            report_data = self._build_report_data()
            gen = ReportGenerator(logo_path=self.logo_path, logo_mime=self.logo_mime)
        except Exception as e:
            wx.MessageBox(f"Failed to build report: {e}", "Report Error", wx.OK | wx.ICON_ERROR)
            return

        dlg = wx.FileDialog(self, "Save PDF Report",
                            wildcard="PDF files (*.pdf)|*.pdf",
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                gen.generate_pdf(report_data, path)
                self.report_status.SetLabel(f"PDF report saved: {path}")
            except Exception as ex:
                html_path = path.rsplit(".", 1)[0] + ".html"
                try:
                    html = gen.generate_html(report_data)
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    self.report_status.SetLabel(
                        f"PDF failed ({ex}). HTML saved: {html_path}")
                except Exception as ex2:
                    wx.MessageBox(f"Report generation failed: {ex2}", "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    # =================================================================
    # Type filter handler
    # =================================================================

    def _on_type_filter(self, event):
        self.excluded_types = set()
        for tn, cb in self._type_cbs.items():
            if not cb.GetValue():
                self.excluded_types.add(tn)
        self._refresh_overview()
