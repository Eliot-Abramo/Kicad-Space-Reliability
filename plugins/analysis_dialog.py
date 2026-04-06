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
    TornadoResult, TornadoEntry, ScenarioResult, ScenarioEntry,
    CriticalityEntry, SmartAction,
    TornadoPerturbation, DEFAULT_PERTURBATIONS, ORBIT_PARAMS,
    tornado_analysis, scenario_analysis, component_criticality,
    single_param_whatif, get_active_sheet_paths, identify_smart_actions,
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
from .component_swap import rank_all_swaps, _get_swap_options
from .growth_tracking import (
    create_snapshot, save_snapshot, load_snapshots,
    compare_revisions, build_growth_timeline,
    delete_snapshot, ReliabilitySnapshot,
)
try:
    from .ui.theme import PALETTE, apply_compact_fonts, apply_theme_recursively, dip_px, dip_size, platform_point_size, style_text_like, style_list_ctrl, ui_font
    from .ui.windowing import center_dialog, get_display_client_area
except ImportError:
    from import_compat import ensure_plugin_paths

    ensure_plugin_paths()
    from theme import PALETTE, apply_compact_fonts, apply_theme_recursively, dip_px, dip_size, platform_point_size, style_text_like, style_list_ctrl, ui_font
    from windowing import center_dialog, get_display_client_area


# =====================================================================
# Colour Scheme
# =====================================================================
class C:
    """Compact colour palette."""
    BG       = PALETTE.background
    WHITE    = PALETTE.card_bg
    HEADER   = PALETTE.header_bg
    TXT      = PALETTE.text
    TXT_M    = PALETTE.text_muted
    TXT_L    = PALETTE.text_soft
    PRI      = PALETTE.primary
    ACCENT   = PALETTE.accent
    OK       = PALETTE.success
    WARN     = PALETTE.warning
    FAIL     = PALETTE.danger
    BORDER   = PALETTE.border
    GRID     = PALETTE.grid
    ROW_ALT  = PALETTE.row_alt
    FIELD_BG = PALETTE.field_bg
    INFO_BG  = PALETTE.info_bg
    BAR = [wx.Colour(59, 130, 246), wx.Colour(14, 165, 142), wx.Colour(245, 158, 11),
           wx.Colour(239, 68, 68), wx.Colour(99, 102, 241), wx.Colour(20, 184, 166),
           wx.Colour(168, 85, 247), wx.Colour(249, 115, 22)]


# =====================================================================
# Chart Panels
# =====================================================================

def _trunc(s, maxlen=22):
    return (s[:maxlen-1] + "\u2026") if len(s) > maxlen else s


def _adaptive_font(dc, base_size, w, h, min_size=8):
    scale = min(w / 600.0, h / 400.0)
    sz = max(min_size, int(base_size * max(0.7, min(1.3, scale))))
    sz = platform_point_size(sz, minimum=min_size)
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

        n_bins = min(50, max(20, cw // 10))
        hist, edges = np.histogram(self.samples, bins=n_bins)
        max_count = max(hist) if max(hist) > 0 else 1
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
                x0 = int(v2x(edges[i]))
                x1 = int(v2x(edges[i + 1]))
                bar_w = max(1, x1 - x0 - 1)
                bh = max(1, int((count / max_count) * ch))
                dc.DrawRectangle(x0, mt + ch - bh, bar_w, bh)

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
            if val_range < 1e-4:
                lbl = f"{val:.6f}"
            elif val_range < 0.01:
                lbl = f"{val:.5f}"
            elif val_range < 0.1:
                lbl = f"{val:.4f}"
            else:
                lbl = f"{val:.3f}"
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

class _SortableListCtrl(wx.ListCtrl):
    """ListCtrl with column-click sorting and per-item tooltips."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._sort_col = -1
        self._sort_asc = True
        self._rows = []
        self._colors = {}
        self.Bind(wx.EVT_LIST_COL_CLICK, self._on_col_click)
        self.Bind(wx.EVT_MOTION, self._on_motion)

    def append_row(self, values, color=None):
        row_idx = len(self._rows)
        self._rows.append([str(v) for v in values])
        if color:
            self._colors[row_idx] = color
        idx = self.InsertItem(self.GetItemCount(), str(values[0]))
        for i, v in enumerate(values[1:], 1):
            self.SetItem(idx, i, str(v))
        if color:
            self.SetItemTextColour(idx, color)
        return idx

    def DeleteAllItems(self):
        super().DeleteAllItems()
        self._rows.clear()
        self._colors.clear()

    def _on_col_click(self, event):
        col = event.GetColumn()
        if col == self._sort_col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True

        def sort_key(row_tuple):
            _, row = row_tuple
            val = row[col] if col < len(row) else ""
            val = val.rstrip("%").lstrip("+")
            try:
                return (0, float(val))
            except ValueError:
                return (1, val.lower())

        indexed = list(enumerate(self._rows))
        indexed.sort(key=sort_key, reverse=not self._sort_asc)

        self.DeleteAllItems()
        new_colors = {}
        for new_i, (orig_i, row) in enumerate(indexed):
            self.InsertItem(new_i, row[0])
            for ci, v in enumerate(row[1:], 1):
                self.SetItem(new_i, ci, v)
            if orig_i in self._colors:
                self.SetItemTextColour(new_i, self._colors[orig_i])
                new_colors[new_i] = self._colors[orig_i]

        self._rows = [row for _, row in indexed]
        old_colors = self._colors.copy()
        self._colors = {}
        for new_i, (orig_i, _) in enumerate(indexed):
            if orig_i in old_colors:
                self._colors[new_i] = old_colors[orig_i]

    def _on_motion(self, event):
        idx, flags = self.HitTest(event.GetPosition())
        if idx >= 0 and idx < len(self._rows):
            tip = " | ".join(self._rows[idx])
            self.SetToolTip(tip)
        else:
            self.SetToolTip("")
        event.Skip()


def _make_list(parent, columns, col_widths=None):
    lc = _SortableListCtrl(parent,
                            style=wx.LC_REPORT | wx.BORDER_SIMPLE)
    style_list_ctrl(lc)
    lc.SetFont(ui_font(lc, role="small"))
    for i, name in enumerate(columns):
        w = col_widths[i] if col_widths and i < len(col_widths) else 120
        lc.InsertColumn(i, name, width=dip_px(parent, w))
    return lc


def _add_row(lc, values, color=None):
    if isinstance(lc, _SortableListCtrl):
        return lc.append_row(values, color)
    idx = lc.InsertItem(lc.GetItemCount(), str(values[0]))
    for i, v in enumerate(values[1:], 1):
        lc.SetItem(idx, i, str(v))
    if color:
        lc.SetItemTextColour(idx, color)
    return idx


def _autosize_columns(lc, min_width=90):
    """Auto-fit every column to its content, enforcing a minimum width."""
    scaled_min_width = dip_px(lc, min_width)
    for col in range(lc.GetColumnCount()):
        lc.SetColumnWidth(col, wx.LIST_AUTOSIZE)
        if lc.GetColumnWidth(col) < scaled_min_width:
            lc.SetColumnWidth(col, scaled_min_width)


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _style_button(btn, role="primary"):
    palette = {
        "primary": (C.PRI, wx.WHITE),
        "success": (C.OK, wx.WHITE),
        "accent": (C.ACCENT, wx.WHITE),
        "danger": (C.FAIL, wx.WHITE),
        "warning": (C.WARN, wx.WHITE),
    }
    bg, fg = palette.get(role, (C.PRI, wx.WHITE))
    btn.SetBackgroundColour(bg)
    btn.SetForegroundColour(fg)
    btn.SetFont(ui_font(btn, role="body", weight=wx.FONTWEIGHT_BOLD))
    btn.SetMinSize(wx.Size(-1, dip_px(btn, 36)))


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
        rect = get_display_client_area(parent=parent)
        w = min(1400, int(rect.Width * 0.88))
        h = min(950, int(rect.Height * 0.90))
        super().__init__(parent, title=title, size=(w, h),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        self.SetMinSize(dip_size(self, 1000, 700))
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
        self._positioned_on_show = False

        self._build_ui()
        apply_compact_fonts(self)
        apply_theme_recursively(self, background=C.BG)
        self._load_persisted_state()
        self._on_load_history(None)
        self.Bind(wx.EVT_SHOW, self._on_dialog_show)
        wx.CallAfter(self._refresh_dialog_layout, True)
        wx.CallLater(120, self._refresh_dialog_layout, True)
        wx.CallLater(320, self._refresh_dialog_layout, True)

    # =================================================================
    # Persistence: save/load analysis state
    # =================================================================

    def _state_path(self) -> Optional[Path]:
        if not self.project_path:
            return None
        p = Path(self.project_path)
        rel_dir = p / "Reliability" if not p.name == "Reliability" else p
        return rel_dir / "analysis_state.json"

    def _save_persisted_state(self):
        """Serialize analysis results to JSON so they survive dialog close."""
        import json
        sp = self._state_path()
        if not sp:
            return
        state = {}
        if self.mc_result:
            try:
                # Save lightweight MC stats (no huge sample arrays)
                d = self.mc_result.to_dict()
                d.pop("samples", None)
                d.pop("lambda_samples_fit", None)
                state["mc"] = d
            except Exception as e:
                state["_mc_error"] = str(e)
        if self.tornado_result:
            try:
                state["tornado"] = self.tornado_result.to_dict()
            except Exception as e:
                state["_tornado_error"] = str(e)
        if self.scenario_result:
            try:
                state["scenario"] = self.scenario_result.to_dict()
            except Exception as e:
                state["_scenario_error"] = str(e)
        if self.criticality_results:
            try:
                state["criticality"] = [
                    {"reference": e.reference, "component_type": e.component_type,
                     "base_lambda_fit": e.base_lambda_fit, "fields": e.fields}
                    for e in self.criticality_results
                ]
            except Exception as e:
                state["_criticality_error"] = str(e)
        if self.budget_result:
            try:
                state["budget"] = self.budget_result.to_dict()
            except Exception as e:
                state["_budget_error"] = str(e)
        try:
            sp.parent.mkdir(parents=True, exist_ok=True)
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            try:
                wx.CallAfter(self.status.SetLabel,
                             f"Warning: could not save analysis state: {e}")
            except Exception:
                pass

    def _load_persisted_state(self):
        """Load previously saved analysis results."""
        import json
        sp = self._state_path()
        if not sp or not sp.exists():
            return
        try:
            with open(sp, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            return

        if "mc" in state and state["mc"]:
            try:
                mc_dict = state["mc"]
                self.mc_result = UncertaintyResult(
                    nominal_lambda=mc_dict.get("nominal_lambda", 0),
                    nominal_reliability=mc_dict.get("nominal_reliability", 1),
                    nominal_mttf_hours=mc_dict.get("nominal_mttf_hours", float('inf')),
                    mean_reliability=mc_dict.get("mean_reliability", 1),
                    median_reliability=mc_dict.get("median_reliability", 1),
                    std_reliability=mc_dict.get("std_reliability", 0),
                    ci_lower=mc_dict.get("ci_lower", 1),
                    ci_upper=mc_dict.get("ci_upper", 1),
                    confidence_level=mc_dict.get("confidence_level", 0.9),
                    mean_lambda_fit=mc_dict.get("mean_lambda_fit", 0),
                    std_lambda_fit=mc_dict.get("std_lambda_fit", 0),
                    ci_lower_lambda_fit=mc_dict.get("ci_lower_lambda_fit", 0),
                    ci_upper_lambda_fit=mc_dict.get("ci_upper_lambda_fit", 0),
                    mean_ci_halfwidth=mc_dict.get("mean_ci_halfwidth", 0),
                    lambda_samples=np.array(mc_dict.get("lambda_samples_fit", [])) / 1e9,
                    reliability_samples=np.array(mc_dict.get("samples", [])),
                    parameter_importance=mc_dict.get("parameter_importance", []),
                    convergence_history=mc_dict.get("convergence_history", []),
                    n_simulations=mc_dict.get("n_simulations", 0),
                    n_uncertain_params=mc_dict.get("n_uncertain_params", 0),
                    n_shared_params=mc_dict.get("n_shared_params", 0),
                    n_uncertain_components=mc_dict.get("n_uncertain_components", 0),
                    n_total_components=mc_dict.get("n_total_components", 0),
                    runtime_seconds=mc_dict.get("runtime_seconds", 0),
                    jensen_note=mc_dict.get("jensen_note", ""),
                )
                # Display MC results in UI
                mean_r = self.mc_result.mean_reliability
                p5_r = np.percentile(self.mc_result.reliability_samples, 5) if len(self.mc_result.reliability_samples) > 0 else mean_r
                p95_r = np.percentile(self.mc_result.reliability_samples, 95) if len(self.mc_result.reliability_samples) > 0 else mean_r
                ci_pct = int(self.mc_result.confidence_level * 100)
                self.mc_histogram.set_data(
                    self.mc_result.reliability_samples.tolist() if len(self.mc_result.reliability_samples) > 0 else [],
                    mean_r, p5_r, p95_r, 
                    self.mc_result.ci_lower, self.mc_result.ci_upper,
                    ci_label=f"{ci_pct}% CI", 
                    nominal=self.mc_result.nominal_reliability,
                    jensen_note=self.mc_result.jensen_note,
                )
                self.mc_convergence.set_data(self.mc_result.convergence_history)
                
                # Update stats text
                stats_text = (
                    f"R(t) Nominal: {self.mc_result.nominal_reliability:.6f}\n"
                    f"R(t) Mean:    {mean_r:.6f} ± {self.mc_result.std_reliability:.6f}\n"
                    f"R(t) CI:      [{self.mc_result.ci_lower:.6f}, {self.mc_result.ci_upper:.6f}]\n"
                    f"λ (FIT):      {self.mc_result.mean_lambda_fit:.2f} ± {self.mc_result.std_lambda_fit:.2f}\n"
                    f"N samples:    {self.mc_result.n_simulations}  "
                    f"(N uncertain: {self.mc_result.n_uncertain_components}/{self.mc_result.n_total_components})\n"
                    f"Runtime:      {self.mc_result.runtime_seconds:.1f} s"
                )
                self.mc_stats_text.SetLabel(stats_text)
                self.jensen_label.SetLabel(self.mc_result.jensen_note)
                
                # Update importance chart
                if self.mc_result.parameter_importance:
                    top_imp = self.mc_result.parameter_importance[:10]
                    self.mc_importance.set_data(
                        [(x["name"], x["srrc_sq"]) for x in top_imp],
                        x_label="SRRC²"
                    )
                
                self.status.SetLabel(f"MC results restored: {self.mc_result.n_simulations} samples")
            except Exception as e:
                try:
                    self.status.SetLabel(f"Note: MC results could not be restored ({type(e).__name__})")
                except Exception:
                    pass

        if "tornado" in state and state["tornado"]:
            try:
                entries = [TornadoEntry(**e) for e in state["tornado"].get("entries", [])]
                self.tornado_result = TornadoResult(
                    entries=entries,
                    base_lambda_fit=state["tornado"].get("base_lambda_fit", 0),
                    base_reliability=state["tornado"].get("base_reliability", 1),
                    mission_hours=state["tornado"].get("mission_hours", self.mission_hours),
                )
                self.tornado_chart.set_data(
                    [(e.name, e.swing) for e in entries[:15]],
                    x_label="FIT Swing",
                )
                self.tornado_table.DeleteAllItems()
                for e in entries:
                    _add_row(self.tornado_table, [
                        e.name, f"{e.low_value:.2f}", f"{e.base_value:.2f}",
                        f"{e.high_value:.2f}", f"{e.swing:.2f}", e.perturbation_desc,
                    ])
                _autosize_columns(self.tornado_table)
            except Exception:
                pass

        if "criticality" in state and state["criticality"]:
            try:
                self.criticality_results = [
                    CriticalityEntry(
                        reference=e["reference"],
                        component_type=e["component_type"],
                        base_lambda_fit=e["base_lambda_fit"],
                        fields=e.get("fields", []),
                    )
                    for e in state["criticality"]
                ]
                self.crit_table.DeleteAllItems()
                for entry in self.criticality_results:
                    top_field = entry.fields[0] if entry.fields else {}
                    _add_row(self.crit_table, [
                        entry.reference, entry.component_type,
                        f"{entry.base_lambda_fit:.2f}",
                        top_field.get("name", "-"),
                        f"{top_field.get('elasticity', 0):.3f}",
                        f"{top_field.get('impact_pct', 0):.1f}%",
                    ])
                _autosize_columns(self.crit_table)
            except Exception:
                pass

        if "scenario" in state and state["scenario"]:
            try:
                entries = [ScenarioEntry(**e) for e in state["scenario"].get("scenarios", [])]
                self.scenario_result = ScenarioResult(
                    scenarios=entries,
                    baseline_lambda_fit=state["scenario"].get("baseline_lambda_fit", 0),
                    baseline_reliability=state["scenario"].get("baseline_reliability", 1),
                    mission_hours=state["scenario"].get("mission_hours", self.mission_hours),
                )
                self.whatif_table.DeleteAllItems()
                for s in entries:
                    color = C.OK if s.delta_lambda_pct < 0 else (
                        C.FAIL if s.delta_lambda_pct > 5 else C.TXT)
                    _add_row(self.whatif_table, [
                        s.name, s.description, f"{s.lambda_fit:.2f}",
                        f"{s.reliability:.6f}", f"{s.delta_lambda_pct:+.1f}%",
                        f"{s.delta_reliability:+.6f}",
                    ], color=color)
                _autosize_columns(self.whatif_table)
            except Exception:
                pass

    def _on_clear_all(self, event):
        """Clear all analysis results and delete persisted state."""
        if wx.MessageBox("Clear all analysis results?", "Confirm",
                         wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
            return
        self.mc_result = None
        self.tornado_result = None
        self.scenario_result = None
        self.criticality_results = []
        self.budget_result = None
        self.derating_result = None
        self.swap_results = []

        for table in (self.tornado_table, self.crit_table, self.whatif_table,
                      self.budget_sheet_list, self.budget_list, self.improve_list, self.smart_list):
            table.DeleteAllItems()

        self.mc_histogram.samples = None
        self.mc_histogram.Refresh()
        self.mc_convergence.history = []
        self.mc_convergence.Refresh()
        self.mc_stats_text.SetLabel("Run analysis to see results.")
        self.mc_importance.data = []
        self.mc_importance.Refresh()
        self.jensen_label.SetLabel("")
        self.analysis_summary.SetLabel(
            "No analyses have been run yet. Start with Uncertainty Analysis to put a confidence band around the current FIT estimate."
        )
        self.mc_interpretation.SetLabel(
            "Interpretation: Monte Carlo shows the spread in mission reliability when your uncertain inputs vary within the bounds you entered."
        )
        self.tornado_interpretation.SetLabel(
            "Interpretation: larger swing means that parameter is a stronger local design lever at the current operating point."
        )
        self.crit_interpretation.SetLabel(
            "Interpretation: higher elasticity means a small proportional parameter change creates a larger proportional change in component failure rate."
        )
        self.tornado_chart.data = []
        self.tornado_chart.Refresh()
        self.budget_info.SetLabel("Set target and run budget allocation.")
        if hasattr(self, "budget_cards"):
            self.budget_cards["target"].SetLabel("--")
            self.budget_cards["actual"].SetLabel("--")
            self.budget_cards["gap"].SetLabel("--")
        self.wi_result.SetValue("")
        self.smart_detail.SetLabel("")

        sp = self._state_path()
        if sp and sp.exists():
            try:
                sp.unlink()
            except Exception:
                pass

        self._refresh_report_preview()
        self.status.SetLabel("All results cleared.")

    def EndModal(self, retCode):
        """Override to save state before closing."""
        self._save_persisted_state()
        super().EndModal(retCode)

    def Destroy(self):
        """Also save state if dialog is destroyed directly."""
        self._save_persisted_state()
        return super().Destroy()

    # =================================================================
    # Data helpers
    # =================================================================

    def _filtered(self):
        result = {}
        seen_refs = set()
        for path, data in self._active_data.items():
            comps = []
            for c in data.get("components", []):
                if c.get("class", "Unknown") in self.excluded_types:
                    continue
                ref = c.get("ref", "?")
                if ref in seen_refs:
                    continue
                seen_refs.add(ref)
                comps.append(c)
            new_lam = sum(_safe_float(c.get("lambda", 0)) for c in comps)
            result[path] = {**data, "components": comps, "lambda": new_lam}
        return result

    def _all_components(self):
        comps = []
        seen = set()
        for data in self._filtered().values():
            for c in data.get("components", []):
                ref = c.get("ref", "?")
                if ref not in seen:
                    seen.add(ref)
                    comps.append(c)
        return comps

    def _find_component(self, reference):
        for comp in self._all_components():
            if comp.get("ref", "") == reference:
                return comp
        return None

    def _current_system_lambda(self):
        return sum(_safe_float(d.get("lambda", 0)) for d in self._filtered().values())

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
        t.SetFont(ui_font(hdr, role="title", weight=wx.FONTWEIGHT_BOLD))
        t.SetForegroundColour(wx.WHITE)
        hs.Add(t, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)
        r = self._sys_r()
        yrs = self.mission_hours / 8760
        info = (f"\u03BB = {self._sys_fit():.2f} FIT  |  "
                f"R(t) = {r:.6f}  |  {yrs:.1f}y  |  "
                f"{len(self._active_data)} sheets")
        il = wx.StaticText(hdr, label=info)
        il.SetFont(ui_font(hdr, role="small"))
        il.SetForegroundColour(wx.Colour(200, 215, 240))
        hs.Add(il, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)

        btn_clear = wx.Button(hdr, label="Clear All")
        _style_button(btn_clear, "warning")
        btn_clear.Bind(wx.EVT_BUTTON, self._on_clear_all)
        btn_clear.SetToolTip("Clear all analysis results and reset tables")
        hs.Add(btn_clear, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)

        self.btn_cancel = wx.Button(hdr, label="Cancel")
        _style_button(self.btn_cancel, "danger")
        self.btn_cancel.Bind(wx.EVT_BUTTON, lambda e: self._cancel_analysis())
        self.btn_cancel.SetToolTip("Cancel running analysis")
        hs.Add(self.btn_cancel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)

        hdr.SetSizer(hs)
        sizer.Add(hdr, 0, wx.EXPAND)

        # Notebook
        self.nb = wx.Notebook(self)
        self.nb.SetBackgroundColour(C.BG)
        self.nb.SetPadding(wx.Size(20, 8))
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self._on_notebook_page_changed)
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

    def _on_dialog_show(self, event):
        if event.IsShown():
            wx.CallAfter(self._refresh_dialog_layout, not self._positioned_on_show)
            wx.CallLater(120, self._refresh_dialog_layout, not self._positioned_on_show)
            wx.CallLater(320, self._refresh_dialog_layout, not self._positioned_on_show)
        event.Skip()

    def _on_notebook_page_changed(self, event):
        wx.CallAfter(self._refresh_dialog_layout)
        event.Skip()

    def _refresh_dialog_layout(self, recenter=False):
        try:
            apply_theme_recursively(self, background=C.BG)
            self.Layout()
            if hasattr(self, "nb"):
                apply_theme_recursively(self.nb, background=C.BG)
                self.nb.Layout()
                for page_index in range(self.nb.GetPageCount()):
                    page = self.nb.GetPage(page_index)
                    if not page:
                        continue
                    apply_theme_recursively(page, background=C.BG)
                    page.Layout()
                    if isinstance(page, scrolled.ScrolledPanel):
                        page.SetupScrolling(scroll_x=False, scrollToTop=False)
                        page.FitInside()
                    page.SendSizeEvent()
                    for child in page.GetChildren():
                        child.SendSizeEvent()
                        child.Refresh()
                    page.Refresh()
            for widget_name in (
                "contrib_chart", "mc_histogram", "mc_convergence", "mc_importance", "tornado_chart"
            ):
                widget = getattr(self, widget_name, None)
                if widget:
                    widget.SendSizeEvent()
                    widget.Refresh()
            self.SendSizeEvent()
            if recenter:
                center_dialog(self, self.GetParent())
                self._positioned_on_show = True
            self.Refresh()
            self.Update()
        except Exception:
            pass

    # =================================================================
    # Tab 1: Overview (was Dashboard)
    # =================================================================

    def _tab_overview(self):
        panel = wx.Panel(self.nb)
        panel.SetBackgroundColour(C.BG)
        main = wx.BoxSizer(wx.VERTICAL)

        hero = wx.Panel(panel)
        hero.SetBackgroundColour(C.INFO_BG)
        hs = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(hero, label="Design Overview")
        title.SetFont(ui_font(hero, role="section", weight=wx.FONTWEIGHT_BOLD))
        title.SetForegroundColour(C.HEADER)
        hs.Add(title, 0, wx.ALL, 10)
        self.overview_message = wx.StaticText(
            hero,
            label="This page highlights current reliability risk, the biggest contributors, and the next analysis to run.",
        )
        self.overview_message.SetForegroundColour(C.TXT_M)
        self.overview_message.Wrap(1200)
        hs.Add(self.overview_message, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        hero.SetSizer(hs)
        main.Add(hero, 0, wx.EXPAND | wx.ALL, 6)

        # Component type filter
        fp = wx.Panel(panel)
        fp.SetBackgroundColour(C.WHITE)
        fs = wx.BoxSizer(wx.VERTICAL)
        fl = wx.StaticText(fp, label="Component Types (uncheck to exclude from all analyses):")
        fl.SetFont(ui_font(fp, role="body", weight=wx.FONTWEIGHT_BOLD))
        fs.Add(fl, 0, wx.ALL, 8)
        tr = wx.WrapSizer(wx.HORIZONTAL)
        for tn in sorted(self._component_types()):
            cb = wx.CheckBox(fp, label=tn)
            cb.SetValue(True)
            cb.SetFont(ui_font(cb, role="small"))
            cb.Bind(wx.EVT_CHECKBOX, self._on_type_filter)
            tr.Add(cb, 0, wx.ALL, 4)
            self._type_cbs[tn] = cb
        fs.Add(tr, 0, wx.LEFT, 10)
        fp.SetSizer(fs)
        main.Add(fp, 0, wx.EXPAND | wx.ALL, 6)

        stats = wx.BoxSizer(wx.HORIZONTAL)
        self.summary_cards = {}
        for key, label in [
            ("fit", "System FIT"),
            ("reliability", "Mission R(t)"),
            ("mttf", "MTTF"),
            ("components", "Components"),
            ("classification", "Classification Review"),
        ]:
            card, value = self._overview_card(panel, label)
            self.summary_cards[key] = value
            stats.Add(card, 1, wx.EXPAND | wx.ALL, 4)
        main.Add(stats, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        action_row = wx.BoxSizer(wx.HORIZONTAL)
        next_card, next_label = self._overview_card(panel, "Next Best Action", wide=True)
        signal_card, signal_label = self._overview_card(panel, "What This Means", wide=True)
        self.next_action_label = next_label
        self.signal_label = signal_label
        action_row.Add(next_card, 1, wx.EXPAND | wx.RIGHT, 6)
        action_row.Add(signal_card, 1, wx.EXPAND)
        main.Add(action_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # Charts row
        charts = wx.BoxSizer(wx.HORIZONTAL)
        self.contrib_chart = HBarPanel(panel, "FIT Contributions (Top 15)")
        charts.Add(self.contrib_chart, 2, wx.EXPAND | wx.RIGHT, 6)

        card = wx.Panel(panel)
        card.SetBackgroundColour(C.WHITE)
        cs = wx.BoxSizer(wx.VERTICAL)
        cl = wx.StaticText(card, label="System Summary")
        cl.SetFont(ui_font(card, role="section", weight=wx.FONTWEIGHT_BOLD))
        cs.Add(cl, 0, wx.ALL, 12)
        self.dash_summary = wx.StaticText(card, label="")
        self.dash_summary.SetFont(ui_font(card, role="mono"))
        cs.Add(self.dash_summary, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 12)
        card.SetSizer(cs)
        charts.Add(card, 1, wx.EXPAND)
        main.Add(charts, 1, wx.EXPAND | wx.ALL, 6)

        # Contribution table
        self.contrib_list = _make_list(panel,
            ["Component", "Type", "\u03BB (FIT)", "Contribution %", "Cumulative %"],
            [180, 200, 110, 130, 120])
        main.Add(self.contrib_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        panel.SetSizer(main)
        self._refresh_overview()
        return panel

    def _overview_card(self, parent, label, wide=False):
        card = wx.Panel(parent)
        card.SetBackgroundColour(C.WHITE)
        sizer = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(card, label=label)
        title.SetForegroundColour(C.TXT_M)
        title.SetFont(ui_font(card, role="small", weight=wx.FONTWEIGHT_BOLD))
        value = wx.StaticText(card, label="--")
        value.SetForegroundColour(C.TXT)
        value.SetFont(ui_font(card, role="section" if wide else "title", weight=wx.FONTWEIGHT_BOLD))
        if wide:
            value.Wrap(300)
        sizer.Add(title, 0, wx.ALL, 10)
        sizer.Add(value, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        card.SetSizer(sizer)
        return card, value

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
                ref, ctype, f"{fit:.2f}",
                f"{pct:.1f}%", f"{cum:.1f}%",
            ])
        _autosize_columns(self.contrib_list)

        r = reliability_from_lambda(total_lam, self.mission_hours)
        mttf = 1.0 / total_lam if total_lam > 0 else float('inf')
        mttf_yr = mttf / 8760
        review_count = sum(
            1
            for data in filtered.values()
            for comp in data.get("components", [])
            if comp.get("classification_review_required")
        )
        high_conf = sum(
            1
            for data in filtered.values()
            for comp in data.get("components", [])
            if comp.get("classification_confidence") == "high"
        )
        total_comps = len(comps)
        if total_comps == 0:
            next_action = "Import components and build the system model first."
            signal = "No active analysis data yet."
        elif review_count:
            next_action = f"Review {review_count} flagged component classifications before publishing a report."
            signal = "The estimate is usable, but some component categories are still heuristic and should be confirmed."
        elif not self.mc_result:
            next_action = "Run Uncertainty Analysis next to quantify the confidence band around the current estimate."
            signal = "You have a deterministic estimate and contributor ranking, but no uncertainty range yet."
        elif not self.tornado_result:
            next_action = "Run Tornado Sensitivity next to identify the highest-leverage design parameters."
            signal = "The uncertainty band is known; the next step is finding which parameters move FIT the most."
        elif not self.criticality_results:
            next_action = "Run Component Criticality next to find which part parameters are worth tightening."
            signal = "You know the system levers; now identify which specific components deserve design effort."
        else:
            next_action = "Open Design Actions to turn the current analyses into targeted improvement work."
            signal = "The current dataset is decision-ready: you have contributors, uncertainty, and ranked levers."

        if hasattr(self, "summary_cards"):
            self.summary_cards["fit"].SetLabel(f"{total_fit:.2f} FIT")
            self.summary_cards["reliability"].SetLabel(f"{r:.6f}")
            self.summary_cards["mttf"].SetLabel(f"{mttf_yr:.1f} years")
            self.summary_cards["components"].SetLabel(f"{total_comps} across {len(filtered)} sheets")
            self.summary_cards["classification"].SetLabel(
                f"{high_conf} high-confidence, {review_count} flagged"
            )
            self.next_action_label.SetLabel(next_action)
            self.signal_label.SetLabel(signal)

        lines = [
            f"System \u03BB:    {total_fit:.2f} FIT",
            f"R(t):         {r:.6f}",
            f"MTTF:         {mttf_yr:.1f} years",
            f"Mission:      {self.mission_hours/8760:.1f} years",
            f"Components:   {len(comps)} ({review_count} flagged for review)",
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
        hint_panel.SetBackgroundColour(C.INFO_BG)
        hs = wx.BoxSizer(wx.VERTICAL)
        hint = wx.StaticText(hint_panel,
            label="Sensitivity workflow: 1) quantify uncertainty around the current estimate, "
                  "2) identify which parameters move system FIT the most, "
                  "3) find which component parameters are worth tightening in the design.")
        hint.SetFont(ui_font(hint_panel, role="small", style=wx.FONTSTYLE_ITALIC))
        hint.SetForegroundColour(C.TXT)
        hint.Wrap(1300)
        hs.Add(hint, 0, wx.ALL, 10)
        self.analysis_summary = wx.StaticText(
            hint_panel,
            label="No analyses have been run yet. Start with Uncertainty Analysis to put a confidence band around the current FIT estimate.",
        )
        self.analysis_summary.SetForegroundColour(C.TXT_M)
        self.analysis_summary.Wrap(1300)
        hs.Add(self.analysis_summary, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
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
        self.unc_pct = wx.SpinCtrlDouble(qp, min=0, max=50, initial=0, inc=1, size=dip_size(qp, 82, -1))
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
        self.unc_n = wx.SpinCtrl(qp, min=500, max=100000, initial=5000, size=dip_size(qp, 96, -1))
        qs.Add(self.unc_n, 0, wx.ALL, 6)

        qs.Add(wx.StaticText(qp, label="CI:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        self.unc_ci = wx.Choice(qp, choices=["80%", "90%", "95%", "99%"])
        self.unc_ci.SetSelection(1)
        qs.Add(self.unc_ci, 0, wx.ALL, 6)

        self.btn_run_mc = wx.Button(qp, label="\u25B6  Run Uncertainty")
        _style_button(self.btn_run_mc, "primary")
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
        pl.SetFont(ui_font(param_panel, role="body", weight=wx.FONTWEIGHT_BOLD))
        ps.Add(pl, 0, wx.ALL, 8)

        self.param_list = _make_list(param_panel,
            ["Parameter", "Components", "Mode", "Low Bound", "High Bound", "Distribution", "Shared"],
            [170, 100, 110, 110, 110, 110, 90])
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
        sl.SetFont(ui_font(sp, role="body", weight=wx.FONTWEIGHT_BOLD))
        ss.Add(sl, 0, wx.ALL, 8)
        self.mc_stats_text = wx.StaticText(sp, label="Run analysis to see results.")
        self.mc_stats_text.SetFont(ui_font(sp, role="mono"))
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
        self.jensen_label.SetFont(ui_font(panel, role="caption", style=wx.FONTSTYLE_ITALIC))
        main.Add(self.jensen_label, 0, wx.LEFT | wx.BOTTOM, 8)
        self.mc_interpretation = wx.StaticText(
            panel,
            label="Interpretation: Monte Carlo shows the spread in mission reliability when your uncertain inputs vary within the bounds you entered.",
        )
        self.mc_interpretation.SetForegroundColour(C.TXT_M)
        self.mc_interpretation.Wrap(1200)
        main.Add(self.mc_interpretation, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

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
        _style_button(self.btn_tornado, "success")
        self.btn_tornado.Bind(wx.EVT_BUTTON, self._on_run_tornado)
        ts.Add(self.btn_tornado, 0, wx.ALL, 6)
        ts.Add(wx.StaticText(tp,
            label="Perturbations use physical units. Double-click table to edit."),
            0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        tp.SetSizer(ts)
        main.Add(tp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.pert_list = _make_list(panel,
            ["Parameter", "Low (-)", "High (+)", "Unit", "Enabled"],
            [180, 100, 100, 100, 90])
        self.pert_list.SetMinSize((-1, 160))
        self._populate_pert_table()
        self.pert_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit_pert)
        main.Add(self.pert_list, 0, wx.EXPAND | wx.ALL, 6)

        self.tornado_chart = HBarPanel(panel, "Tornado: System FIT Swing")
        self.tornado_chart.SetMinSize((-1, 280))
        main.Add(self.tornado_chart, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.tornado_table = _make_list(panel,
            ["Parameter", "Low FIT", "Base FIT", "High FIT", "Swing", "Perturbation"],
            [220, 110, 110, 110, 110, 220])
        self.tornado_table.SetMinSize((-1, 160))
        main.Add(self.tornado_table, 0, wx.EXPAND | wx.ALL, 6)
        self.tornado_interpretation = wx.StaticText(
            panel,
            label="Interpretation: larger swing means that parameter is a stronger local design lever at the current operating point.",
        )
        self.tornado_interpretation.SetForegroundColour(C.TXT_M)
        self.tornado_interpretation.Wrap(1200)
        main.Add(self.tornado_interpretation, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # --- Section 3: Component Criticality ---
        main.Add(self._section_label(panel, "Step 3: Component Criticality (Elasticity)"), 0, wx.EXPAND | wx.ALL, 6)

        elas_note = wx.StaticText(panel,
            label="Elasticity measures how sensitive the system failure rate is to each "
                  "component's parameters. A 10% perturbation is applied to each numeric "
                  "field; the resulting change in component FIT is normalized to give a "
                  "dimensionless elasticity: e = (\u0394\u03BB/\u03BB) / (\u0394x/x). "
                  "Higher elasticity = more design leverage. 'Impact %' shows the "
                  "contribution of that parameter's swing relative to the total system FIT.")
        elas_note.SetForegroundColour(C.TXT_M)
        elas_note.SetFont(ui_font(panel, role="caption", style=wx.FONTSTYLE_ITALIC))
        elas_note.Wrap(1200)
        main.Add(elas_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        cp = wx.Panel(panel)
        cp.SetBackgroundColour(C.WHITE)
        css = wx.BoxSizer(wx.HORIZONTAL)
        css.Add(wx.StaticText(cp, label="Top N (0 = all):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.crit_n = wx.SpinCtrl(cp, min=0, max=500, initial=0, size=dip_size(cp, 82, -1))
        css.Add(self.crit_n, 0, wx.ALL, 6)
        self.btn_crit = wx.Button(cp, label="\u25B6  Run Criticality")
        _style_button(self.btn_crit, "primary")
        self.btn_crit.Bind(wx.EVT_BUTTON, self._on_run_criticality)
        css.Add(self.btn_crit, 0, wx.ALL, 6)
        cp.SetSizer(css)
        main.Add(cp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.crit_table = _make_list(panel,
            ["Reference", "Type", "\u03BB (FIT)", "Top Parameter", "Elasticity", "Impact %"],
            [130, 200, 110, 220, 110, 110])
        self.crit_table.SetMinSize((-1, 240))
        main.Add(self.crit_table, 0, wx.EXPAND | wx.ALL, 6)
        self.crit_interpretation = wx.StaticText(
            panel,
            label="Interpretation: higher elasticity means a small proportional parameter change creates a larger proportional change in component failure rate.",
        )
        self.crit_interpretation.SetForegroundColour(C.TXT_M)
        self.crit_interpretation.Wrap(1200)
        main.Add(self.crit_interpretation, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Bind param table editing
        self.param_list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self._on_edit_param)
        self._populate_param_table()

        panel.SetSizer(main)
        return panel

    def _section_label(self, parent, text):
        p = wx.Panel(parent)
        p.SetBackgroundColour(C.INFO_BG)
        s = wx.BoxSizer(wx.VERTICAL)
        l = wx.StaticText(p, label=text)
        l.SetFont(ui_font(p, role="section", weight=wx.FONTWEIGHT_BOLD))
        l.SetForegroundColour(C.HEADER)
        s.Add(l, 0, wx.ALL, 10)
        p.SetSizer(s)
        return p

    # -- Param table management --

    def _populate_param_table(self, pct=0.0, dist="pert"):
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
        _autosize_columns(self.param_list)

    def _on_apply_quick_unc(self, event):
        pct = self.unc_pct.GetValue()
        dist = "pert" if self.unc_dist.GetSelection() == 0 else "uniform"
        self._populate_param_table(pct, dist)

    def _on_edit_param(self, event):
        idx = event.GetIndex()
        if idx < 0 or idx >= len(self.param_specs):
            return
        spec = self.param_specs[idx]

        dlg = wx.Dialog(self, title=f"Edit: {spec.name}", size=dip_size(self, 400, 300))
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
            _autosize_columns(self.param_list)
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
        _autosize_columns(self.pert_list)

    def _on_edit_pert(self, event):
        idx = event.GetIndex()
        if idx < 0 or idx >= len(self._perturbations):
            return
        p = self._perturbations[idx]

        dlg = wx.Dialog(self, title=f"Edit Perturbation: {p.param_name}",
                        size=dip_size(self, 380, 240))
        ds = wx.BoxSizer(wx.VERTICAL)
        gs = wx.FlexGridSizer(4, 2, 8, 12)
        gs.AddGrowableCol(1)

        gs.Add(wx.StaticText(dlg, label="Enabled:"), 0, wx.ALIGN_CENTER_VERTICAL)
        en_cb = wx.CheckBox(dlg, label="Include in Tornado analysis")
        en_cb.SetValue(p.enabled)
        gs.Add(en_cb, 1, wx.EXPAND)

        gs.Add(wx.StaticText(dlg, label="Low (-):"), 0, wx.ALIGN_CENTER_VERTICAL)
        lo_ctrl = wx.TextCtrl(dlg, value=f"{p.delta_low}")
        gs.Add(lo_ctrl, 1, wx.EXPAND)

        gs.Add(wx.StaticText(dlg, label="High (+):"), 0, wx.ALIGN_CENTER_VERTICAL)
        hi_ctrl = wx.TextCtrl(dlg, value=f"{p.delta_high}")
        gs.Add(hi_ctrl, 1, wx.EXPAND)

        gs.Add(wx.StaticText(dlg, label="Unit:"), 0, wx.ALIGN_CENTER_VERTICAL)
        gs.Add(wx.StaticText(dlg, label=p.unit or "(none)"), 0)

        ds.Add(gs, 0, wx.EXPAND | wx.ALL, 12)
        ds.Add(dlg.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL),
               0, wx.EXPAND | wx.ALL, 12)
        dlg.SetSizer(ds)

        if dlg.ShowModal() == wx.ID_OK:
            try:
                p.delta_low = abs(float(lo_ctrl.GetValue()))
                p.delta_high = abs(float(hi_ctrl.GetValue()))
            except ValueError:
                wx.MessageBox("Invalid number format.",
                              "Input Error", wx.OK | wx.ICON_WARNING)
                dlg.Destroy()
                return
            p.enabled = en_cb.GetValue()
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
                if pname in ORBIT_PARAMS:
                    continue
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
        self.analysis_summary.SetLabel(
            f"Uncertainty complete: {r.n_simulations:,} samples give a {ci_label} reliability band of "
            f"[{r.ci_lower:.6f}, {r.ci_upper:.6f}] around a nominal value of {r.nominal_reliability:.6f}."
        )
        self.mc_interpretation.SetLabel(
            "Interpretation: if this interval is wide, your result is being driven by uncertain inputs rather than only by the nominal design point."
        )

        if r.parameter_importance:
            # Show top 12 parameters by SRRC^2 -- no minimum threshold
            # so that the chart is never empty when there are uncertain params
            sorted_imp = sorted(r.parameter_importance,
                                key=lambda p: -p["srrc_sq"])
            chart_data = [
                (f"{p['name']} ({'S' if p['shared'] else 'I'})",
                 p["srrc_sq"])
                for p in sorted_imp[:12]
                if p["srrc_sq"] > 0
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
            _autosize_columns(self.tornado_table)
            if result.entries:
                top = result.entries[0]
                self.analysis_summary.SetLabel(
                    f"Tornado complete: '{top.name}' is currently the strongest local sensitivity lever with a {top.swing:.2f} FIT swing."
                )
                self.tornado_interpretation.SetLabel(
                    f"Interpretation: prioritize work on '{top.name}' first; it changes system FIT more than the other currently tested perturbations."
                )
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
                    entry.reference, entry.component_type,
                    f"{entry.base_lambda_fit:.2f}",
                    top_field.get("name", "-"),
                    f"{top_field.get('elasticity', 0):.3f}",
                    f"{top_field.get('impact_pct', 0):.1f}%",
                ])
            _autosize_columns(self.crit_table)
            if results and results[0].fields:
                top_field = results[0].fields[0]
                self.analysis_summary.SetLabel(
                    f"Criticality complete: {results[0].reference} is the highest-leverage component, driven most by '{top_field.get('name', '-')}'."
                )
                self.crit_interpretation.SetLabel(
                    f"Interpretation: start with {results[0].reference} and the parameter '{top_field.get('name', '-')}' if you need to tighten the design efficiently."
                )
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

        # --- Section 0: Smart Design Actions (auto-identified) ---
        main.Add(self._section_label(panel,
            "Smart Design Actions (auto-identified from analyses)"), 0, wx.EXPAND | wx.ALL, 6)

        sa_panel = wx.Panel(panel)
        sa_panel.SetBackgroundColour(C.WHITE)
        sa_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_smart = wx.Button(sa_panel, label="\u25B6  Identify Best Actions")
        _style_button(self.btn_smart, "accent")
        self.btn_smart.Bind(wx.EVT_BUTTON, self._on_smart_actions)
        sa_sizer.Add(self.btn_smart, 0, wx.ALL, 6)
        sa_sizer.Add(wx.StaticText(sa_panel,
            label="Combines Tornado + SRRC + Criticality to find the parameters "
                  "with the highest design leverage. Run Step 2/3 first for best results."),
            0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        sa_panel.SetSizer(sa_sizer)
        main.Add(sa_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.smart_list = _make_list(panel,
            ["#", "Parameter", "Score", "FIT Gain", "Suggestion",
             "Components", "Source"],
            [40, 160, 90, 100, 350, 140, 150])
        self.smart_list.SetMinSize((-1, 220))
        main.Add(self.smart_list, 0, wx.EXPAND | wx.ALL, 6)

        self.smart_detail = wx.StaticText(panel, label="")
        self.smart_detail.SetFont(ui_font(panel, role="small", style=wx.FONTSTYLE_ITALIC))
        self.smart_detail.SetForegroundColour(C.TXT_M)
        main.Add(self.smart_detail, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # --- Section 1: What-If Scenarios ---
        main.Add(self._section_label(panel, "1. What-If / Design Margin Scenarios"), 0, wx.EXPAND | wx.ALL, 6)

        wp = wx.Panel(panel)
        wp.SetBackgroundColour(C.WHITE)
        ws = wx.BoxSizer(wx.VERTICAL)
        row1 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_whatif = wx.Button(wp, label="\u25B6  Run Scenarios")
        _style_button(self.btn_whatif, "primary")
        self.btn_whatif.Bind(wx.EVT_BUTTON, self._on_run_whatif)
        row1.Add(self.btn_whatif, 0, wx.ALL, 6)
        row1.Add(wx.StaticText(wp,
            label="Evaluates design-actionable and environmental scenarios. "
                  "Each recomputes every component through IEC TR 62380."),
            0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        ws.Add(row1, 0, wx.EXPAND)

        # Custom scenario row
        row2 = wx.BoxSizer(wx.HORIZONTAL)
        row2.Add(wx.StaticText(wp, label="Custom:"), 0,
                 wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.custom_sc_name = wx.TextCtrl(wp, value="", size=dip_size(wp, 120, -1))
        self.custom_sc_name.SetHint("Name")
        row2.Add(self.custom_sc_name, 0, wx.ALL, 4)
        all_params = self._collect_all_param_names()
        self.custom_sc_param = wx.Choice(wp, choices=all_params, size=dip_size(wp, 156, -1))
        if all_params:
            self.custom_sc_param.SetSelection(0)
        row2.Add(self.custom_sc_param, 0, wx.ALL, 4)
        self.custom_sc_op = wx.Choice(wp, choices=["multiply", "add", "set to"])
        self.custom_sc_op.SetSelection(0)
        row2.Add(self.custom_sc_op, 0, wx.ALL, 4)
        self.custom_sc_val = wx.TextCtrl(wp, value="", size=dip_size(wp, 88, -1))
        self.custom_sc_val.SetHint("Value")
        row2.Add(self.custom_sc_val, 0, wx.ALL, 4)
        btn_add_sc = wx.Button(wp, label="+ Add")
        btn_add_sc.Bind(wx.EVT_BUTTON, self._on_add_custom_scenario)
        row2.Add(btn_add_sc, 0, wx.ALL, 4)
        self.custom_sc_label = wx.StaticText(wp, label="")
        self.custom_sc_label.SetForegroundColour(C.TXT_M)
        row2.Add(self.custom_sc_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        ws.Add(row2, 0, wx.EXPAND)

        wp.SetSizer(ws)
        main.Add(wp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self._custom_scenarios = []

        self.whatif_table = _make_list(panel,
            ["Scenario", "Description", "\u03BB (FIT)", "R(t)", "\u0394\u03BB %", "\u0394R"],
            [180, 360, 110, 120, 100, 110])
        self.whatif_table.SetMinSize((-1, 240))
        main.Add(self.whatif_table, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 2: Budget allocation ---
        main.Add(self._section_label(panel, "2. Reliability Target Closure Planning"), 0, wx.EXPAND | wx.ALL, 6)

        tp = wx.Panel(panel)
        tp.SetBackgroundColour(C.WHITE)
        ts = wx.BoxSizer(wx.HORIZONTAL)
        ts.Add(wx.StaticText(tp, label="R target:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.opt_target = wx.TextCtrl(tp, value="0.999", size=dip_size(tp, 96, -1))
        ts.Add(self.opt_target, 0, wx.ALL, 6)
        ts.Add(wx.StaticText(tp, label="Strategy:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.opt_strategy = wx.Choice(tp,
            choices=["Proportional (Recommended)", "Equal", "Complexity", "Criticality"])
        self.opt_strategy.SetSelection(0)
        ts.Add(self.opt_strategy, 0, wx.ALL, 6)
        ts.Add(wx.StaticText(tp, label="Margin %:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.opt_margin = wx.SpinCtrlDouble(tp, min=0, max=50, initial=10,
                                             inc=1, size=dip_size(tp, 78, -1))
        ts.Add(self.opt_margin, 0, wx.ALL, 6)
        self.btn_budget = wx.Button(tp, label="\u25B6  Allocate Budget")
        _style_button(self.btn_budget, "primary")
        self.btn_budget.Bind(wx.EVT_BUTTON, self._on_run_budget)
        ts.Add(self.btn_budget, 0, wx.ALL, 6)
        tp.SetSizer(ts)
        main.Add(tp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        budget_help = wx.StaticText(panel,
            label="How it works: Converts your target R into a total FIT ceiling, applies the requested design margin, "
                  "and then shows the remaining gap to close. The strategy changes how the allowable FIT is apportioned, "
                  "but the main outputs are always the same: total gap, pressured sheets, and the components that must save the most FIT.\n"
                  "Strategy note: Proportional preserves the current design balance and is the recommended baseline. "
                  "Equal, Complexity, and Criticality are alternate lenses for planning conversations.")
        budget_help.SetForegroundColour(C.TXT_L)
        budget_help.SetFont(ui_font(panel, role="caption", style=wx.FONTSTYLE_ITALIC))
        budget_help.Wrap(1200)
        main.Add(budget_help, 0, wx.LEFT | wx.RIGHT, 12)

        self.budget_info = wx.StaticText(panel, label="Set target and run budget allocation.")
        self.budget_info.SetForegroundColour(C.TXT_M)
        self.budget_info.SetFont(ui_font(panel, role="body"))
        main.Add(self.budget_info, 0, wx.ALL, 10)

        budget_cards = wx.BoxSizer(wx.HORIZONTAL)
        self.budget_cards = {}
        for key, label in [
            ("target", "Effective Budget"),
            ("actual", "Actual FIT"),
            ("gap", "Gap To Close"),
        ]:
            card, value = self._overview_card(panel, label)
            self.budget_cards[key] = value
            budget_cards.Add(card, 1, wx.EXPAND | wx.ALL, 4)
        main.Add(budget_cards, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.budget_sheet_list = _make_list(panel,
            ["Sheet", "Actual FIT", "Budget FIT", "Gap To Close", "Util %", "Hotspots"],
            [180, 110, 110, 120, 100, 120])
        self.budget_sheet_list.SetMinSize((-1, 140))
        main.Add(self.budget_sheet_list, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.budget_list = _make_list(panel,
            ["Reference", "Type", "Actual FIT", "Budget FIT", "Save Needed", "Util %", "Status", "Why"],
            [120, 180, 100, 100, 110, 90, 80, 240])
        self.budget_list.SetMinSize((-1, 260))
        main.Add(self.budget_list, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 3: Improvements ---
        main.Add(self._section_label(panel, "3. Improvement Recommendations (Derating + Swap)"), 0, wx.EXPAND | wx.ALL, 6)

        rp = wx.Panel(panel)
        rp.SetBackgroundColour(C.WHITE)
        rps = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_improve = wx.Button(rp, label="\u25B6  Generate Recommendations")
        _style_button(self.btn_improve, "success")
        self.btn_improve.Bind(wx.EVT_BUTTON, self._on_run_improvements)
        rps.Add(self.btn_improve, 0, wx.ALL, 6)
        self.improve_info = wx.StaticText(rp, label="Run target-closure planning first, then generate improvements against the active gap.")
        self.improve_info.SetForegroundColour(C.TXT_M)
        rps.Add(self.improve_info, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        rp.SetSizer(rps)
        main.Add(rp, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.improve_list = _make_list(panel,
            ["#", "Reference", "Type", "Action", "Current", "Proposed",
             "FIT Saved", "Feasibility"],
            [40, 120, 180, 180, 120, 150, 100, 100])
        self.improve_list.SetMinSize((-1, 280))
        main.Add(self.improve_list, 0, wx.EXPAND | wx.ALL, 6)

        # --- Section 4: Parameter What-If (bidirectional) ---
        main.Add(self._section_label(panel, "4. Parameter What-If (single component)"), 0, wx.EXPAND | wx.ALL, 6)

        bip = wx.Panel(panel)
        bip.SetBackgroundColour(C.WHITE)
        bis = wx.BoxSizer(wx.HORIZONTAL)
        bis.Add(wx.StaticText(bip, label="Component:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        comp_refs = sorted(set(c.get("ref", "?") for c in self._all_components()))
        self.wi_ref = wx.Choice(bip, choices=comp_refs, size=dip_size(bip, 110, -1))
        if comp_refs:
            self.wi_ref.SetSelection(0)
        self.wi_ref.Bind(wx.EVT_CHOICE, self._on_wi_ref_changed)
        bis.Add(self.wi_ref, 0, wx.ALL, 6)
        bis.Add(wx.StaticText(bip, label="Param:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.wi_param = wx.Choice(bip, choices=[], size=dip_size(bip, 160, -1))
        self.wi_param.Bind(wx.EVT_CHOICE, self._on_wi_param_changed)
        bis.Add(self.wi_param, 0, wx.ALL, 6)
        bis.Add(wx.StaticText(bip, label="New value:"), 0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self._wi_val_panel = bip
        self._wi_val_sizer = wx.BoxSizer(wx.VERTICAL)
        self.wi_val = wx.TextCtrl(bip, size=dip_size(bip, 164, -1))
        self._wi_val_sizer.Add(self.wi_val, 0, wx.ALL, 0)
        bis.Add(self._wi_val_sizer, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        self.btn_wi = wx.Button(bip, label="Evaluate")
        self.btn_wi.Bind(wx.EVT_BUTTON, self._on_whatif_single)
        bis.Add(self.btn_wi, 0, wx.ALL, 6)
        bip.SetSizer(bis)
        main.Add(bip, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.wi_hint = wx.StaticText(panel, label="Pick a component and parameter to preview a local design-point change.")
        self.wi_hint.SetForegroundColour(C.TXT_M)
        main.Add(self.wi_hint, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        if comp_refs:
            self._on_wi_ref_changed(None)

        self.wi_result = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.wi_result.SetFont(ui_font(panel, role="mono"))
        style_text_like(self.wi_result, read_only=True)
        self.wi_result.SetMinSize(wx.Size(-1, dip_px(panel, 132)))
        self.wi_result.SetMaxSize(wx.Size(-1, dip_px(panel, 176)))
        main.Add(self.wi_result, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # --- Section 5: History ---
        main.Add(self._section_label(panel, "5. Reliability History"), 0, wx.EXPAND | wx.ALL, 6)

        snap_panel = wx.Panel(panel)
        snap_panel.SetBackgroundColour(C.WHITE)
        snap_s = wx.BoxSizer(wx.HORIZONTAL)
        snap_s.Add(wx.StaticText(snap_panel, label="Version:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.snap_label = wx.TextCtrl(snap_panel, size=dip_size(snap_panel, 136, -1))
        snap_s.Add(self.snap_label, 0, wx.ALL, 6)
        snap_s.Add(wx.StaticText(snap_panel, label="Notes:"), 0,
               wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        self.snap_notes = wx.TextCtrl(snap_panel, size=dip_size(snap_panel, 236, -1))
        snap_s.Add(self.snap_notes, 0, wx.ALL, 6)
        btn_snap = wx.Button(snap_panel, label="Save Snapshot")
        btn_snap.Bind(wx.EVT_BUTTON, self._on_save_snapshot)
        snap_s.Add(btn_snap, 0, wx.ALL, 6)
        btn_load = wx.Button(snap_panel, label="Load History")
        btn_load.Bind(wx.EVT_BUTTON, self._on_load_history)
        snap_s.Add(btn_load, 0, wx.ALL, 6)
        btn_compare = wx.Button(snap_panel, label="Compare Latest 2")
        btn_compare.Bind(wx.EVT_BUTTON, self._on_compare_latest_history)
        snap_s.Add(btn_compare, 0, wx.ALL, 6)
        btn_delete = wx.Button(snap_panel, label="Delete Selected")
        btn_delete.Bind(wx.EVT_BUTTON, self._on_delete_selected_history)
        snap_s.Add(btn_delete, 0, wx.ALL, 6)
        snap_panel.SetSizer(snap_s)
        main.Add(snap_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 6)

        self.history_list = _make_list(panel,
            ["Version", "Date", "System FIT", "R(t)", "Components", "Notes"],
            [130, 170, 110, 120, 110, 280])
        self.history_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_history_selected)
        self.history_list.SetMinSize((-1, 180))
        main.Add(self.history_list, 0, wx.EXPAND | wx.ALL, 6)

        self.history_summary = wx.StaticText(panel, label="Save a snapshot to start a design history.")
        self.history_summary.SetForegroundColour(C.TXT_M)
        main.Add(self.history_summary, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.history_detail = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE
        )
        self.history_detail.SetFont(ui_font(panel, role="mono"))
        style_text_like(self.history_detail, read_only=True)
        self.history_detail.SetMinSize((-1, 150))
        main.Add(self.history_detail, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(main)
        return panel

    # -- Smart Design Actions --

    def _on_smart_actions(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return

        # If no analyses have been run yet, run Tornado + Criticality first
        if not self.tornado_result and not self.criticality_results and not self.mc_result:
            wx.MessageBox(
                "Run at least one analysis first (Tornado, Criticality, or MC) "
                "to provide data for the smart ranking.",
                "No Analysis Data", wx.OK | wx.ICON_INFORMATION)
            return

        active = list(filtered.keys())
        excluded = set(self.excluded_types)
        mh = self.mission_hours

        mc_imp = self.mc_result.parameter_importance if self.mc_result else None

        def work(cancel_event):
            return identify_smart_actions(
                filtered, mh,
                tornado_result=self.tornado_result,
                criticality_results=self.criticality_results or None,
                mc_importance=mc_imp,
                active_sheets=active,
                excluded_types=excluded,
                top_n=15,
            )

        def on_done(actions):
            self.smart_list.DeleteAllItems()
            for i, a in enumerate(actions, 1):
                refs = ", ".join(a.component_refs[:5])
                if len(a.component_refs) > 5:
                    refs += f" (+{len(a.component_refs)-5})"
                _add_row(self.smart_list, [
                    str(i), a.parameter, f"{a.score:.3f}",
                    f"{a.fit_improvement:.1f}", a.suggested_change,
                    refs, a.source,
                ])
            _autosize_columns(self.smart_list)
            if actions:
                top = actions[0]
                self.smart_detail.SetLabel(
                    f"Top action: {top.parameter} -- {top.reasoning}")
                self.smart_detail.Wrap(self.GetSize().Width - 40)
            self.status.SetLabel(
                f"Smart actions: {len(actions)} design levers identified")

        self._start_analysis(work, on_done,
                             label="Identifying smart design actions...",
                             buttons_to_disable=[self.btn_smart])

    # -- Custom scenario management --

    def _on_add_custom_scenario(self, event):
        name = self.custom_sc_name.GetValue().strip()
        sel = self.custom_sc_param.GetSelection()
        param = self.custom_sc_param.GetString(sel) if sel >= 0 else ""
        val_str = self.custom_sc_val.GetValue().strip()
        if not name or not param or not val_str:
            wx.MessageBox("Fill in name, parameter, and value.",
                          "Missing Input", wx.OK | wx.ICON_WARNING)
            return
        try:
            val = float(val_str)
        except ValueError:
            wx.MessageBox(f"'{val_str}' is not a valid number.",
                          "Input Error", wx.OK | wx.ICON_ERROR)
            return

        op = self.custom_sc_op.GetSelection()
        if op == 0:
            fn = lambda v, m=val: v * m
            desc = f"Multiply {param} by {val}"
        elif op == 1:
            fn = lambda v, a=val: v + a
            desc = f"Add {val} to {param}"
        else:
            fn = lambda v, s=val: s
            desc = f"Set {param} to {val}"

        self._custom_scenarios.append((name, desc, {param: fn}))
        self.custom_sc_label.SetLabel(f"{len(self._custom_scenarios)} custom scenario(s)")
        self.custom_sc_name.SetValue("")
        self.custom_sc_val.SetValue("")

    # -- What-If Scenarios (threaded) --

    def _on_run_whatif(self, event):
        filtered = self._filtered()
        if not filtered:
            wx.MessageBox("No active data.", "Error", wx.OK | wx.ICON_WARNING)
            return
        active = list(filtered.keys())
        excluded = set(self.excluded_types)
        mh = self.mission_hours
        custom = list(self._custom_scenarios) if self._custom_scenarios else None

        def work(cancel_event):
            return scenario_analysis(
                filtered, mh,
                active_sheets=active,
                excluded_types=excluded,
                custom_scenarios=custom,
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
            _autosize_columns(self.whatif_table)
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
            self.budget_sheet_list.DeleteAllItems()
            self.budget_list.DeleteAllItems()
            for sb in result.sheet_budgets:
                util_label = getattr(sb, "utilization", 0.0)
                util_text = f"{util_label*100:.0f}%" if math.isfinite(util_label) else "N/A"
                sheet_color = C.FAIL if sb.required_savings_fit > 0 else (C.WARN if sb.utilization > 0.9 else C.OK)
                _add_row(self.budget_sheet_list, [
                    sb.sheet_name,
                    f"{sb.actual_fit:.2f}",
                    f"{sb.budget_fit:.2f}",
                    f"{sb.required_savings_fit:.2f}",
                    util_text,
                    str(sb.n_over_budget),
                ], color=sheet_color)
            for cb in result.top_offenders:
                color = C.OK if cb.get("within_budget") else C.FAIL
                why = "Within budget" if cb.get("within_budget") else (
                    f"Needs {cb.get('required_savings_fit', 0):.2f} FIT reduction to hit allocation"
                )
                util_text = cb.get("utilization_label") or f"{cb.get('utilization_pct', 0):.0f}%"
                _add_row(self.budget_list, [
                    cb.get("reference", "?"), cb.get("component_type", "?"),
                    f"{cb.get('actual_fit', 0):.2f}", f"{cb.get('budget_fit', 0):.2f}",
                    f"{cb.get('required_savings_fit', 0):.2f}", util_text,
                    cb.get("status", "PASS"), why,
                ], color=color)
            _autosize_columns(self.budget_sheet_list)
            _autosize_columns(self.budget_list)
            self.budget_cards["target"].SetLabel(
                f"{result.effective_budget_fit:.1f} FIT ({result.design_margin_pct:.0f}% margin)"
            )
            self.budget_cards["actual"].SetLabel(f"{result.actual_fit:.1f} FIT")
            self.budget_cards["gap"].SetLabel(
                "Within budget" if result.fit_gap_to_close <= 0 else f"{result.fit_gap_to_close:.1f} FIT"
            )
            strategy_note = {
                "proportional": "recommended baseline for existing designs",
                "equal": "useful for simple equal-share target discussions",
                "complexity": "weights heavier sheets more strongly",
                "criticality": "tightens high-leverage components more aggressively",
            }.get(result.strategy, "allocation lens")
            icon = "\u2705" if result.system_within_budget else "\u274C"
            self.budget_info.SetLabel(
                f"{icon}  Target R={target_r} gives {result.target_fit:.1f} FIT before margin and "
                f"{result.effective_budget_fit:.1f} FIT after margin. "
                f"Actual = {result.actual_fit:.1f} FIT. "
                + ("No closure work is required. " if result.fit_gap_to_close <= 0 else
                   f"Close {result.fit_gap_to_close:.1f} FIT to meet the planning budget. ")
                + f"Strategy: {result.strategy.title()} ({strategy_note})."
            )
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
        target_fit = self.budget_result.effective_budget_fit if self.budget_result else (
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
                        "proposed": sw.get("description", ""),
                        "fit_saved": abs(sw.get("delta_fit", 0)),
                        "feasibility": "swap",
                    })
            merged.sort(key=lambda x: -x["fit_saved"])

            self.improve_list.DeleteAllItems()
            total_saved = 0.0
            for i, rec in enumerate(merged, 1):
                total_saved += rec["fit_saved"]
                _add_row(self.improve_list, [
                    str(i), rec["ref"], rec["type"],
                    rec["action"], rec["current"], rec["proposed"],
                    f"{rec['fit_saved']:.2f}", rec["feasibility"],
                ])
            _autosize_columns(self.improve_list)

            gap = (sf - tf) if sf > tf else 0
            shortlist = len(self.budget_result.top_offenders) if self.budget_result else 0
            self.improve_info.SetLabel(
                f"{len(merged)} recommendations. "
                f"Total potential savings: {total_saved:.1f} FIT"
                + (f" (covers {total_saved/gap*100:.0f}% of gap)" if gap > 0 else "")
                + (f" | budget shortlist: top {shortlist} offenders" if shortlist else "")
            )
            self.status.SetLabel(f"Improvements: {len(merged)} recommendations generated")

        self._start_analysis(work, on_done,
                             label="Generating improvement recommendations...",
                             buttons_to_disable=[self.btn_improve])

    # -- Single-param what-if (instant, no thread needed) --

    def _on_whatif_single(self, event):
        ref_sel = self.wi_ref.GetSelection()
        ref = self.wi_ref.GetString(ref_sel) if ref_sel >= 0 else ""
        param_sel = self.wi_param.GetSelection()
        param = self.wi_param.GetString(param_sel) if param_sel >= 0 else ""
        
        # Get value from either TextCtrl or Choice
        if isinstance(self.wi_val, wx.Choice):
            val_str = self.wi_val.GetString(self.wi_val.GetSelection()) if self.wi_val.GetSelection() >= 0 else ""
        else:
            val_str = self.wi_val.GetValue().strip() if isinstance(self.wi_val, wx.TextCtrl) else ""
        
        if not ref or not param or not val_str:
            wx.MessageBox("Fill in component reference, parameter name, and new value.",
                          "Missing Input", wx.OK | wx.ICON_WARNING)
            return
        if isinstance(self.wi_val, wx.TextCtrl):
            try:
                new_val = float(val_str)
            except ValueError:
                new_val = val_str
        else:
            new_val = val_str

        comp = self._find_component(ref)
        if comp is None:
            self.wi_result.SetValue(f"Component '{ref}' not found in active sheets.")
            return

        try:
            system_lambda = self._current_system_lambda()
            result = single_param_whatif(
                comp, param, new_val, system_lambda, self.mission_hours)
        except Exception as e:
            self.wi_result.SetValue(f"Error: {e}")
            return

        # Format output nicely
        old_val = result.get('old_value', '—')
        comp_fit_before = result.get('comp_fit_before', 0)
        comp_fit_after = result.get('comp_fit_after', 0)
        comp_delta = result.get('comp_delta_fit', 0)
        sys_fit_before = result.get('sys_fit_before', 0)
        sys_fit_after = result.get('sys_fit_after', 0)
        sys_delta = result.get('sys_delta_fit', 0)
        r_before = result.get('r_before', 0)
        r_after = result.get('r_after', 0)
        r_delta = result.get('delta_r', 0)

        comp_direction = "improves" if comp_delta < -0.01 else "worsens" if comp_delta > 0.01 else "is neutral"
        sys_direction = "improves" if sys_delta < -0.01 else "worsens" if sys_delta > 0.01 else "is neutral"
        mttf_after = (1.0 / (sys_fit_after * 1e-9)) if sys_fit_after > 0 else float("inf")

        output = (
            f"What-If Analysis: {ref}\n"
            f"Parameter: {param}\n"
            f"Change: {old_val} -> {new_val}\n"
            f"\n"
            f"Component impact:\n"
            f"  FIT: {comp_fit_before:.2f} -> {comp_fit_after:.2f} ({comp_delta:+.2f})\n"
            f"  Interpretation: component reliability {comp_direction}.\n"
            f"\n"
            f"System impact:\n"
            f"  FIT: {sys_fit_before:.2f} -> {sys_fit_after:.2f} ({sys_delta:+.2f})\n"
            f"  R(t): {r_before:.6f} -> {r_after:.6f} ({r_delta:+.6f})\n"
            f"  MTTF after change: {mttf_after:.0f} h\n"
            f"  Interpretation: system reliability {sys_direction}."
        )
        self.wi_result.SetValue(output)

    # -- History --

    def _on_save_snapshot(self, event):
        if not self.project_path:
            wx.MessageBox("No project path set.", "Error", wx.OK | wx.ICON_WARNING)
            return
        label = self.snap_label.GetValue().strip() or "v1"
        notes = self.snap_notes.GetValue().strip()
        try:
            snap = create_snapshot(self._filtered(), self._current_system_lambda(),
                                   self.mission_hours, label, notes)
            path = save_snapshot(snap, self.project_path)
            self.status.SetLabel(f"Snapshot '{label}' saved to {path}")
            self.snap_label.SetValue("")
            self.snap_notes.SetValue("")
            self._on_load_history(None)
        except Exception as e:
            wx.MessageBox(f"Failed to save snapshot: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def _on_load_history(self, event):
        if not self.project_path:
            self.history_summary.SetLabel("History is unavailable until the dialog has a project path.")
            self.history_detail.SetValue("")
            return
        try:
            snapshots = load_snapshots(self.project_path)
            self._history_snapshots = sorted(snapshots, key=lambda s: s.timestamp)
            self.history_list.DeleteAllItems()
            self._history_row_map = []
            for s in reversed(self._history_snapshots):
                _add_row(self.history_list, [
                    s.version_label, s.timestamp[:19],
                    f"{s.system_fit:.2f}", f"{s.system_reliability:.6f}",
                    str(s.n_components), s.notes,
                ])
                self._history_row_map.append(s.version_label)
            _autosize_columns(self.history_list)
            timeline = build_growth_timeline(self.project_path)
            if not self._history_snapshots:
                self.history_summary.SetLabel("No snapshots yet. Save a snapshot to create a revision history.")
                self.history_detail.SetValue("")
            elif len(self._history_snapshots) == 1:
                only = self._history_snapshots[0]
                self.history_summary.SetLabel(
                    f"1 snapshot stored. Latest: {only.version_label} at {only.system_fit:.2f} FIT."
                )
                self.history_detail.SetValue(self._format_snapshot_detail(only))
            else:
                latest = self._history_snapshots[-1]
                previous = self._history_snapshots[-2]
                comp = compare_revisions(previous, latest)
                self.history_summary.SetLabel(
                    f"{len(timeline.snapshots)} snapshots loaded. Latest comparison: "
                    f"{previous.version_label} -> {latest.version_label} ({comp.system_delta_fit:+.2f} FIT)."
                )
                self.history_detail.SetValue(self._format_revision_comparison(comp, timeline))
        except Exception as e:
            wx.MessageBox(f"Failed to load history: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def _format_snapshot_detail(self, snapshot):
        return (
            f"Snapshot: {snapshot.version_label}\n"
            f"Timestamp: {snapshot.timestamp[:19]}\n"
            f"System FIT: {snapshot.system_fit:.2f}\n"
            f"Reliability: {snapshot.system_reliability:.6f}\n"
            f"Mission: {snapshot.mission_hours:.0f} h\n"
            f"Components: {snapshot.n_components}\n"
            f"Sheets: {snapshot.n_sheets}\n"
            f"Notes: {snapshot.notes or '-'}"
        )

    def _format_revision_comparison(self, comparison, timeline=None):
        lines = [
            f"Revision Comparison: {comparison.from_version} -> {comparison.to_version}",
            f"System FIT: {comparison.system_fit_before:.2f} -> {comparison.system_fit_after:.2f} "
            f"({comparison.system_delta_fit:+.2f}, {comparison.system_delta_percent:+.1f}%)",
            f"Reliability: {comparison.reliability_before:.6f} -> {comparison.reliability_after:.6f} "
            f"({comparison.reliability_improvement:+.6f})",
            f"Changed components: improved {comparison.components_improved}, "
            f"degraded {comparison.components_degraded}, added {comparison.components_added}, "
            f"removed {comparison.components_removed}, unchanged {comparison.components_unchanged}",
        ]
        if timeline and timeline.snapshots:
            first = timeline.snapshots[0]
            last = timeline.snapshots[-1]
            lines.append(
                f"Timeline: {len(timeline.snapshots)} snapshots from {first.version_label} to {last.version_label}"
            )
        if comparison.top_improvements:
            lines.append("")
            lines.append("Top improvements:")
            for change in comparison.top_improvements[:3]:
                lines.append(f"  {change.reference}: {change.delta_fit:+.2f} FIT ({change.change_type})")
        if comparison.top_degradations:
            lines.append("")
            lines.append("Top degradations:")
            for change in comparison.top_degradations[:3]:
                lines.append(f"  {change.reference}: {change.delta_fit:+.2f} FIT ({change.change_type})")
        return "\n".join(lines)

    def _on_history_selected(self, event):
        if not getattr(self, "_history_snapshots", None):
            return
        idx = event.GetIndex()
        if idx < 0 or idx >= len(self._history_row_map):
            return
        version = self._history_row_map[idx]
        for snapshot in self._history_snapshots:
            if snapshot.version_label == version:
                self.history_detail.SetValue(self._format_snapshot_detail(snapshot))
                break

    def _on_compare_latest_history(self, event):
        if len(getattr(self, "_history_snapshots", [])) < 2:
            wx.MessageBox("Need at least two snapshots to compare revisions.",
                          "History", wx.OK | wx.ICON_INFORMATION)
            return
        timeline = build_growth_timeline(self.project_path)
        before = self._history_snapshots[-2]
        after = self._history_snapshots[-1]
        comparison = compare_revisions(before, after)
        self.history_detail.SetValue(self._format_revision_comparison(comparison, timeline))
        self.history_summary.SetLabel(
            f"Comparing latest revisions: {before.version_label} -> {after.version_label}."
        )

    def _on_delete_selected_history(self, event):
        idx = self.history_list.GetFirstSelected()
        if idx < 0 or not getattr(self, "_history_row_map", None):
            wx.MessageBox("Select a snapshot to delete.", "History", wx.OK | wx.ICON_INFORMATION)
            return
        version = self._history_row_map[idx]
        if wx.MessageBox(
            f"Delete snapshot '{version}'?",
            "Confirm Delete",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        ) != wx.YES:
            return
        if delete_snapshot(self.project_path, version):
            self.status.SetLabel(f"Deleted snapshot '{version}'")
            self._on_load_history(None)
        else:
            wx.MessageBox(f"Could not delete snapshot '{version}'.", "History", wx.OK | wx.ICON_ERROR)

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
        _style_button(self.btn_html, "primary")
        self.btn_html.Bind(wx.EVT_BUTTON, self._on_gen_html)
        rs.Add(self.btn_html, 0, wx.ALL, 10)
        self.btn_pdf = wx.Button(rp, label="Generate PDF Report")
        _style_button(self.btn_pdf, "success")
        self.btn_pdf.SetToolTip("Requires optional dependency support: install weasyprint or reportlab.")
        self.btn_pdf.Bind(wx.EVT_BUTTON, self._on_gen_pdf)
        rs.Add(self.btn_pdf, 0, wx.ALL, 10)
        btn_preview = wx.Button(rp, label="Refresh Preview")
        btn_preview.Bind(wx.EVT_BUTTON, self._on_refresh_preview)
        rs.Add(btn_preview, 0, wx.ALL, 10)
        rs.Add(wx.StaticText(rp,
            label="Reports include all analyses run in this session. "
                  "Run analyses first, then generate."),
            0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        rp.SetSizer(rs)
        main.Add(rp, 0, wx.EXPAND | wx.ALL, 6)

        self.report_status = wx.StaticText(panel, label="")
        self.report_status.SetForegroundColour(C.TXT_M)
        self.report_status.SetFont(ui_font(panel, role="body"))
        main.Add(self.report_status, 0, wx.ALL, 10)

        # Report preview
        preview_card = wx.Panel(panel)
        preview_card.SetBackgroundColour(C.WHITE)
        pcs = wx.BoxSizer(wx.VERTICAL)
        pl = wx.StaticText(preview_card, label="Report Preview")
        pl.SetFont(ui_font(preview_card, role="section", weight=wx.FONTWEIGHT_BOLD))
        pcs.Add(pl, 0, wx.ALL, 10)
        self.report_preview = wx.TextCtrl(
            preview_card, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.report_preview.SetFont(ui_font(preview_card, role="mono"))
        style_text_like(self.report_preview, read_only=True)
        pcs.Add(self.report_preview, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        preview_card.SetSizer(pcs)
        main.Add(preview_card, 1, wx.EXPAND | wx.ALL, 6)

        panel.SetSizer(main)
        self._refresh_report_preview()
        return panel

    def _on_refresh_preview(self, event):
        self._refresh_report_preview()

    def _refresh_report_preview(self):
        """Build a text summary of what will be in the report."""
        r = self._sys_r()
        fit = self._sys_fit()
        mttf_yr = (1.0 / self.system_lambda / 8760) if self.system_lambda > 0 else float("inf")
        yrs = self.mission_hours / 8760
        comps = self._all_components()

        lines = [
            "=" * 50,
            "  RELIABILITY ANALYSIS REPORT PREVIEW",
            "=" * 50,
            "",
            "  Includes a plain-language introduction explaining what the tool computes",
            "  and how to interpret the report before the technical sections begin.",
            "",
            f"  System R(t):     {r:.6f}",
            f"  System FIT:      {fit:.2f}",
            f"  MTTF:            {mttf_yr:.1f} years",
            f"  Mission:         {yrs:.1f} years",
            f"  Components:      {len(comps)}",
            f"  Sheets:          {len(self._active_data)}",
            "",
            "  SECTIONS INCLUDED:",
        ]

        lines.append(f"    [x] System Summary")
        lines.append(f"    [x] FIT Contributions ({len(comps)} components)")
        lines.append(f"    [x] Sheet Breakdown ({len(self._active_data)} sheets)")

        if self.mc_result:
            lines.append(f"    [x] Monte Carlo ({self.mc_result.n_simulations:,} samples)")
            if self.mc_result.parameter_importance:
                lines.append(f"    [x] Parameter Importance (SRRC)")
        else:
            lines.append(f"    [ ] Monte Carlo (not run)")

        if self.tornado_result:
            lines.append(f"    [x] Tornado Sensitivity ({len(self.tornado_result.entries)} params)")
        else:
            lines.append(f"    [ ] Tornado (not run)")

        if self.scenario_result:
            lines.append(f"    [x] Design Scenarios ({len(self.scenario_result.scenarios)} scenarios)")
        else:
            lines.append(f"    [ ] Design Scenarios (not run)")

        if self.criticality_results:
            lines.append(f"    [x] Component Criticality ({len(self.criticality_results)} components)")
        else:
            lines.append(f"    [ ] Component Criticality (not run)")

        if self.budget_result:
            lines.append(f"    [x] Target Closure Planning")
        else:
            lines.append(f"    [ ] Target Closure Planning (not run)")

        if self.derating_result:
            lines.append(f"    [x] Derating Guidance")
        else:
            lines.append(f"    [ ] Derating (not run)")

        if self.swap_results:
            lines.append(f"    [x] Component Swap Analysis")
        else:
            lines.append(f"    [ ] Swap Analysis (not run)")

        lines.extend(["", "  Run more analyses to enrich the report."])
        self.report_preview.SetValue("\n".join(lines))

    def _build_report_data(self):
        filtered = self._filtered()
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / 8760
        all_components = [
            comp for sheet in filtered.values() for comp in sheet.get("components", [])
        ]
        classification_summary = {
            "total": len(all_components),
            "review_required": sum(1 for c in all_components if c.get("classification_review_required")),
            "high_confidence": sum(1 for c in all_components if c.get("classification_confidence") == "high"),
            "explicit": sum(1 for c in all_components if c.get("classification_source") == "explicit field"),
            "manual": sum(1 for c in all_components if c.get("classification_source") == "manual"),
        }

        mc_dict = None
        if self.mc_result is not None:
            samples = getattr(self.mc_result, "reliability_samples", None)
            if samples is not None and len(samples) > 0:
                mc_dict = self.mc_result.to_dict()
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

        swap_dict = None
        if self.swap_results:
            swap_dict = {"recommendations": self.swap_results}

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
            budget=budget_dict,
            derating=derating_dict,
            swap_analysis=swap_dict,
            classification_summary=classification_summary,
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
            path = self._ensure_report_extension(dlg.GetPath(), ".html")
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
            path = self._ensure_report_extension(dlg.GetPath(), ".pdf")
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

    @staticmethod
    def _ensure_report_extension(path: str, suffix: str) -> str:
        if not path:
            return path
        lower_path = path.lower()
        lower_suffix = suffix.lower()
        return path if lower_path.endswith(lower_suffix) else path + suffix

    # =================================================================
    # What-If dropdown helpers
    # =================================================================

    def _collect_all_param_names(self):
        """Collect all numeric parameter names from active components."""
        names = set()
        for c in self._all_components():
            for k, v in c.get("params", {}).items():
                if k.startswith("_") or k in ORBIT_PARAMS:
                    continue
                try:
                    float(v)
                    names.add(k)
                except (TypeError, ValueError):
                    pass
        return sorted(names)

    def _on_wi_ref_changed(self, event):
        """Update parameter dropdown when component selection changes."""
        ref_sel = self.wi_ref.GetSelection()
        if ref_sel < 0:
            return
        ref = self.wi_ref.GetString(ref_sel)
        comp = self._find_component(ref)
        if not comp:
            return
        params = []
        for k, v in comp.get("params", {}).items():
            if k.startswith("_") or k in ORBIT_PARAMS:
                continue
            params.append(k)
        self.wi_param.Clear()
        for p in sorted(params):
            self.wi_param.Append(p)
        if params:
            self.wi_param.SetSelection(0)
            self._on_wi_param_changed(None)

    def _on_wi_param_changed(self, event):
        """Update value control when parameter selection changes."""
        ref_sel = self.wi_ref.GetSelection()
        param_sel = self.wi_param.GetSelection()
        if ref_sel < 0 or param_sel < 0:
            return
        ref = self.wi_ref.GetString(ref_sel)
        param = self.wi_param.GetString(param_sel)
        
        comp = self._find_component(ref)
        if not comp:
            return
        
        # Get current value
        current_val = comp.get("params", {}).get(param)
        comp_type = comp.get("class", "Unknown")
        
        # Get available options for this parameter
        try:
            swap_opts = _get_swap_options(comp_type, comp.get("params", {}))
        except Exception:
            swap_opts = {}
        
        # Remove old value control from dedicated holder
        if isinstance(self.wi_val, (wx.TextCtrl, wx.Choice)):
            self._wi_val_sizer.Detach(self.wi_val)
            self.wi_val.Destroy()
        
        # Create new value control (dropdown or textctrl)
        if param in swap_opts and swap_opts[param]:
            # Create dropdown for categorical parameters
            self.wi_val = wx.Choice(self._wi_val_panel, choices=swap_opts[param], size=dip_size(self._wi_val_panel, 144, -1))
            if current_val and str(current_val) in swap_opts[param]:
                self.wi_val.SetStringSelection(str(current_val))
            elif swap_opts[param]:
                self.wi_val.SetSelection(0)
        else:
            # Create textctrl for numeric parameters
            self.wi_val = wx.TextCtrl(self._wi_val_panel, size=dip_size(self._wi_val_panel, 144, -1))
            if current_val is not None:
                self.wi_val.SetValue(str(current_val))
        
        self._wi_val_sizer.Add(self.wi_val, 0, wx.ALL, 0)
        self._wi_val_panel.Layout()
        unit_hint = "categorical choice" if isinstance(self.wi_val, wx.Choice) else "numeric or literal value"
        self.wi_hint.SetLabel(
            f"{ref}.{param} current value: {current_val!s}. Enter a {unit_hint} and evaluate the local impact."
        )

    # =================================================================
    # Type filter handler
    # =================================================================

    def _on_type_filter(self, event):
        self.excluded_types = set()
        for tn, cb in self._type_cbs.items():
            if not cb.GetValue():
                self.excluded_types.add(tn)
        self._refresh_overview()
