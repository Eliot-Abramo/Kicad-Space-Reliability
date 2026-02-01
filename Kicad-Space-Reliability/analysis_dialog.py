"""
Analysis Dialog - Industrial-Grade Reliability Analysis Suite
=============================================================
Professional visualization of Monte Carlo uncertainty and Sobol sensitivity analysis.
Clean, legible design with publication-quality graphics.
"""

import wx
import wx.lib.scrolledpanel as scrolled
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .monte_carlo import (
    MonteCarloResult,
    quick_monte_carlo,
    monte_carlo_sheet,
    monte_carlo_blocks,
    SheetMCResult,
    ComponentMCInput,
    MonteCarloConfig,
    MonteCarloEngine,
    monte_carlo_components_optimized,
)
from .mc_calibration_dialog import MonteCarloCalibrationDialog, MCProgressDialog
import threading
from .sensitivity_analysis import (
    SobolResult,
    SobolAnalyzer,
    ImportanceAnalyzer,
    ImportanceMeasures,
    SensitivityAnalyzer,
    quick_sensitivity,
)
from .reliability_math import reliability_from_lambda, calculate_component_lambda
from .report_generator import (
    ReportGenerator,
    ReportData,
    ComponentReportData,
    BlockReportData,
)


# =============================================================================
# Professional Color Scheme
# =============================================================================


class Colors:
    """Clean industrial color palette."""

    # Backgrounds
    BG_LIGHT = wx.Colour(252, 252, 253)
    BG_WHITE = wx.Colour(255, 255, 255)
    BG_HEADER = wx.Colour(37, 99, 235)  # Blue header

    # Text
    TEXT_DARK = wx.Colour(17, 24, 39)
    TEXT_MEDIUM = wx.Colour(75, 85, 99)
    TEXT_LIGHT = wx.Colour(156, 163, 175)
    TEXT_WHITE = wx.Colour(255, 255, 255)

    # Accents
    PRIMARY = wx.Colour(37, 99, 235)  # Blue
    SUCCESS = wx.Colour(34, 197, 94)  # Green
    WARNING = wx.Colour(245, 158, 11)  # Amber
    DANGER = wx.Colour(239, 68, 68)  # Red

    # Chart colors (colorblind-safe)
    CHART = [
        wx.Colour(59, 130, 246),  # Blue
        wx.Colour(16, 185, 129),  # Green
        wx.Colour(245, 158, 11),  # Amber
        wx.Colour(239, 68, 68),  # Red
        wx.Colour(139, 92, 246),  # Purple
        wx.Colour(236, 72, 153),  # Pink
        wx.Colour(20, 184, 166),  # Teal
        wx.Colour(249, 115, 22),  # Orange
    ]

    # Borders
    BORDER = wx.Colour(229, 231, 235)
    GRID = wx.Colour(243, 244, 246)


# =============================================================================
# Professional Chart Components
# =============================================================================


class HistogramPanel(wx.Panel):
    """Clean histogram visualization."""

    def __init__(self, parent, title="Distribution"):
        super().__init__(parent, size=(500, 320))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.samples = None
        self.mean = None
        self.p5 = None
        self.p95 = None

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, samples: np.ndarray, mean: float, p5: float, p95: float):
        self.samples = samples
        self.mean = mean
        self.p5 = p5
        self.p95 = p95
        self.Refresh()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()

        # Background
        dc.SetBrush(wx.Brush(Colors.BG_WHITE))
        dc.SetPen(wx.Pen(Colors.BORDER, 1))
        dc.DrawRectangle(0, 0, w, h)

        # Title
        dc.SetFont(
            wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 16, 12)

        if self.samples is None or len(self.samples) == 0:
            dc.SetFont(
                wx.Font(
                    10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
                )
            )
            dc.SetTextForeground(Colors.TEXT_LIGHT)
            dc.DrawText("Run analysis to see distribution", w // 2 - 100, h // 2)
            return

        # Chart area
        margin_l, margin_r, margin_t, margin_b = 60, 30, 45, 55
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b

        if chart_w <= 0 or chart_h <= 0:
            return

        # Histogram data
        n_bins = 35
        hist, edges = np.histogram(self.samples, bins=n_bins)
        max_count = max(hist) if max(hist) > 0 else 1
        bar_width = max(1, chart_w // n_bins - 1)

        # Draw grid
        dc.SetPen(wx.Pen(Colors.GRID, 1))
        for i in range(5):
            y = margin_t + chart_h * i // 4
            dc.DrawLine(margin_l, y, w - margin_r, y)

        # Draw bars
        dc.SetBrush(wx.Brush(Colors.PRIMARY))
        dc.SetPen(wx.Pen(Colors.PRIMARY.ChangeLightness(85), 1))

        for i, count in enumerate(hist):
            if count > 0:
                x = margin_l + i * chart_w // n_bins
                bar_h = int((count / max_count) * chart_h)
                dc.DrawRectangle(
                    int(x), margin_t + chart_h - bar_h, int(bar_width), bar_h
                )

        # Draw reference lines
        min_val, max_val = edges[0], edges[-1]
        val_range = max_val - min_val
        if val_range <= 0:
            val_range = 1

        def val_to_x(v):
            return margin_l + (v - min_val) / val_range * chart_w

        # Mean line (red)
        if self.mean is not None:
            x_mean = val_to_x(self.mean)
            dc.SetPen(wx.Pen(Colors.DANGER, 2))
            dc.DrawLine(int(x_mean), margin_t, int(x_mean), margin_t + chart_h)

        # Percentile lines (dashed amber)
        dc.SetPen(wx.Pen(Colors.WARNING, 2, wx.PENSTYLE_SHORT_DASH))
        if self.p5 is not None:
            x_p5 = val_to_x(self.p5)
            dc.DrawLine(int(x_p5), margin_t, int(x_p5), margin_t + chart_h)
        if self.p95 is not None:
            x_p95 = val_to_x(self.p95)
            dc.DrawLine(int(x_p95), margin_t, int(x_p95), margin_t + chart_h)

        # Axes
        dc.SetPen(wx.Pen(Colors.TEXT_MEDIUM, 1))
        dc.DrawLine(margin_l, margin_t + chart_h, w - margin_r, margin_t + chart_h)
        dc.DrawLine(margin_l, margin_t, margin_l, margin_t + chart_h)

        # X-axis labels
        dc.SetFont(
            wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        dc.SetTextForeground(Colors.TEXT_MEDIUM)
        for i in range(5):
            val = min_val + val_range * i / 4
            x = margin_l + chart_w * i // 4
            label = f"{val:.4f}"
            tw, _ = dc.GetTextExtent(label)
            dc.DrawText(label, x - tw // 2, margin_t + chart_h + 8)

        dc.DrawText("Reliability R(t)", margin_l + chart_w // 2 - 50, h - 18)

        # Legend
        legend_x = w - margin_r - 130
        legend_y = margin_t + 5

        dc.SetFont(
            wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )

        dc.SetPen(wx.Pen(Colors.DANGER, 2))
        dc.DrawLine(legend_x, legend_y + 6, legend_x + 20, legend_y + 6)
        dc.DrawText("Mean", legend_x + 25, legend_y)

        dc.SetPen(wx.Pen(Colors.WARNING, 2, wx.PENSTYLE_SHORT_DASH))
        dc.DrawLine(legend_x, legend_y + 20, legend_x + 20, legend_y + 20)
        dc.DrawText("5th/95th %ile", legend_x + 25, legend_y + 14)


class HorizontalBarPanel(wx.Panel):
    """Clean horizontal bar chart for rankings/contributions."""

    def __init__(self, parent, title="Chart"):
        super().__init__(parent, size=(500, 400))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.data = []  # List of (name, value, color_idx)
        self.max_value = 1.0
        self.x_label = "Value"

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(
        self,
        data: List[Tuple[str, float]],
        max_value: float = None,
        x_label: str = "Value",
    ):
        """Set data as list of (name, value) tuples."""
        self.data = [
            (name, val, i % len(Colors.CHART)) for i, (name, val) in enumerate(data)
        ]
        self.max_value = (
            max_value
            if max_value
            else (max(d[1] for d in self.data) if self.data else 1.0)
        )
        self.max_value = max(self.max_value, 0.001)
        self.x_label = x_label
        self.Refresh()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()

        # Background
        dc.SetBrush(wx.Brush(Colors.BG_WHITE))
        dc.SetPen(wx.Pen(Colors.BORDER, 1))
        dc.DrawRectangle(0, 0, w, h)

        # Title
        dc.SetFont(
            wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 16, 12)

        if not self.data:
            dc.SetFont(
                wx.Font(
                    10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
                )
            )
            dc.SetTextForeground(Colors.TEXT_LIGHT)
            dc.DrawText("No data available", w // 2 - 50, h // 2)
            return

        # Chart area - larger left margin for labels
        margin_l, margin_r, margin_t, margin_b = 140, 30, 45, 45
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b

        if chart_w <= 0 or chart_h <= 0:
            return

        n_bars = min(len(self.data), 15)  # Limit to 15 bars
        bar_height = min(22, max(12, (chart_h - 10) // n_bars))
        spacing = max(2, (chart_h - n_bars * bar_height) // (n_bars + 1))

        # Draw grid
        dc.SetPen(wx.Pen(Colors.GRID, 1))
        for i in range(5):
            x = margin_l + chart_w * i // 4
            dc.DrawLine(x, margin_t, x, margin_t + chart_h)

        # Draw bars
        dc.SetFont(
            wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )

        for i, (name, value, color_idx) in enumerate(self.data[:n_bars]):
            y = margin_t + spacing + i * (bar_height + spacing)

            # Bar
            bar_w = int((value / self.max_value) * chart_w)
            bar_w = max(bar_w, 2)  # Minimum visible width

            color = Colors.CHART[color_idx]
            dc.SetBrush(wx.Brush(color))
            dc.SetPen(wx.Pen(color.ChangeLightness(85), 1))
            dc.DrawRoundedRectangle(margin_l, y, bar_w, bar_height, 3)

            # Label (truncate if needed)
            dc.SetTextForeground(Colors.TEXT_DARK)
            display_name = name[:18] + "..." if len(name) > 18 else name
            tw, th = dc.GetTextExtent(display_name)
            dc.DrawText(display_name, margin_l - tw - 8, y + (bar_height - th) // 2)

            # Value
            val_text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            vw, vh = dc.GetTextExtent(val_text)
            if bar_w > vw + 10:
                dc.SetTextForeground(Colors.TEXT_WHITE)
                dc.DrawText(val_text, margin_l + 6, y + (bar_height - vh) // 2)
            else:
                dc.SetTextForeground(Colors.TEXT_DARK)
                dc.DrawText(val_text, margin_l + bar_w + 6, y + (bar_height - vh) // 2)

        # X-axis
        dc.SetPen(wx.Pen(Colors.TEXT_MEDIUM, 1))
        dc.DrawLine(margin_l, margin_t + chart_h, w - margin_r, margin_t + chart_h)

        # X-axis labels
        dc.SetFont(
            wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        dc.SetTextForeground(Colors.TEXT_MEDIUM)
        for i in range(5):
            val = self.max_value * i / 4
            x = margin_l + chart_w * i // 4
            label = f"{val:.2f}"
            tw, _ = dc.GetTextExtent(label)
            dc.DrawText(label, x - tw // 2, margin_t + chart_h + 6)

        dc.DrawText(self.x_label, margin_l + chart_w // 2 - 40, h - 15)


class ConvergencePanel(wx.Panel):
    """Convergence plot showing running mean."""

    def __init__(self, parent, title="Convergence"):
        super().__init__(parent, size=(400, 200))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.samples = None

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())

    def set_data(self, samples: np.ndarray):
        self.samples = samples
        self.Refresh()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        w, h = self.GetSize()

        dc.SetBrush(wx.Brush(Colors.BG_WHITE))
        dc.SetPen(wx.Pen(Colors.BORDER, 1))
        dc.DrawRectangle(0, 0, w, h)

        dc.SetFont(
            wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 12, 8)

        if self.samples is None or len(self.samples) == 0:
            return

        margin_l, margin_r, margin_t, margin_b = 55, 20, 35, 35
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b

        if chart_w <= 0 or chart_h <= 0:
            return

        # Calculate running mean
        cumsum = np.cumsum(self.samples)
        running_mean = cumsum / np.arange(1, len(self.samples) + 1)

        # Sample at intervals
        step = max(1, len(running_mean) // 100)
        points = [(i, running_mean[i]) for i in range(0, len(running_mean), step)]
        if len(running_mean) - 1 not in [p[0] for p in points]:
            points.append((len(running_mean) - 1, running_mean[-1]))

        # Value range
        vals = [p[1] for p in points]
        v_min, v_max = min(vals), max(vals)
        v_range = v_max - v_min
        if v_range < 1e-9:
            v_range = abs(v_max) * 0.1 if v_max != 0 else 0.01
        v_min -= v_range * 0.1
        v_max += v_range * 0.1
        v_range = v_max - v_min

        n_max = len(self.samples)

        # Grid
        dc.SetPen(wx.Pen(Colors.GRID, 1))
        for i in range(5):
            y = margin_t + chart_h * i // 4
            dc.DrawLine(margin_l, y, w - margin_r, y)

        # Line
        dc.SetPen(wx.Pen(Colors.PRIMARY, 2))
        prev_pt = None
        for n, v in points:
            x = margin_l + (n / n_max) * chart_w
            y = margin_t + chart_h - ((v - v_min) / v_range) * chart_h
            if prev_pt:
                dc.DrawLine(int(prev_pt[0]), int(prev_pt[1]), int(x), int(y))
            prev_pt = (x, y)

        # Final value line
        final_y = margin_t + chart_h - ((running_mean[-1] - v_min) / v_range) * chart_h
        dc.SetPen(wx.Pen(Colors.SUCCESS, 1, wx.PENSTYLE_SHORT_DASH))
        dc.DrawLine(margin_l, int(final_y), w - margin_r, int(final_y))

        # Axes
        dc.SetPen(wx.Pen(Colors.TEXT_MEDIUM, 1))
        dc.DrawLine(margin_l, margin_t + chart_h, w - margin_r, margin_t + chart_h)

        # Labels
        dc.SetFont(
            wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        dc.SetTextForeground(Colors.TEXT_MEDIUM)
        dc.DrawText("Simulations", margin_l + chart_w // 2 - 30, h - 12)


class StatsCard(wx.Panel):
    """Clean statistics display card."""

    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_WHITE)
        self.stats = {}

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(self, label="Summary Statistics")
        title.SetFont(
            wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
        title.SetForegroundColour(Colors.TEXT_DARK)
        self.sizer.Add(title, 0, wx.ALL, 12)

        self.content_sizer = wx.FlexGridSizer(0, 2, 6, 16)
        self.content_sizer.AddGrowableCol(1)
        self.sizer.Add(
            self.content_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12
        )

        self.SetSizer(self.sizer)

    def set_stats(self, stats: Dict[str, Any]):
        self.content_sizer.Clear(True)

        display_order = [
            ("Mean (mu)", "mean", ".6f"),
            ("Std Dev (sigma)", "std", ".6f"),
            ("Median", "median", ".6f"),
            ("5th Percentile", "p5", ".6f"),
            ("95th Percentile", "p95", ".6f"),
            ("90% CI Width", "ci_width", ".6f"),
            ("CV", "cv", ".2%"),
            ("Simulations", "n_sims", ",d"),
            ("Converged", "converged", "bool"),
        ]

        for label, key, fmt in display_order:
            if key not in stats:
                continue

            lbl = wx.StaticText(self, label=label + ":")
            lbl.SetFont(
                wx.Font(
                    9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
                )
            )
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

            val_lbl = wx.StaticText(self, label=text)
            val_lbl.SetFont(
                wx.Font(
                    10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
                )
            )
            val_lbl.SetForegroundColour(color)
            self.content_sizer.Add(val_lbl, 0, wx.ALIGN_CENTER_VERTICAL)

        self.Layout()


# =============================================================================
# Main Analysis Dialog
# =============================================================================


class AnalysisDialog(wx.Dialog):
    """
    Professional reliability analysis dialog.
    Clean design, proper charts, comprehensive reporting.
    """

    def __init__(
        self,
        parent,
        system_lambda: float,
        mission_hours: float,
        sheet_data: Dict[str, Dict] = None,
        block_structure: Dict = None,
        title: str = "Reliability Analysis Suite",
    ):

        display = wx.Display(0)
        rect = display.GetClientArea()
        w = min(1350, int(rect.Width * 0.85))
        h = min(900, int(rect.Height * 0.88))

        super().__init__(
            parent,
            title=title,
            size=(w, h),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX,
        )

        self.SetMinSize((1000, 700))
        self.SetBackgroundColour(Colors.BG_LIGHT)

        self.system_lambda = system_lambda
        self.mission_hours = mission_hours
        self.sheet_data = sheet_data or {}
        self.block_structure = block_structure or {}

        self.mc_result: Optional[MonteCarloResult] = None
        self.sobol_result: Optional[SobolResult] = None
        self.sheet_mc_results: Dict[str, SheetMCResult] = (
            {}
        )  # Per-sheet Monte Carlo results

        # Monte Carlo configuration and engine
        self.mc_config = MonteCarloConfig()
        self.mc_engine = MonteCarloEngine()

        self._create_ui()
        self.Centre()

    def _create_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header
        header = self._create_header()
        main_sizer.Add(header, 0, wx.EXPAND)

        # Notebook
        self.notebook = wx.Notebook(self)
        self.notebook.SetBackgroundColour(Colors.BG_LIGHT)

        self.notebook.AddPage(self._create_mc_tab(), "Monte Carlo")
        self.notebook.AddPage(self._create_sensitivity_tab(), "Sensitivity")
        self.notebook.AddPage(self._create_contributions_tab(), "Contributions")
        self.notebook.AddPage(self._create_report_tab(), "Full Report")

        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 12)

        # Footer
        footer = self._create_footer()
        main_sizer.Add(footer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        self.SetSizer(main_sizer)

    def _create_header(self) -> wx.Panel:
        panel = wx.Panel(self)
        panel.SetBackgroundColour(Colors.BG_HEADER)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        title = wx.StaticText(panel, label="Reliability Analysis Suite")
        title.SetFont(
            wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
        title.SetForegroundColour(Colors.TEXT_WHITE)
        sizer.Add(title, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 14)

        # System info
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        info = wx.StaticText(
            panel,
            label=f"Lambda = {self.system_lambda*1e9:.2f} FIT  |  R = {r:.6f}  |  {years:.1f} years",
        )
        info.SetFont(
            wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        info.SetForegroundColour(wx.Colour(191, 219, 254))
        sizer.Add(info, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 14)

        panel.SetSizer(sizer)
        return panel

    def _create_mc_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        # Controls
        ctrl_panel = wx.Panel(panel)
        ctrl_panel.SetBackgroundColour(Colors.BG_WHITE)
        ctrl = wx.BoxSizer(wx.HORIZONTAL)

        ctrl.Add(
            wx.StaticText(ctrl_panel, label="Simulations:"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
            12,
        )
        self.mc_n = wx.SpinCtrl(
            ctrl_panel, min=1000, max=100000, initial=5000, size=(100, -1)
        )
        ctrl.Add(self.mc_n, 0, wx.ALL, 8)

        ctrl.Add(
            wx.StaticText(ctrl_panel, label="Uncertainty (%):"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
            12,
        )
        self.mc_unc = wx.SpinCtrlDouble(
            ctrl_panel, min=5, max=100, initial=25, inc=5, size=(80, -1)
        )
        ctrl.Add(self.mc_unc, 0, wx.ALL, 8)

        self.btn_mc = wx.Button(ctrl_panel, label="Run System MC")
        self.btn_mc.SetBackgroundColour(Colors.PRIMARY)
        self.btn_mc.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_mc.Bind(wx.EVT_BUTTON, self._on_run_mc)
        self.btn_mc.SetToolTip(
            "Run Monte Carlo with component-level uncertainty propagation"
        )
        ctrl.Add(self.btn_mc, 0, wx.ALL, 8)

        # Per-sheet MC button
        self.btn_mc_sheets = wx.Button(ctrl_panel, label="Run Per-Sheet MC")
        self.btn_mc_sheets.SetBackgroundColour(Colors.WARNING)
        self.btn_mc_sheets.SetForegroundColour(Colors.TEXT_DARK)
        self.btn_mc_sheets.Bind(wx.EVT_BUTTON, self._on_run_mc_sheets)
        self.btn_mc_sheets.SetToolTip(
            "Run Monte Carlo for each sheet/block - adds graphs to report"
        )
        ctrl.Add(self.btn_mc_sheets, 0, wx.ALL, 8)

        # Calibration button
        self.btn_calibrate = wx.Button(ctrl_panel, label="⚙ Calibrate...")
        self.btn_calibrate.SetToolTip(
            "Configure uncertainty distributions and simulation parameters"
        )
        self.btn_calibrate.Bind(wx.EVT_BUTTON, self._on_calibrate_mc)
        ctrl.Add(self.btn_calibrate, 0, wx.ALL, 8)

        ctrl_panel.SetSizer(ctrl)
        main.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 8)

        # Notification about full report
        note_panel = wx.Panel(panel)
        note_panel.SetBackgroundColour(wx.Colour(255, 251, 235))  # Warm yellow
        note_sizer = wx.BoxSizer(wx.HORIZONTAL)
        note_icon = wx.StaticText(note_panel, label="ℹ️")
        note_icon.SetFont(
            wx.Font(
                12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
            )
        )
        note_sizer.Add(note_icon, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        note_text = wx.StaticText(
            note_panel,
            label="Tip: Comprehensive analysis data, component details, and per-sheet graphs are available in the Full Report tab.",
        )
        note_text.SetFont(
            wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        note_text.SetForegroundColour(Colors.TEXT_DARK)
        note_sizer.Add(note_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        note_panel.SetSizer(note_sizer)
        main.Add(note_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Charts row
        charts = wx.BoxSizer(wx.HORIZONTAL)

        # Left: histogram + convergence
        left = wx.BoxSizer(wx.VERTICAL)
        self.histogram = HistogramPanel(panel, "Reliability Distribution")
        left.Add(self.histogram, 2, wx.EXPAND)
        self.convergence = ConvergencePanel(panel, "Mean Convergence")
        left.Add(self.convergence, 1, wx.EXPAND | wx.TOP, 8)
        charts.Add(left, 2, wx.EXPAND | wx.RIGHT, 8)

        # Right: stats
        self.stats_card = StatsCard(panel)
        charts.Add(self.stats_card, 1, wx.EXPAND)

        main.Add(charts, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        panel.SetSizer(main)
        return panel

    def _create_sensitivity_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        # Controls
        ctrl_panel = wx.Panel(panel)
        ctrl_panel.SetBackgroundColour(Colors.BG_WHITE)
        ctrl = wx.BoxSizer(wx.HORIZONTAL)

        ctrl.Add(
            wx.StaticText(ctrl_panel, label="Samples (N):"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
            12,
        )
        self.sobol_n = wx.SpinCtrl(
            ctrl_panel, min=256, max=4096, initial=1024, size=(100, -1)
        )
        ctrl.Add(self.sobol_n, 0, wx.ALL, 8)

        self.btn_sobol = wx.Button(ctrl_panel, label="Run Sensitivity Analysis")
        self.btn_sobol.SetBackgroundColour(Colors.SUCCESS)
        self.btn_sobol.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_sobol.Bind(wx.EVT_BUTTON, self._on_run_sobol)
        ctrl.Add(self.btn_sobol, 0, wx.ALL, 8)

        ctrl_panel.SetSizer(ctrl)
        main.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 8)

        # Charts
        charts = wx.BoxSizer(wx.HORIZONTAL)

        self.sobol_first_chart = HorizontalBarPanel(panel, "First-Order Indices (S1)")
        charts.Add(self.sobol_first_chart, 1, wx.EXPAND | wx.RIGHT, 8)

        self.sobol_total_chart = HorizontalBarPanel(panel, "Total-Order Indices (ST)")
        charts.Add(self.sobol_total_chart, 1, wx.EXPAND)

        main.Add(charts, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Interaction info
        self.interaction_info = wx.StaticText(
            panel, label="Run analysis to see parameter sensitivities and interactions."
        )
        self.interaction_info.SetFont(
            wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        self.interaction_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.interaction_info, 0, wx.ALL, 12)

        panel.SetSizer(main)
        return panel

    def _create_contributions_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        # Controls row
        ctrl_panel = wx.Panel(panel)
        ctrl_panel.SetBackgroundColour(Colors.BG_WHITE)
        ctrl = wx.BoxSizer(wx.HORIZONTAL)

        ctrl.Add(
            wx.StaticText(ctrl_panel, label="View:"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
            12,
        )
        self.contrib_view = wx.Choice(
            ctrl_panel, choices=["Components", "Sheets", "Top Contributors"]
        )
        self.contrib_view.SetSelection(0)
        self.contrib_view.Bind(wx.EVT_CHOICE, self._on_contrib_view_change)
        ctrl.Add(self.contrib_view, 0, wx.ALL, 8)

        self.btn_importance = wx.Button(
            ctrl_panel, label="📊 Compute Importance Measures"
        )
        self.btn_importance.SetBackgroundColour(Colors.PRIMARY)
        self.btn_importance.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_importance.Bind(wx.EVT_BUTTON, self._on_compute_importance)
        self.btn_importance.SetToolTip(
            "Calculate Birnbaum, RAW, RRW importance for all components"
        )
        ctrl.Add(self.btn_importance, 0, wx.ALL, 8)

        ctrl_panel.SetSizer(ctrl)
        main.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 8)

        # Split view: chart on left, table on right
        splitter = wx.SplitterWindow(panel, style=wx.SP_LIVE_UPDATE)
        splitter.SetMinimumPaneSize(250)

        # Left: Chart
        left_panel = wx.Panel(splitter)
        left_panel.SetBackgroundColour(Colors.BG_WHITE)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        self.contrib_chart = HorizontalBarPanel(
            left_panel, "Failure Rate Contributions"
        )
        left_sizer.Add(self.contrib_chart, 1, wx.EXPAND | wx.ALL, 4)

        left_panel.SetSizer(left_sizer)

        # Right: Scrollable table with importance measures
        right_panel = wx.Panel(splitter)
        right_panel.SetBackgroundColour(Colors.BG_WHITE)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(right_panel, label="Component Contributions")
        title.SetFont(
            wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        )
        title.SetForegroundColour(Colors.TEXT_DARK)
        right_sizer.Add(title, 0, wx.ALL, 8)

        # Enhanced list with more columns
        self.contrib_list = wx.ListCtrl(
            right_panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE
        )
        self.contrib_list.SetBackgroundColour(Colors.BG_WHITE)
        self.contrib_list.InsertColumn(0, "#", width=40)
        self.contrib_list.InsertColumn(1, "Component", width=150)
        self.contrib_list.InsertColumn(2, "Type", width=120)
        self.contrib_list.InsertColumn(3, "λ (FIT)", width=80)
        self.contrib_list.InsertColumn(4, "Contrib %", width=75)
        self.contrib_list.InsertColumn(5, "Cumul %", width=70)
        self.contrib_list.InsertColumn(6, "Birnbaum", width=75)
        self.contrib_list.InsertColumn(7, "RAW", width=65)
        self.contrib_list.InsertColumn(8, "Derating", width=80)
        right_sizer.Add(self.contrib_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Summary panel
        self.contrib_summary = wx.StaticText(right_panel, label="")
        self.contrib_summary.SetFont(
            wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        )
        self.contrib_summary.SetForegroundColour(Colors.TEXT_MEDIUM)
        right_sizer.Add(self.contrib_summary, 0, wx.ALL, 8)

        right_panel.SetSizer(right_sizer)

        splitter.SplitVertically(left_panel, right_panel, 350)
        main.Add(splitter, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        panel.SetSizer(main)

        # Initialize importance data storage
        self.importance_measures: Dict[str, ImportanceMeasures] = {}

        self._update_contributions()
        return panel

    def _create_report_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)

        # Report text
        self.report_text = wx.TextCtrl(
            panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE
        )
        self.report_text.SetBackgroundColour(Colors.BG_WHITE)
        self.report_text.SetFont(
            wx.Font(
                10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
            )
        )
        main.Add(self.report_text, 1, wx.EXPAND | wx.ALL, 8)

        # Export buttons
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

    def _create_footer(self) -> wx.Panel:
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
    # Analysis Methods
    # =========================================================================

    def _on_calibrate_mc(self, event):
        """Open Monte Carlo calibration dialog."""
        dlg = MonteCarloCalibrationDialog(self, self.mc_config)
        if dlg.ShowModal() == wx.ID_OK:
            self.mc_config = dlg.get_config()
            # Update UI to reflect new config
            self.mc_n.SetValue(self.mc_config.n_simulations)
            self.mc_unc.SetValue(self.mc_config.base_uncertainty_cv * 100)
            self.status.SetLabel("Monte Carlo configuration updated")
        dlg.Destroy()

    def _on_run_mc(self, event):
        """Run Monte Carlo with threading and progress dialog."""
        # Update config from UI
        self.mc_config.n_simulations = self.mc_n.GetValue()
        self.mc_config.base_uncertainty_cv = self.mc_unc.GetValue() / 100.0

        components = self._extract_components_for_mc()
        if not components:
            wx.MessageBox(
                "No components found. Using simplified Monte Carlo.",
                "Warning",
                wx.OK | wx.ICON_WARNING,
            )
            # Fallback to simple MC
            self.mc_result = quick_monte_carlo(
                self.system_lambda,
                self.mission_hours,
                uncertainty_percent=self.mc_unc.GetValue(),
                n_simulations=self.mc_n.GetValue(),
            )
            self._update_mc_display()
            return

        # Create MC inputs
        mc_inputs = []
        for comp in components:
            mc_inputs.append(
                ComponentMCInput(
                    reference=comp.get("ref", "?"),
                    component_type=comp.get("type", "Resistor"),
                    base_params=comp.get("params", {}),
                )
            )

        # Disable buttons
        self.btn_mc.Disable()
        self.btn_mc_sheets.Disable()
        self.btn_calibrate.Disable()

        # Create and show progress dialog
        progress_dlg = MCProgressDialog(self, "Monte Carlo Simulation")
        progress_dlg.Show()

        # Thread-safe result storage
        self._mc_thread_result = None
        self._mc_thread_error = None

        def run_mc_thread():
            """Worker thread for Monte Carlo."""
            try:

                def progress_cb(current, total):
                    # Update progress dialog from main thread
                    wx.CallAfter(
                        progress_dlg.update, current / total * 100, f"{current}/{total}"
                    )
                    return not progress_dlg.cancelled

                result, _ = monte_carlo_components_optimized(
                    mc_inputs, self.mission_hours, self.mc_config, progress_cb
                )
                self._mc_thread_result = result
            except Exception as e:
                self._mc_thread_error = e
            finally:
                wx.CallAfter(self._mc_thread_complete, progress_dlg)

        # Start worker thread
        thread = threading.Thread(target=run_mc_thread, daemon=True)
        thread.start()

    def _mc_thread_complete(self, progress_dlg):
        """Called when MC thread completes (from main thread via CallAfter)."""
        progress_dlg.Destroy()

        # Re-enable buttons
        self.btn_mc.Enable()
        self.btn_mc_sheets.Enable()
        self.btn_calibrate.Enable()

        if self._mc_thread_error:
            wx.MessageBox(str(self._mc_thread_error), "Error", wx.OK | wx.ICON_ERROR)
            self.status.SetLabel(f"Error: {self._mc_thread_error}")
            return

        if self._mc_thread_result:
            self.mc_result = self._mc_thread_result
            self._update_mc_display()

    def _update_mc_display(self):
        """Update all MC displays with current mc_result."""
        if not self.mc_result:
            return

        # Update histogram
        self.histogram.set_data(
            self.mc_result.samples,
            self.mc_result.mean,
            self.mc_result.percentile_5,
            self.mc_result.percentile_95,
        )

        # Update convergence
        self.convergence.set_data(self.mc_result.samples)

        # Update stats
        ci_lo, ci_hi = self.mc_result.confidence_interval(0.90)
        self.stats_card.set_stats(
            {
                "mean": self.mc_result.mean,
                "std": self.mc_result.std,
                "median": self.mc_result.percentile_50,
                "p5": self.mc_result.percentile_5,
                "p95": self.mc_result.percentile_95,
                "ci_width": ci_hi - ci_lo,
                "cv": (
                    self.mc_result.std / self.mc_result.mean
                    if self.mc_result.mean > 0
                    else 0
                ),
                "n_sims": self.mc_result.n_simulations,
                "converged": self.mc_result.converged,
            }
        )

        runtime = self.mc_result.runtime_seconds
        status_msg = (
            f"Monte Carlo complete: {self.mc_result.n_simulations:,} simulations"
        )
        if runtime > 0:
            status_msg += f" in {runtime:.1f}s"
        self.status.SetLabel(status_msg)
        self._update_report()

    def _extract_components_for_mc(self) -> List[Dict]:
        """Extract component data from sheet_data for Monte Carlo analysis."""
        components = []

        for sheet_path, data in self.sheet_data.items():
            sheet_comps = data.get("components", [])
            for comp in sheet_comps:
                # Build component dict for MC
                comp_type = comp.get("class", "Resistor")
                if comp_type in ("Unknown", "", None):
                    comp_type = "Resistor"

                # Get params if stored, otherwise use defaults
                params = comp.get("params", {})
                if not params:
                    params = {
                        "t_ambient": 25.0,
                        "t_junction": 85.0,
                        "n_cycles": 5256,
                        "delta_t": 3.0,
                        "operating_power": 0.01,
                        "rated_power": 0.125,
                    }

                components.append(
                    {
                        "ref": comp.get("ref", "?"),
                        "type": comp_type,
                        "params": params,
                    }
                )

        return components

    def _on_run_mc_sheets(self, event):
        """Run Monte Carlo analysis for each sheet/block individually with threading."""
        if not self.sheet_data:
            wx.MessageBox(
                "No sheet data available.", "No Data", wx.OK | wx.ICON_WARNING
            )
            return

        # Disable buttons
        self.btn_mc_sheets.Disable()
        self.btn_mc.Disable()
        self.btn_calibrate.Disable()

        n = min(self.mc_n.GetValue(), 2000)  # Limit per-sheet
        unc = self.mc_unc.GetValue()

        # Create and show progress dialog
        progress_dlg = MCProgressDialog(self, "Per-Sheet Monte Carlo")
        progress_dlg.Show()

        self._sheet_mc_thread_results = {}
        self._sheet_mc_thread_error = None

        def run_sheet_mc_thread():
            """Worker thread for per-sheet Monte Carlo."""
            try:
                total_sheets = len(self.sheet_data)
                completed = 0

                for sheet_path, data in self.sheet_data.items():
                    if progress_dlg.cancelled:
                        break

                    sheet_name = sheet_path.rstrip("/").split("/")[-1] or "Root"
                    wx.CallAfter(
                        progress_dlg.update,
                        completed / total_sheets * 100,
                        f"Processing {sheet_name}",
                    )

                    sheet_comps = data.get("components", [])
                    if not sheet_comps:
                        completed += 1
                        continue

                    mc_components = [
                        {
                            "ref": comp.get("ref", "?"),
                            "value": comp.get("value", ""),
                            "class": comp.get("class", "Resistor"),
                            "params": comp.get("params", {}),
                        }
                        for comp in sheet_comps
                    ]

                    mc_result, lambda_samples = monte_carlo_sheet(
                        mc_components,
                        self.mission_hours,
                        n_simulations=n,
                        uncertainty_percent=unc,
                        seed=42 + completed,
                    )

                    self._sheet_mc_thread_results[sheet_path] = SheetMCResult(
                        sheet_path=sheet_path,
                        mc_result=mc_result,
                        lambda_samples=lambda_samples,
                    )

                    completed += 1

            except Exception as e:
                self._sheet_mc_thread_error = e
            finally:
                wx.CallAfter(self._sheet_mc_thread_complete, progress_dlg, n)

        thread = threading.Thread(target=run_sheet_mc_thread, daemon=True)
        thread.start()

    def _sheet_mc_thread_complete(self, progress_dlg, n_per_sheet):
        """Called when per-sheet MC thread completes."""
        progress_dlg.Destroy()

        self.btn_mc_sheets.Enable()
        self.btn_mc.Enable()
        self.btn_calibrate.Enable()

        if self._sheet_mc_thread_error:
            wx.MessageBox(
                str(self._sheet_mc_thread_error), "Error", wx.OK | wx.ICON_ERROR
            )
            self.status.SetLabel(f"Error: {self._sheet_mc_thread_error}")
            return

        self.sheet_mc_results = self._sheet_mc_thread_results
        completed = len(self.sheet_mc_results)
        self.status.SetLabel(
            f"Per-sheet MC complete: {completed} sheets × {n_per_sheet} simulations each"
        )
        self._update_report()

    def _on_run_sobol(self, event):
        if not self.sheet_data:
            wx.MessageBox(
                "No sheet data for sensitivity analysis.",
                "No Data",
                wx.OK | wx.ICON_WARNING,
            )
            return

        self.status.SetLabel("Running Sobol analysis...")
        self.btn_sobol.Disable()
        wx.Yield()

        try:
            n = self.sobol_n.GetValue()

            # Build parameter bounds from sheet lambdas (Dict not list!)
            param_bounds = {}
            for path, data in self.sheet_data.items():
                lam = data.get("lambda", 0)
                if lam > 0:
                    name = path.rstrip("/").split("/")[-1] or "Root"
                    # Ensure unique names
                    base_name = name
                    i = 1
                    while name in param_bounds:
                        name = f"{base_name}_{i}"
                        i += 1
                    param_bounds[name] = (lam * 0.7, lam * 1.3)

            if len(param_bounds) < 2:
                wx.MessageBox(
                    "Need at least 2 sheets with non-zero lambda for sensitivity analysis.",
                    "Insufficient Data",
                    wx.OK | wx.ICON_WARNING,
                )
                self.btn_sobol.Enable()
                return

            # Model: sum lambdas -> reliability
            def model(params: Dict[str, float]) -> float:
                total_lam = sum(params.values())
                return reliability_from_lambda(total_lam, self.mission_hours)

            # Run analysis
            analyzer = SobolAnalyzer(seed=42)
            self.sobol_result = analyzer.analyze(model, param_bounds, n_samples=n)

            # Update charts
            first_data = list(
                zip(self.sobol_result.parameter_names, self.sobol_result.S_first)
            )
            first_data.sort(key=lambda x: -x[1])
            self.sobol_first_chart.set_data(
                first_data, max_value=1.0, x_label="First-Order Index"
            )

            total_data = list(
                zip(self.sobol_result.parameter_names, self.sobol_result.S_total)
            )
            total_data.sort(key=lambda x: -x[1])
            self.sobol_total_chart.set_data(
                total_data, max_value=1.0, x_label="Total-Order Index"
            )

            # Interaction info
            significant = [
                self.sobol_result.parameter_names[i]
                for i in self.sobol_result.significant_interactions
            ]

            if significant:
                info = f"Warning: Significant interactions detected in: {', '.join(significant)}"
                self.interaction_info.SetForegroundColour(Colors.WARNING)
            else:
                info = "OK: No significant parameter interactions detected."
                self.interaction_info.SetForegroundColour(Colors.SUCCESS)
            self.interaction_info.SetLabel(info)

            self.status.SetLabel(
                f"Sensitivity analysis complete: {len(param_bounds)} parameters"
            )
            self._update_report()

        except Exception as e:
            import traceback

            traceback.print_exc()
            wx.MessageBox(str(e), "Analysis Error", wx.OK | wx.ICON_ERROR)
            self.status.SetLabel(f"Error: {e}")
        finally:
            self.btn_sobol.Enable()

    def _on_contrib_view_change(self, event):
        """Handle contribution view change."""
        self._update_contributions()

    def _on_compute_importance(self, event):
        """Compute importance measures for all components."""
        if not self.sheet_data:
            wx.MessageBox("No data available.", "No Data", wx.OK | wx.ICON_WARNING)
            return

        self.status.SetLabel("Computing importance measures...")
        self.btn_importance.Disable()
        wx.Yield()

        try:
            # Gather all component lambdas
            component_lambdas = {}
            for path, data in self.sheet_data.items():
                sheet_name = path.rstrip("/").split("/")[-1] or "Root"
                for comp in data.get("components", []):
                    ref = comp.get("ref", "?")
                    key = f"{sheet_name}/{ref}"
                    lam = comp.get("lambda", 0)
                    if lam > 0:
                        component_lambdas[key] = lam

            if len(component_lambdas) < 2:
                wx.MessageBox(
                    "Need at least 2 components for importance analysis.",
                    "Insufficient Data",
                    wx.OK | wx.ICON_WARNING,
                )
                return

            # Calculate importance measures
            analyzer = ImportanceAnalyzer()
            total_lambda = sum(component_lambdas.values())

            self.importance_measures = {}
            for name, lam in component_lambdas.items():
                r_comp = reliability_from_lambda(lam, self.mission_hours)
                contrib_pct = (lam / total_lambda * 100) if total_lambda > 0 else 0

                # Calculate Birnbaum importance (approximation for series system)
                # dR_sys/dR_comp ≈ R_sys / R_comp for series
                r_sys = reliability_from_lambda(total_lambda, self.mission_hours)
                birnbaum = r_sys / r_comp if r_comp > 0 else 0

                # RAW: how much worse if component fails
                lam_without = total_lambda - lam
                r_without = reliability_from_lambda(lam_without, self.mission_hours)
                raw = r_without / r_sys if r_sys > 0 else 1

                # RRW: how much better if component is perfect
                rrw = r_sys / r_without if r_without > 0 else 1

                # Derating recommendation based on contribution
                if contrib_pct > 20:
                    derating = 0.5
                    action = "50% derating, redundancy recommended"
                elif contrib_pct > 10:
                    derating = 0.6
                    action = "60% derating recommended"
                elif contrib_pct > 5:
                    derating = 0.7
                    action = "70% derating recommended"
                elif contrib_pct > 2:
                    derating = 0.8
                    action = "80% derating recommended"
                else:
                    derating = 1.0
                    action = "Standard operation"

                self.importance_measures[name] = ImportanceMeasures(
                    component_name=name,
                    lambda_fit=lam * 1e9,
                    reliability=r_comp,
                    contribution_pct=contrib_pct,
                    birnbaum=birnbaum,
                    raw=raw,
                    rrw=rrw,
                    fussell_vesely=contrib_pct / 100,  # Approximation for series
                    derating_factor=derating,
                    recommended_action=action,
                )

            # Assign criticality ranks
            sorted_by_contrib = sorted(
                self.importance_measures.items(), key=lambda x: -x[1].contribution_pct
            )
            for rank, (name, measures) in enumerate(sorted_by_contrib, 1):
                measures.criticality_rank = rank

            self.status.SetLabel(
                f"Computed importance for {len(self.importance_measures)} components"
            )
            self._update_contributions()

        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
            self.status.SetLabel(f"Error: {e}")
        finally:
            self.btn_importance.Enable()

    def _update_contributions(self):
        if not self.sheet_data:
            return

        view_mode = (
            self.contrib_view.GetSelection() if hasattr(self, "contrib_view") else 0
        )

        # Gather contributions based on view mode
        contribs = []
        total_lam = 0

        if view_mode == 1:  # Sheets view
            for path, data in self.sheet_data.items():
                lam = data.get("lambda", 0)
                if lam > 0:
                    name = path.rstrip("/").split("/")[-1] or "Root"
                    comp_type = "Sheet"
                    contribs.append((name, lam, comp_type, path))
                    total_lam += lam
        else:  # Components or Top Contributors view
            for path, data in self.sheet_data.items():
                sheet_name = path.rstrip("/").split("/")[-1] or "Root"
                for comp in data.get("components", []):
                    lam = comp.get("lambda", 0)
                    if lam > 0:
                        ref = comp.get("ref", "?")
                        comp_type = comp.get("class", "Unknown")
                        key = f"{sheet_name}/{ref}"
                        contribs.append((key, lam, comp_type, path))
                        total_lam += lam

        if total_lam == 0:
            return

        contribs.sort(key=lambda x: -x[1])

        # For top contributors, limit to top 30
        if view_mode == 2:
            contribs = contribs[:30]

        # Update chart
        chart_data = [(c[0][:25], c[1] / total_lam) for c in contribs[:15]]
        self.contrib_chart.set_data(
            chart_data, max_value=1.0, x_label="Relative Contribution"
        )

        # Update list with importance measures if available
        self.contrib_list.DeleteAllItems()
        cumulative = 0
        for i, (name, lam, comp_type, path) in enumerate(contribs):
            pct = lam / total_lam * 100
            cumulative += pct

            idx = self.contrib_list.InsertItem(i, str(i + 1))
            self.contrib_list.SetItem(idx, 1, name[:25])
            self.contrib_list.SetItem(idx, 2, comp_type[:20])
            self.contrib_list.SetItem(idx, 3, f"{lam*1e9:.2f}")
            self.contrib_list.SetItem(idx, 4, f"{pct:.1f}%")
            self.contrib_list.SetItem(idx, 5, f"{cumulative:.1f}%")

            # Add importance measures if computed
            if (
                hasattr(self, "importance_measures")
                and name in self.importance_measures
            ):
                im = self.importance_measures[name]
                self.contrib_list.SetItem(idx, 6, f"{im.birnbaum:.4f}")
                self.contrib_list.SetItem(idx, 7, f"{im.raw:.2f}")
                self.contrib_list.SetItem(idx, 8, f"{int(im.derating_factor*100)}%")
            else:
                self.contrib_list.SetItem(idx, 6, "-")
                self.contrib_list.SetItem(idx, 7, "-")
                self.contrib_list.SetItem(idx, 8, "-")

        # Update summary
        n_total = len(contribs)
        n_top = sum(1 for c in contribs if (c[1] / total_lam * 100) > 5)
        summary = f"Total: {n_total} items | >5% contribution: {n_top} | Total λ: {total_lam*1e9:.2f} FIT"
        if hasattr(self, "importance_measures") and self.importance_measures:
            summary += f" | Importance computed: ✓"
        self.contrib_summary.SetLabel(summary)

    def _update_report(self):
        lines = []

        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)

        lines.append("=" * 70)
        lines.append("           RELIABILITY ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append("SYSTEM PARAMETERS")
        lines.append("-" * 50)
        lines.append(
            f"  Mission Duration:      {years:.2f} years ({self.mission_hours:.0f} hours)"
        )
        lines.append(f"  System Failure Rate:   {self.system_lambda*1e9:.2f} FIT")
        lines.append(f"  Point Estimate R(t):   {r:.6f}")
        lines.append("")

        # Monte Carlo
        if self.mc_result:
            mc = self.mc_result
            lines.append("MONTE CARLO ANALYSIS")
            lines.append("-" * 50)
            lines.append(f"  Simulations:           {mc.n_simulations:,}")
            lines.append(f"  Mean Reliability:      {mc.mean:.6f}")
            lines.append(f"  Standard Deviation:    {mc.std:.6f}")
            lines.append(f"  Median:                {mc.percentile_50:.6f}")
            lines.append(f"  5th Percentile:        {mc.percentile_5:.6f}")
            lines.append(f"  95th Percentile:       {mc.percentile_95:.6f}")
            ci_lo, ci_hi = mc.confidence_interval(0.90)
            lines.append(f"  90% Confidence:        [{ci_lo:.6f}, {ci_hi:.6f}]")
            cv = mc.std / mc.mean if mc.mean > 0 else 0
            lines.append(f"  Coeff. of Variation:   {cv*100:.2f}%")
            lines.append("")

        # Sensitivity
        if self.sobol_result:
            sr = self.sobol_result
            lines.append("SENSITIVITY ANALYSIS (SOBOL INDICES)")
            lines.append("-" * 50)
            lines.append(f"  {'Parameter':<25} {'S1':>10} {'ST':>10} {'ST-S1':>10}")
            lines.append("  " + "-" * 57)

            ranked = sorted(
                zip(sr.parameter_names, sr.S_first, sr.S_total), key=lambda x: -x[2]
            )
            for name, s1, st in ranked:
                interact = st - s1
                flag = "  *" if interact > 0.1 * st and st > 0.01 else ""
                lines.append(
                    f"  {name:<25} {s1:>10.4f} {st:>10.4f} {interact:>10.4f}{flag}"
                )

            if sr.significant_interactions:
                lines.append("")
                names = [sr.parameter_names[i] for i in sr.significant_interactions]
                lines.append(f"  * Significant interactions: {', '.join(names)}")
            lines.append("")

        # Contributions
        if self.sheet_data:
            lines.append("FAILURE RATE CONTRIBUTIONS")
            lines.append("-" * 50)

            contribs = []
            total_lam = 0
            for path, data in self.sheet_data.items():
                lam = data.get("lambda", 0)
                if lam > 0:
                    name = path.rstrip("/").split("/")[-1] or "Root"
                    contribs.append((name, lam, path))
                    total_lam += lam

            contribs.sort(key=lambda x: -x[1])

            if total_lam > 0:
                cumul = 0
                for name, lam, path in contribs[:20]:
                    pct = lam / total_lam * 100
                    cumul += pct
                    lines.append(
                        f"  {name:<25} {lam*1e9:>8.2f} FIT  ({pct:>5.1f}%)  cum: {cumul:>5.1f}%"
                    )

                if len(contribs) > 20:
                    lines.append(f"  ... and {len(contribs) - 20} more")
            lines.append("")

        # Component details
        if self.sheet_data:
            lines.append("COMPONENT DETAILS BY SHEET")
            lines.append("-" * 50)

            for path in sorted(self.sheet_data.keys()):
                data = self.sheet_data[path]
                sheet_name = path.rstrip("/").split("/")[-1] or "Root"
                sheet_lam = data.get("lambda", 0)
                sheet_r = data.get("r", 1)

                lines.append(f"")
                lines.append(f"  [{sheet_name}]")
                lines.append(f"  Path: {path}")
                lines.append(
                    f"  Sheet Lambda: {sheet_lam*1e9:.2f} FIT, R: {sheet_r:.6f}"
                )

                components = data.get("components", [])
                if components:
                    lines.append(f"  Components ({len(components)}):")
                    for c in components[:15]:
                        ref = c.get("ref", "?")
                        val = c.get("value", "")[:15]
                        cls = c.get("class", "")[:20]
                        c_lam = c.get("lambda", 0)
                        c_r = c.get("r", 1)
                        lines.append(
                            f"    {ref:<8} {val:<15} {cls:<20} L={c_lam*1e9:>6.2f} FIT  R={c_r:.6f}"
                        )
                    if len(components) > 15:
                        lines.append(f"    ... and {len(components) - 15} more")

                # Add per-sheet MC results if available
                if path in self.sheet_mc_results:
                    smc = self.sheet_mc_results[path].mc_result
                    lines.append(f"  Monte Carlo Analysis ({smc.n_simulations} sims):")
                    lines.append(f"    Mean R:    {smc.mean:.6f}")
                    lines.append(f"    Std:       {smc.std:.6f}")
                    lines.append(
                        f"    5%-95%:    [{smc.percentile_5:.6f}, {smc.percentile_95:.6f}]"
                    )

        # Per-sheet Monte Carlo Summary
        if self.sheet_mc_results:
            lines.append("")
            lines.append("PER-SHEET MONTE CARLO SUMMARY")
            lines.append("-" * 50)
            lines.append(
                f"  {'Sheet':<25} {'Mean R':>10} {'Std':>10} {'5%':>10} {'95%':>10}"
            )
            lines.append("  " + "-" * 67)

            for path in sorted(self.sheet_mc_results.keys()):
                smc = self.sheet_mc_results[path].mc_result
                sheet_name = path.rstrip("/").split("/")[-1] or "Root"
                lines.append(
                    f"  {sheet_name:<25} {smc.mean:>10.6f} {smc.std:>10.6f} {smc.percentile_5:>10.6f} {smc.percentile_95:>10.6f}"
                )
            lines.append("")

            # Distribution summary (text-based histogram)
            lines.append("  Per-Sheet Reliability Distribution (text histogram):")
            lines.append("  " + "-" * 50)
            for path in sorted(self.sheet_mc_results.keys()):
                smc = self.sheet_mc_results[path].mc_result
                sheet_name = path.rstrip("/").split("/")[-1] or "Root"
                # Simple text bar
                bar_len = int((smc.mean - 0.9) * 100)  # Scale 0.9-1.0 to 0-10
                bar_len = max(0, min(40, bar_len))
                bar = "█" * bar_len + "░" * (40 - bar_len)
                lines.append(f"  {sheet_name:<20} |{bar}| {smc.mean:.4f}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  Generated by KiCad Reliability Calculator v2.0.0")
        lines.append("  IEC TR 62380 Methodology")
        lines.append("=" * 70)

        self.report_text.SetValue("\n".join(lines))

    def _on_export_html(self, event):
        dlg = wx.FileDialog(
            self,
            "Export HTML Report",
            wildcard="HTML (*.html)|*.html",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            try:
                report_data = self._build_report_data()
                generator = ReportGenerator()
                html = generator.generate_html(report_data)
                with open(dlg.GetPath(), "w", encoding="utf-8") as f:
                    f.write(html)
                self.status.SetLabel(f"Exported: {dlg.GetPath()}")
            except Exception as e:
                wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _on_export_csv(self, event):
        dlg = wx.FileDialog(
            self,
            "Export CSV",
            wildcard="CSV (*.csv)|*.csv",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            try:
                report_data = self._build_report_data()
                generator = ReportGenerator()
                csv = generator.generate_csv(report_data)
                with open(dlg.GetPath(), "w", encoding="utf-8") as f:
                    f.write(csv)
                self.status.SetLabel(f"Exported: {dlg.GetPath()}")
            except Exception as e:
                wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _build_report_data(self) -> ReportData:
        """Build ReportData for export."""
        years = self.mission_hours / (365 * 24)
        mttf = 1 / self.system_lambda if self.system_lambda > 0 else float("inf")

        # Build blocks from sheet data
        blocks = []
        for path, data in self.sheet_data.items():
            sheet_name = path.rstrip("/").split("/")[-1] or "Root"

            # Build component list
            components = []
            for c in data.get("components", []):
                lam = c.get("lambda", 0)
                total_lam = data.get("lambda", 1e-15)
                contrib = (lam / total_lam * 100) if total_lam > 0 else 0

                comp = ComponentReportData(
                    ref=c.get("ref", "?"),
                    value=c.get("value", ""),
                    component_type=c.get("class", "Unknown"),
                    lambda_fit=lam * 1e9,
                    reliability=c.get("r", 1.0),
                    contribution_pct=contrib,
                    parameters=c.get("params", {}),
                )
                components.append(comp)

            block = BlockReportData(
                id=path,
                name=sheet_name,
                path=path,
                connection_type="series",
                lambda_total=data.get("lambda", 0),
                reliability=data.get("r", 1.0),
                components=components,
            )

            # Add per-sheet MC if available
            if path in self.sheet_mc_results:
                smc = self.sheet_mc_results[path].mc_result
                block.mc_result = {
                    "mean": smc.mean,
                    "std": smc.std,
                    "percentile_5": smc.percentile_5,
                    "percentile_95": smc.percentile_95,
                    "n_simulations": smc.n_simulations,
                }

            blocks.append(block)

        # Build Monte Carlo dict
        mc_dict = None
        mc_samples = None
        if self.mc_result:
            ci_lo, ci_hi = self.mc_result.confidence_interval(0.90)
            mc_dict = {
                "mean": self.mc_result.mean,
                "std": self.mc_result.std,
                "percentile_5": self.mc_result.percentile_5,
                "percentile_50": self.mc_result.percentile_50,
                "percentile_95": self.mc_result.percentile_95,
                "ci_90_low": ci_lo,
                "ci_90_high": ci_hi,
                "n_simulations": self.mc_result.n_simulations,
                "converged": self.mc_result.converged,
            }
            mc_samples = list(self.mc_result.samples[:5000])  # Limit for report

        # Build sensitivity dict
        sensitivity_dict = None
        if self.sobol_result:
            sensitivity_dict = self.sobol_result.to_dict()

            # Add importance measures if available
            if hasattr(self, "importance_measures") and self.importance_measures:
                sensitivity_dict["importance_measures"] = {
                    name: im.to_dict() for name, im in self.importance_measures.items()
                }

        return ReportData(
            project_name="KiCad Project",
            mission_hours=self.mission_hours,
            mission_years=years,
            n_cycles=5256,  # Default
            delta_t=3.0,
            system_reliability=reliability_from_lambda(
                self.system_lambda, self.mission_hours
            ),
            system_lambda=self.system_lambda,
            system_mttf_hours=mttf,
            blocks=blocks,
            monte_carlo=mc_dict,
            mc_samples=mc_samples,
            sensitivity=sensitivity_dict,
            sheets=self.sheet_data,
        )

    def _generate_html(self) -> str:
        """Legacy HTML generation - uses new report generator."""
        report_data = self._build_report_data()
        generator = ReportGenerator()
        return generator.generate_html(report_data)

    def _generate_csv(self) -> str:
        """Legacy CSV generation - uses new report generator."""
        report_data = self._build_report_data()
        generator = ReportGenerator()
        return generator.generate_csv(report_data)
