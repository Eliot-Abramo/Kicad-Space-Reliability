"""
Analysis Dialog - Industrial-Grade Uncertainty & Sensitivity Visualization
==========================================================================
Professional presentation of Monte Carlo and Sobol sensitivity analysis results
with consistent scales, clear legends, and publication-quality graphics.
"""

import wx
import wx.lib.scrolledpanel as scrolled
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .monte_carlo import MonteCarloResult, quick_monte_carlo, MonteCarloAnalyzer, ParameterDistribution
from .sensitivity_analysis import SobolResult, SobolAnalyzer
from .reliability_math import reliability_from_lambda, calculate_component_lambda


# =============================================================================
# Design System - Industrial Color Palette
# =============================================================================

class AnalysisColors:
    """Professional industrial color palette for analysis visualization."""
    
    # Primary colors
    PRIMARY = wx.Colour(30, 136, 229)      # Blue
    SECONDARY = wx.Colour(67, 160, 71)     # Green
    TERTIARY = wx.Colour(251, 140, 0)      # Orange
    
    # Chart colors (colorblind-safe palette)
    CHART_BLUE = wx.Colour(55, 126, 184)
    CHART_GREEN = wx.Colour(77, 175, 74)
    CHART_ORANGE = wx.Colour(255, 127, 0)
    CHART_RED = wx.Colour(228, 26, 28)
    CHART_PURPLE = wx.Colour(152, 78, 163)
    CHART_YELLOW = wx.Colour(255, 255, 51)
    CHART_BROWN = wx.Colour(166, 86, 40)
    CHART_PINK = wx.Colour(247, 129, 191)
    
    # Grays
    BACKGROUND = wx.Colour(250, 250, 252)
    PANEL_BG = wx.Colour(255, 255, 255)
    GRID = wx.Colour(230, 230, 230)
    BORDER = wx.Colour(200, 200, 200)
    TEXT_PRIMARY = wx.Colour(33, 37, 41)
    TEXT_SECONDARY = wx.Colour(108, 117, 125)
    TEXT_MUTED = wx.Colour(173, 181, 189)
    
    # Status colors
    SUCCESS = wx.Colour(40, 167, 69)
    WARNING = wx.Colour(255, 193, 7)
    DANGER = wx.Colour(220, 53, 69)
    INFO = wx.Colour(23, 162, 184)
    
    # Chart palette for multiple series
    PALETTE = [CHART_BLUE, CHART_GREEN, CHART_ORANGE, CHART_RED, 
               CHART_PURPLE, CHART_BROWN, CHART_PINK]


@dataclass
class AnalysisSettings:
    """Settings for analysis."""
    mc_simulations: int = 10000
    mc_uncertainty_percent: float = 25.0
    mc_seed: Optional[int] = None
    sobol_samples: int = 1024
    confidence_level: float = 0.90


@dataclass
class SystemAnalysisResult:
    """Complete analysis results for a system."""
    system_lambda: float
    system_reliability: float
    mission_hours: float
    monte_carlo: Optional[MonteCarloResult] = None
    sensitivity: Optional[Dict[str, SobolResult]] = None
    component_contributions: Optional[Dict[str, float]] = None
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None


# =============================================================================
# Chart Drawing Components
# =============================================================================

class ChartPanel(wx.Panel):
    """Base class for chart panels with professional styling."""
    
    MARGIN_LEFT = 70
    MARGIN_RIGHT = 30
    MARGIN_TOP = 40
    MARGIN_BOTTOM = 50
    
    def __init__(self, parent, title: str = "", size=(400, 300)):
        super().__init__(parent, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.title = title
        self.data = None
        
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: self.Refresh())
    
    def on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        w, h = self.GetSize()
        
        # Background
        gc.SetBrush(wx.Brush(AnalysisColors.PANEL_BG))
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.DrawRectangle(0, 0, w, h)
        
        # Border
        gc.SetPen(wx.Pen(AnalysisColors.BORDER, 1))
        gc.DrawRectangle(0, 0, w-1, h-1)
        
        # Title
        if self.title:
            font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
            gc.SetFont(font, AnalysisColors.TEXT_PRIMARY)
            tw = gc.GetTextExtent(self.title)[0]
            gc.DrawText(self.title, (w - tw) / 2, 8)
        
        # Draw chart area
        self.draw_chart(gc, w, h)
    
    def draw_chart(self, gc, w, h):
        """Override in subclasses to draw specific chart type."""
        pass
    
    def draw_axis(self, gc, x0, y0, x1, y1, is_x=True):
        """Draw axis line."""
        gc.SetPen(wx.Pen(AnalysisColors.TEXT_SECONDARY, 1))
        gc.StrokeLine(x0, y0, x1, y1)


class HistogramChart(ChartPanel):
    """Professional histogram visualization for Monte Carlo results."""
    
    def __init__(self, parent, title: str = "Distribution", n_bins: int = 40):
        super().__init__(parent, title, size=(500, 350))
        self.n_bins = n_bins
        self.samples = None
        self.stats = {}
    
    def set_data(self, samples: np.ndarray, stats: Dict[str, float] = None):
        self.samples = samples
        self.stats = stats or {}
        self.Refresh()
    
    def draw_chart(self, gc, w, h):
        if self.samples is None or len(self.samples) == 0:
            return
        
        # Calculate chart area
        left = self.MARGIN_LEFT
        right = w - self.MARGIN_RIGHT
        top = self.MARGIN_TOP
        bottom = h - self.MARGIN_BOTTOM
        chart_w = right - left
        chart_h = bottom - top
        
        # Calculate histogram
        hist, bin_edges = np.histogram(self.samples, bins=self.n_bins)
        max_count = max(hist)
        
        # Draw grid
        gc.SetPen(wx.Pen(AnalysisColors.GRID, 1, wx.PENSTYLE_DOT))
        for i in range(5):
            y = top + chart_h * i / 4
            gc.StrokeLine(left, y, right, y)
        
        # Draw bars
        bar_width = chart_w / len(hist) - 1
        gc.SetPen(wx.Pen(AnalysisColors.CHART_BLUE.ChangeLightness(80), 1))
        gc.SetBrush(wx.Brush(AnalysisColors.CHART_BLUE))
        
        for i, count in enumerate(hist):
            if count > 0:
                x = left + i * (chart_w / len(hist))
                bar_h = (count / max_count) * chart_h
                gc.DrawRectangle(x, bottom - bar_h, bar_width, bar_h)
        
        # Draw mean line
        if 'mean' in self.stats:
            mean_val = self.stats['mean']
            min_val, max_val = bin_edges[0], bin_edges[-1]
            if min_val < mean_val < max_val:
                x_mean = left + (mean_val - min_val) / (max_val - min_val) * chart_w
                gc.SetPen(wx.Pen(AnalysisColors.CHART_RED, 2, wx.PENSTYLE_SOLID))
                gc.StrokeLine(x_mean, top, x_mean, bottom)
        
        # Draw percentile lines
        for pct_key, color, style in [('p5', AnalysisColors.CHART_ORANGE, wx.PENSTYLE_SHORT_DASH),
                                       ('p95', AnalysisColors.CHART_ORANGE, wx.PENSTYLE_SHORT_DASH)]:
            if pct_key in self.stats:
                pct_val = self.stats[pct_key]
                min_val, max_val = bin_edges[0], bin_edges[-1]
                if min_val < pct_val < max_val:
                    x_pct = left + (pct_val - min_val) / (max_val - min_val) * chart_w
                    gc.SetPen(wx.Pen(color, 1.5, style))
                    gc.StrokeLine(x_pct, top, x_pct, bottom)
        
        # X-axis labels
        font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, AnalysisColors.TEXT_SECONDARY)
        
        min_val, max_val = bin_edges[0], bin_edges[-1]
        for i in range(5):
            val = min_val + (max_val - min_val) * i / 4
            x = left + chart_w * i / 4
            label = f"{val:.4f}"
            tw = gc.GetTextExtent(label)[0]
            gc.DrawText(label, x - tw/2, bottom + 5)
        
        # Y-axis label
        gc.PushState()
        gc.Translate(15, (top + bottom) / 2)
        gc.Rotate(-math.pi / 2)
        gc.DrawText("Frequency", -30, 0)
        gc.PopState()
        
        # X-axis label
        gc.DrawText("Reliability R(t)", left + chart_w/2 - 40, h - 15)
        
        # Draw axes
        gc.SetPen(wx.Pen(AnalysisColors.TEXT_SECONDARY, 1))
        gc.StrokeLine(left, top, left, bottom)
        gc.StrokeLine(left, bottom, right, bottom)
        
        # Legend
        self._draw_legend(gc, right - 150, top + 5)
    
    def _draw_legend(self, gc, x, y):
        font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, AnalysisColors.TEXT_PRIMARY)
        
        items = [
            (AnalysisColors.CHART_RED, "Mean", "solid"),
            (AnalysisColors.CHART_ORANGE, "5th/95th %ile", "dash"),
        ]
        
        for i, (color, label, style) in enumerate(items):
            ly = y + i * 14
            if style == "solid":
                gc.SetPen(wx.Pen(color, 2))
            else:
                gc.SetPen(wx.Pen(color, 1.5, wx.PENSTYLE_SHORT_DASH))
            gc.StrokeLine(x, ly + 5, x + 20, ly + 5)
            gc.DrawText(label, x + 25, ly)


class BarChart(ChartPanel):
    """Professional horizontal bar chart for sensitivity analysis."""
    
    def __init__(self, parent, title: str = "Sensitivity"):
        super().__init__(parent, title, size=(500, 400))
        self.data = []  # List of (name, value, error_low, error_high)
        self.max_value = 1.0
        self.show_total = False
    
    def set_data(self, data: List[Tuple[str, float, float, float]], max_value: float = None, show_total: bool = False):
        self.data = sorted(data, key=lambda x: x[1], reverse=True)
        self.max_value = max_value or max(d[1] + d[3] for d in data) if data else 1.0
        self.max_value = max(self.max_value, 0.01)
        self.show_total = show_total
        self.Refresh()
    
    def draw_chart(self, gc, w, h):
        if not self.data:
            return
        
        left = self.MARGIN_LEFT + 60  # Extra for parameter names
        right = w - self.MARGIN_RIGHT
        top = self.MARGIN_TOP
        bottom = h - self.MARGIN_BOTTOM
        chart_w = right - left
        chart_h = bottom - top
        
        n_bars = len(self.data)
        if n_bars == 0:
            return
        
        bar_height = min(25, chart_h / n_bars - 5)
        spacing = (chart_h - n_bars * bar_height) / (n_bars + 1)
        
        # Draw grid
        gc.SetPen(wx.Pen(AnalysisColors.GRID, 1, wx.PENSTYLE_DOT))
        for i in range(5):
            x = left + chart_w * i / 4
            gc.StrokeLine(x, top, x, bottom)
        
        font = wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, AnalysisColors.TEXT_PRIMARY)
        
        for i, (name, value, err_low, err_high) in enumerate(self.data):
            y = top + spacing + i * (bar_height + spacing)
            
            # Select color based on index
            color = AnalysisColors.PALETTE[i % len(AnalysisColors.PALETTE)]
            
            # Draw bar
            bar_w = (value / self.max_value) * chart_w
            gc.SetBrush(wx.Brush(color))
            gc.SetPen(wx.Pen(color.ChangeLightness(80), 1))
            gc.DrawRectangle(left, y, bar_w, bar_height)
            
            # Draw error bars if significant
            if err_high > 0.001:
                err_x_low = left + ((value - err_low) / self.max_value) * chart_w
                err_x_high = left + ((value + err_high) / self.max_value) * chart_w
                err_x_low = max(err_x_low, left)
                err_x_high = min(err_x_high, right)
                
                gc.SetPen(wx.Pen(AnalysisColors.TEXT_PRIMARY, 1.5))
                mid_y = y + bar_height / 2
                gc.StrokeLine(err_x_low, mid_y, err_x_high, mid_y)
                gc.StrokeLine(err_x_low, mid_y - 3, err_x_low, mid_y + 3)
                gc.StrokeLine(err_x_high, mid_y - 3, err_x_high, mid_y + 3)
            
            # Parameter name
            gc.SetFont(font, AnalysisColors.TEXT_PRIMARY)
            name_short = name[:15] + "..." if len(name) > 15 else name
            tw = gc.GetTextExtent(name_short)[0]
            gc.DrawText(name_short, left - tw - 8, y + (bar_height - 12) / 2)
            
            # Value label
            val_text = f"{value:.3f}"
            if bar_w > 50:
                gc.SetFont(font, wx.WHITE)
                gc.DrawText(val_text, left + 5, y + (bar_height - 12) / 2)
            else:
                gc.SetFont(font, AnalysisColors.TEXT_PRIMARY)
                gc.DrawText(val_text, left + bar_w + 5, y + (bar_height - 12) / 2)
        
        # X-axis
        gc.SetPen(wx.Pen(AnalysisColors.TEXT_SECONDARY, 1))
        gc.StrokeLine(left, bottom, right, bottom)
        
        # X-axis labels
        font_small = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font_small, AnalysisColors.TEXT_SECONDARY)
        for i in range(5):
            val = self.max_value * i / 4
            x = left + chart_w * i / 4
            label = f"{val:.2f}"
            tw = gc.GetTextExtent(label)[0]
            gc.DrawText(label, x - tw/2, bottom + 5)
        
        # X-axis title
        gc.DrawText("Sensitivity Index", left + chart_w/2 - 50, h - 15)


class ConvergenceChart(ChartPanel):
    """Convergence plot for Monte Carlo analysis."""
    
    def __init__(self, parent, title: str = "Convergence"):
        super().__init__(parent, title, size=(400, 250))
        self.history = []  # List of (n_samples, mean_value)
    
    def set_data(self, samples: np.ndarray):
        # Calculate running mean
        self.history = []
        running_sum = 0.0
        for i, s in enumerate(samples):
            running_sum += s
            if (i + 1) % 100 == 0 or i == len(samples) - 1:
                self.history.append((i + 1, running_sum / (i + 1)))
        self.Refresh()
    
    def draw_chart(self, gc, w, h):
        if not self.history:
            return
        
        left = self.MARGIN_LEFT
        right = w - self.MARGIN_RIGHT
        top = self.MARGIN_TOP
        bottom = h - self.MARGIN_BOTTOM
        chart_w = right - left
        chart_h = bottom - top
        
        # Get data range
        n_max = self.history[-1][0]
        values = [h[1] for h in self.history]
        v_min, v_max = min(values), max(values)
        v_range = v_max - v_min
        if v_range < 1e-9:
            v_range = v_max * 0.1 if v_max > 0 else 0.01
        
        # Padding
        v_min -= v_range * 0.1
        v_max += v_range * 0.1
        v_range = v_max - v_min
        
        # Draw grid
        gc.SetPen(wx.Pen(AnalysisColors.GRID, 1, wx.PENSTYLE_DOT))
        for i in range(5):
            y = top + chart_h * i / 4
            gc.StrokeLine(left, y, right, y)
        
        # Draw convergence line
        gc.SetPen(wx.Pen(AnalysisColors.CHART_BLUE, 2))
        path = gc.CreatePath()
        for i, (n, v) in enumerate(self.history):
            x = left + (n / n_max) * chart_w
            y = bottom - ((v - v_min) / v_range) * chart_h
            if i == 0:
                path.MoveToPoint(x, y)
            else:
                path.AddLineToPoint(x, y)
        gc.StrokePath(path)
        
        # Final value line
        final_val = self.history[-1][1]
        y_final = bottom - ((final_val - v_min) / v_range) * chart_h
        gc.SetPen(wx.Pen(AnalysisColors.CHART_GREEN, 1.5, wx.PENSTYLE_SHORT_DASH))
        gc.StrokeLine(left, y_final, right, y_final)
        
        # Axes
        gc.SetPen(wx.Pen(AnalysisColors.TEXT_SECONDARY, 1))
        gc.StrokeLine(left, top, left, bottom)
        gc.StrokeLine(left, bottom, right, bottom)
        
        # Labels
        font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, AnalysisColors.TEXT_SECONDARY)
        
        # X-axis labels
        for i in range(5):
            n = int(n_max * i / 4)
            x = left + chart_w * i / 4
            label = f"{n}"
            tw = gc.GetTextExtent(label)[0]
            gc.DrawText(label, x - tw/2, bottom + 5)
        
        # Y-axis labels
        for i in range(5):
            v = v_min + v_range * (4 - i) / 4
            y = top + chart_h * i / 4
            label = f"{v:.5f}"
            tw = gc.GetTextExtent(label)[0]
            gc.DrawText(label, left - tw - 5, y - 5)
        
        gc.DrawText("Simulations", left + chart_w/2 - 35, h - 12)


# =============================================================================
# Statistics Panel
# =============================================================================

class StatisticsPanel(wx.Panel):
    """Panel showing key statistics in a professional card layout."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.SetBackgroundColour(AnalysisColors.PANEL_BG)
        self.stats = {}
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
    
    def set_stats(self, stats: Dict[str, Any]):
        self.stats = stats
        self.sizer.Clear(True)
        
        # Title
        title = wx.StaticText(self, label="Summary Statistics")
        title.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.sizer.Add(title, 0, wx.ALL, 10)
        
        # Stats grid
        grid = wx.FlexGridSizer(0, 2, 8, 20)
        grid.AddGrowableCol(1, 1)
        
        stat_order = [
            ("Mean (μ)", "mean", ".6f"),
            ("Std Dev (σ)", "std", ".6f"),
            ("Median (50%)", "median", ".6f"),
            ("5th Percentile", "p5", ".6f"),
            ("95th Percentile", "p95", ".6f"),
            ("90% CI Width", "ci_width", ".6f"),
            ("Coefficient of Variation", "cv", ".2%"),
            ("Simulations", "n_sims", "d"),
            ("Converged", "converged", ""),
        ]
        
        for label, key, fmt in stat_order:
            if key in self.stats:
                lbl = wx.StaticText(self, label=label + ":")
                lbl.SetForegroundColour(AnalysisColors.TEXT_SECONDARY)
                grid.Add(lbl, 0, wx.ALIGN_RIGHT)
                
                val = self.stats[key]
                if fmt == "":
                    val_text = "Yes ✓" if val else "No"
                    color = AnalysisColors.SUCCESS if val else AnalysisColors.WARNING
                elif fmt == ".2%":
                    val_text = f"{val*100:.2f}%"
                    color = AnalysisColors.TEXT_PRIMARY
                elif fmt == "d":
                    val_text = f"{int(val):,}"
                    color = AnalysisColors.TEXT_PRIMARY
                else:
                    val_text = f"{val:{fmt}}"
                    color = AnalysisColors.TEXT_PRIMARY
                
                val_lbl = wx.StaticText(self, label=val_text)
                val_lbl.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
                val_lbl.SetForegroundColour(color)
                grid.Add(val_lbl, 0)
        
        self.sizer.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self.Layout()


# =============================================================================
# Main Analysis Dialog
# =============================================================================

class AnalysisDialog(wx.Dialog):
    """
    Industrial-grade analysis dialog with Monte Carlo and Sobol sensitivity.
    Features professional visualization, consistent scales, and publication-quality graphics.
    """
    
    def __init__(self, parent, system_lambda: float, mission_hours: float,
                 component_lambdas: Dict[str, float] = None,
                 sheet_data: Dict[str, Dict] = None,
                 title: str = "System Reliability Analysis"):
        
        display = wx.Display(0)
        rect = display.GetClientArea()
        w = min(1400, int(rect.Width * 0.85))
        h = min(950, int(rect.Height * 0.9))
        
        super().__init__(parent, title=title, size=(w, h),
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        
        self.SetMinSize((1100, 750))
        self.SetBackgroundColour(AnalysisColors.BACKGROUND)
        
        self.system_lambda = system_lambda
        self.mission_hours = mission_hours
        self.component_lambdas = component_lambdas or {}
        self.sheet_data = sheet_data or {}
        
        self.mc_result: Optional[MonteCarloResult] = None
        self.sobol_results: Dict[str, SobolResult] = {}
        self.settings = AnalysisSettings()
        
        self._create_ui()
        self.Centre()
    
    def _create_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header
        header = self._create_header()
        main_sizer.Add(header, 0, wx.EXPAND)
        
        # Notebook for tabs
        self.notebook = wx.Notebook(self)
        
        # Tab 1: Monte Carlo Analysis
        mc_panel = self._create_monte_carlo_tab()
        self.notebook.AddPage(mc_panel, "Monte Carlo Analysis")
        
        # Tab 2: Sensitivity Analysis
        sens_panel = self._create_sensitivity_tab()
        self.notebook.AddPage(sens_panel, "Sensitivity Analysis")
        
        # Tab 3: Component Contributions
        contrib_panel = self._create_contributions_tab()
        self.notebook.AddPage(contrib_panel, "Component Contributions")
        
        # Tab 4: Summary Report
        report_panel = self._create_report_tab()
        self.notebook.AddPage(report_panel, "Summary Report")
        
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 10)
        
        # Footer with buttons
        footer = self._create_footer()
        main_sizer.Add(footer, 0, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(main_sizer)
    
    def _create_header(self) -> wx.Panel:
        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.Colour(38, 50, 56))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        title = wx.StaticText(panel, label="⚡ Reliability Analysis Suite")
        title.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        title.SetForegroundColour(wx.WHITE)
        sizer.Add(title, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)
        
        # System summary
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        summary = wx.StaticText(panel, label=f"System: λ = {self.system_lambda*1e9:.2f} FIT  |  R = {r:.6f}")
        summary.SetForegroundColour(wx.Colour(176, 190, 197))
        sizer.Add(summary, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 12)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_monte_carlo_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(AnalysisColors.BACKGROUND)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Settings bar
        settings_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        settings_sizer.Add(wx.StaticText(panel, label="Simulations:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.mc_n_sims = wx.SpinCtrl(panel, min=1000, max=100000, initial=10000, size=(100, -1))
        settings_sizer.Add(self.mc_n_sims, 0, wx.RIGHT, 15)
        
        settings_sizer.Add(wx.StaticText(panel, label="Uncertainty (%):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.mc_uncertainty = wx.SpinCtrlDouble(panel, min=5, max=100, initial=25, inc=5, size=(80, -1))
        settings_sizer.Add(self.mc_uncertainty, 0, wx.RIGHT, 15)
        
        self.btn_run_mc = wx.Button(panel, label="Run Monte Carlo Analysis")
        self.btn_run_mc.SetBackgroundColour(AnalysisColors.PRIMARY)
        self.btn_run_mc.SetForegroundColour(wx.WHITE)
        self.btn_run_mc.Bind(wx.EVT_BUTTON, self._on_run_monte_carlo)
        settings_sizer.Add(self.btn_run_mc, 0)
        
        main_sizer.Add(settings_sizer, 0, wx.ALL, 15)
        
        # Charts area
        charts_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left: Histogram
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.mc_histogram = HistogramChart(panel, "Reliability Distribution")
        left_sizer.Add(self.mc_histogram, 1, wx.EXPAND)
        
        self.mc_convergence = ConvergenceChart(panel, "Mean Convergence")
        left_sizer.Add(self.mc_convergence, 0, wx.EXPAND | wx.TOP, 10)
        
        charts_sizer.Add(left_sizer, 2, wx.EXPAND | wx.RIGHT, 10)
        
        # Right: Statistics
        self.mc_stats_panel = StatisticsPanel(panel)
        charts_sizer.Add(self.mc_stats_panel, 1, wx.EXPAND)
        
        main_sizer.Add(charts_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 15)
        
        # Results text
        self.mc_results_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.mc_results_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.mc_results_text.SetMinSize((-1, 100))
        main_sizer.Add(self.mc_results_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 15)
        
        panel.SetSizer(main_sizer)
        return panel
    
    def _create_sensitivity_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(AnalysisColors.BACKGROUND)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Settings bar
        settings_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        settings_sizer.Add(wx.StaticText(panel, label="Sobol Samples (N):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.sobol_n = wx.SpinCtrl(panel, min=256, max=8192, initial=1024, size=(100, -1))
        settings_sizer.Add(self.sobol_n, 0, wx.RIGHT, 15)
        
        self.btn_run_sobol = wx.Button(panel, label="Run Sensitivity Analysis")
        self.btn_run_sobol.SetBackgroundColour(AnalysisColors.SECONDARY)
        self.btn_run_sobol.SetForegroundColour(wx.WHITE)
        self.btn_run_sobol.Bind(wx.EVT_BUTTON, self._on_run_sensitivity)
        settings_sizer.Add(self.btn_run_sobol, 0)
        
        main_sizer.Add(settings_sizer, 0, wx.ALL, 15)
        
        # Charts
        charts_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # First-order indices
        self.sobol_first_chart = BarChart(panel, "First-Order Sensitivity (S₁)")
        charts_sizer.Add(self.sobol_first_chart, 1, wx.EXPAND | wx.RIGHT, 10)
        
        # Total-order indices
        self.sobol_total_chart = BarChart(panel, "Total-Order Sensitivity (Sₜ)")
        charts_sizer.Add(self.sobol_total_chart, 1, wx.EXPAND)
        
        main_sizer.Add(charts_sizer, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        
        # Interaction indicator
        self.interaction_panel = wx.Panel(panel)
        self.interaction_panel.SetBackgroundColour(AnalysisColors.PANEL_BG)
        int_sizer = wx.BoxSizer(wx.VERTICAL)
        self.interaction_text = wx.StaticText(self.interaction_panel, label="")
        int_sizer.Add(self.interaction_text, 1, wx.ALL | wx.EXPAND, 10)
        self.interaction_panel.SetSizer(int_sizer)
        main_sizer.Add(self.interaction_panel, 0, wx.EXPAND | wx.ALL, 15)
        
        # Results table
        self.sobol_results_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.sobol_results_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.sobol_results_text.SetMinSize((-1, 150))
        main_sizer.Add(self.sobol_results_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 15)
        
        panel.SetSizer(main_sizer)
        return panel
    
    def _create_contributions_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(AnalysisColors.BACKGROUND)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="Component Failure Rate Contributions")
        title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(title, 0, wx.ALL, 15)
        
        # Contribution chart
        self.contrib_chart = BarChart(panel, "Relative Contribution to System λ")
        main_sizer.Add(self.contrib_chart, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        
        # Contribution table
        self.contrib_list = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SIMPLE)
        self.contrib_list.InsertColumn(0, "Component/Sheet", width=200)
        self.contrib_list.InsertColumn(1, "λ (FIT)", width=100)
        self.contrib_list.InsertColumn(2, "Contribution (%)", width=120)
        self.contrib_list.InsertColumn(3, "Cumulative (%)", width=120)
        main_sizer.Add(self.contrib_list, 1, wx.EXPAND | wx.ALL, 15)
        
        panel.SetSizer(main_sizer)
        self._update_contributions()
        return panel
    
    def _create_report_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(AnalysisColors.BACKGROUND)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="Analysis Summary Report")
        title.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(title, 0, wx.ALL, 15)
        
        self.report_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.report_text.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main_sizer.Add(self.report_text, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 15)
        
        # Export buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_html = wx.Button(panel, label="Export HTML Report")
        btn_html.Bind(wx.EVT_BUTTON, self._on_export_html)
        btn_sizer.Add(btn_html, 0, wx.RIGHT, 10)
        
        btn_csv = wx.Button(panel, label="Export CSV Data")
        btn_csv.Bind(wx.EVT_BUTTON, self._on_export_csv)
        btn_sizer.Add(btn_csv, 0)
        
        main_sizer.Add(btn_sizer, 0, wx.ALL, 15)
        
        panel.SetSizer(main_sizer)
        self._update_report()
        return panel
    
    def _create_footer(self) -> wx.Panel:
        panel = wx.Panel(self)
        panel.SetBackgroundColour(AnalysisColors.BACKGROUND)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.status_text = wx.StaticText(panel, label="Ready. Configure parameters and run analysis.")
        self.status_text.SetForegroundColour(AnalysisColors.TEXT_SECONDARY)
        sizer.Add(self.status_text, 1, wx.ALIGN_CENTER_VERTICAL)
        
        close_btn = wx.Button(panel, label="Close", size=(100, -1))
        close_btn.Bind(wx.EVT_BUTTON, lambda e: self.EndModal(wx.ID_OK))
        sizer.Add(close_btn, 0)
        
        panel.SetSizer(sizer)
        return panel
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def _on_run_monte_carlo(self, event):
        self.status_text.SetLabel("Running Monte Carlo analysis...")
        self.btn_run_mc.Disable()
        wx.Yield()
        
        try:
            n_sims = self.mc_n_sims.GetValue()
            uncertainty = self.mc_uncertainty.GetValue()
            
            self.mc_result = quick_monte_carlo(
                self.system_lambda,
                self.mission_hours,
                uncertainty_percent=uncertainty,
                n_simulations=n_sims
            )
            
            # Update histogram
            self.mc_histogram.set_data(
                self.mc_result.samples,
                {
                    'mean': self.mc_result.mean,
                    'p5': self.mc_result.percentile_5,
                    'p95': self.mc_result.percentile_95
                }
            )
            
            # Update convergence
            self.mc_convergence.set_data(self.mc_result.samples)
            
            # Update statistics
            ci_low, ci_high = self.mc_result.confidence_interval(0.90)
            self.mc_stats_panel.set_stats({
                'mean': self.mc_result.mean,
                'std': self.mc_result.std,
                'median': self.mc_result.percentile_50,
                'p5': self.mc_result.percentile_5,
                'p95': self.mc_result.percentile_95,
                'ci_width': ci_high - ci_low,
                'cv': self.mc_result.std / self.mc_result.mean if self.mc_result.mean > 0 else 0,
                'n_sims': n_sims,
                'converged': self.mc_result.converged
            })
            
            # Update text results
            text = self._format_mc_results()
            self.mc_results_text.SetValue(text)
            
            self.status_text.SetLabel(f"Monte Carlo complete: {n_sims:,} simulations")
            self._update_report()
            
        except Exception as e:
            wx.MessageBox(f"Error: {str(e)}", "Analysis Error", wx.OK | wx.ICON_ERROR)
            self.status_text.SetLabel(f"Error: {str(e)}")
        
        finally:
            self.btn_run_mc.Enable()
    
    def _on_run_sensitivity(self, event):
        if not self.sheet_data:
            wx.MessageBox("No sheet data available for sensitivity analysis.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        
        self.status_text.SetLabel("Running Sobol sensitivity analysis...")
        self.btn_run_sobol.Disable()
        wx.Yield()
        
        try:
            n_samples = self.sobol_n.GetValue()
            
            # Build parameter distributions from sheet lambdas
            parameters = []
            param_names = []
            base_lambdas = {}
            
            for sheet_path, data in self.sheet_data.items():
                lam = data.get('lambda', 0)
                if lam > 0:
                    name = sheet_path.rstrip('/').split('/')[-1] or 'Root'
                    param_names.append(name)
                    base_lambdas[name] = lam
                    # ±30% uniform uncertainty on each sheet lambda
                    parameters.append(ParameterDistribution(
                        name=name,
                        distribution="uniform",
                        params={"low": lam * 0.7, "high": lam * 1.3}
                    ))
            
            if len(parameters) < 2:
                wx.MessageBox("Need at least 2 parameters for sensitivity analysis.", "Insufficient Data", wx.OK | wx.ICON_WARNING)
                return
            
            # Define model: sum of lambdas -> reliability
            def model(params: Dict[str, float]) -> float:
                total_lam = sum(params.values())
                return reliability_from_lambda(total_lam, self.mission_hours)
            
            # Run Sobol analysis
            analyzer = SobolAnalyzer(seed=42)
            results = analyzer.analyze(model, parameters, n_samples=n_samples)
            self.sobol_results = results
            
            # Update first-order chart
            first_data = [(name, res.S_first, res.S_first_ci[0] if res.S_first_ci else 0,
                          res.S_first_ci[1] - res.S_first if res.S_first_ci else 0)
                         for name, res in results.items()]
            self.sobol_first_chart.set_data(first_data, max_value=1.0)
            
            # Update total-order chart
            total_data = [(name, res.S_total, res.S_total_ci[0] if res.S_total_ci else 0,
                          res.S_total_ci[1] - res.S_total if res.S_total_ci else 0)
                         for name, res in results.items()]
            self.sobol_total_chart.set_data(total_data, max_value=1.0)
            
            # Check for interactions
            interactions = []
            for name, res in results.items():
                if res.S_total - res.S_first > 0.05:
                    interactions.append((name, res.S_total - res.S_first))
            
            if interactions:
                int_text = "⚠️ Significant interactions detected:\n"
                for name, diff in sorted(interactions, key=lambda x: -x[1]):
                    int_text += f"  • {name}: Sₜ - S₁ = {diff:.3f}\n"
                self.interaction_text.SetLabel(int_text)
                self.interaction_panel.SetBackgroundColour(wx.Colour(255, 248, 225))
            else:
                self.interaction_text.SetLabel("✓ No significant parameter interactions detected.")
                self.interaction_panel.SetBackgroundColour(wx.Colour(232, 245, 233))
            self.interaction_panel.Refresh()
            
            # Update text results
            text = self._format_sobol_results()
            self.sobol_results_text.SetValue(text)
            
            self.status_text.SetLabel(f"Sensitivity analysis complete: {len(parameters)} parameters")
            self._update_report()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            wx.MessageBox(f"Error: {str(e)}", "Analysis Error", wx.OK | wx.ICON_ERROR)
            self.status_text.SetLabel(f"Error: {str(e)}")
        
        finally:
            self.btn_run_sobol.Enable()
    
    def _update_contributions(self):
        """Update component contribution display."""
        if not self.sheet_data:
            return
        
        # Gather all lambdas
        contributions = []
        total_lambda = 0
        
        for sheet_path, data in self.sheet_data.items():
            lam = data.get('lambda', 0)
            if lam > 0:
                name = sheet_path.rstrip('/').split('/')[-1] or 'Root'
                contributions.append((name, lam))
                total_lambda += lam
        
        if total_lambda == 0:
            return
        
        # Sort by contribution
        contributions.sort(key=lambda x: -x[1])
        
        # Update chart
        chart_data = [(name, lam / total_lambda, 0, 0) for name, lam in contributions]
        self.contrib_chart.set_data(chart_data, max_value=1.0)
        
        # Update list
        self.contrib_list.DeleteAllItems()
        cumulative = 0
        for i, (name, lam) in enumerate(contributions):
            pct = lam / total_lambda * 100
            cumulative += pct
            idx = self.contrib_list.InsertItem(i, name)
            self.contrib_list.SetItem(idx, 1, f"{lam*1e9:.2f}")
            self.contrib_list.SetItem(idx, 2, f"{pct:.1f}%")
            self.contrib_list.SetItem(idx, 3, f"{cumulative:.1f}%")
    
    def _update_report(self):
        """Update summary report text."""
        lines = []
        lines.append("=" * 60)
        lines.append("        RELIABILITY ANALYSIS SUMMARY REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # System info
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        lines.append("SYSTEM PARAMETERS")
        lines.append("-" * 40)
        lines.append(f"  Mission Duration:     {years:.1f} years ({self.mission_hours:.0f} h)")
        lines.append(f"  System Failure Rate:  {self.system_lambda*1e9:.2f} FIT")
        lines.append(f"  Point Estimate R(t):  {r:.6f}")
        lines.append("")
        
        # Monte Carlo results
        if self.mc_result:
            lines.append("MONTE CARLO ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"  Simulations:          {self.mc_result.n_simulations:,}")
            lines.append(f"  Mean Reliability:     {self.mc_result.mean:.6f}")
            lines.append(f"  Standard Deviation:   {self.mc_result.std:.6f}")
            lines.append(f"  90% Confidence Int:   [{self.mc_result.percentile_5:.6f}, {self.mc_result.percentile_95:.6f}]")
            cv = self.mc_result.std / self.mc_result.mean if self.mc_result.mean > 0 else 0
            lines.append(f"  Coeff. of Variation:  {cv*100:.2f}%")
            lines.append("")
        
        # Sensitivity results
        if self.sobol_results:
            lines.append("SENSITIVITY ANALYSIS (Sobol Indices)")
            lines.append("-" * 40)
            lines.append(f"  {'Parameter':<20} {'S₁':>10} {'Sₜ':>10} {'Interact.':>12}")
            lines.append("  " + "-" * 54)
            for name, res in sorted(self.sobol_results.items(), key=lambda x: -x[1].S_total):
                interact = res.S_total - res.S_first
                flag = "  ⚠️" if interact > 0.05 else ""
                lines.append(f"  {name:<20} {res.S_first:>10.4f} {res.S_total:>10.4f} {interact:>10.4f}{flag}")
            lines.append("")
        
        # Contributions
        if self.sheet_data:
            lines.append("FAILURE RATE CONTRIBUTIONS")
            lines.append("-" * 40)
            contributions = []
            total = 0
            for path, data in self.sheet_data.items():
                lam = data.get('lambda', 0)
                if lam > 0:
                    name = path.rstrip('/').split('/')[-1] or 'Root'
                    contributions.append((name, lam))
                    total += lam
            
            if total > 0:
                for name, lam in sorted(contributions, key=lambda x: -x[1])[:10]:
                    pct = lam / total * 100
                    lines.append(f"  {name:<25} {lam*1e9:>8.2f} FIT  ({pct:>5.1f}%)")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("  Report generated by KiCad Reliability Calculator v2.0.0")
        lines.append("  Analysis based on IEC TR 62380")
        lines.append("=" * 60)
        
        self.report_text.SetValue("\n".join(lines))
    
    def _format_mc_results(self) -> str:
        if not self.mc_result:
            return ""
        r = self.mc_result
        lines = [
            f"Monte Carlo Results ({r.n_simulations:,} simulations)",
            f"{'='*50}",
            f"Mean:        {r.mean:.6f}",
            f"Std Dev:     {r.std:.6f}",
            f"Median:      {r.percentile_50:.6f}",
            f"5th %ile:    {r.percentile_5:.6f}",
            f"95th %ile:   {r.percentile_95:.6f}",
            f"90% CI:      [{r.percentile_5:.6f}, {r.percentile_95:.6f}]",
        ]
        return "\n".join(lines)
    
    def _format_sobol_results(self) -> str:
        if not self.sobol_results:
            return ""
        
        lines = [
            "Sobol Sensitivity Indices",
            "=" * 65,
            f"{'Parameter':<20} {'S₁ (First)':>12} {'Sₜ (Total)':>12} {'Interaction':>12}",
            "-" * 65,
        ]
        
        for name, res in sorted(self.sobol_results.items(), key=lambda x: -x[1].S_total):
            interact = res.S_total - res.S_first
            flag = " ⚠️" if interact > 0.05 else ""
            lines.append(f"{name:<20} {res.S_first:>12.4f} {res.S_total:>12.4f} {interact:>12.4f}{flag}")
        
        lines.append("-" * 65)
        lines.append("S₁: First-order sensitivity (main effect)")
        lines.append("Sₜ: Total-order sensitivity (includes interactions)")
        lines.append("⚠️: Significant interaction (Sₜ - S₁ > 0.05)")
        
        return "\n".join(lines)
    
    def _on_export_html(self, event):
        dlg = wx.FileDialog(self, "Export HTML Report", wildcard="HTML (*.html)|*.html",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            html = self._generate_html_report()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            self.status_text.SetLabel(f"Exported: {path}")
        dlg.Destroy()
    
    def _on_export_csv(self, event):
        dlg = wx.FileDialog(self, "Export CSV Data", wildcard="CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            csv = self._generate_csv_report()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(csv)
            self.status_text.SetLabel(f"Exported: {path}")
        dlg.Destroy()
    
    def _generate_html_report(self) -> str:
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        
        html = f"""<!DOCTYPE html>
<html><head>
<title>Reliability Analysis Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; 
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
h1 {{ color: #1a237e; border-bottom: 3px solid #1e88e5; padding-bottom: 10px; }}
h2 {{ color: #37474f; }}
.metric {{ font-size: 28px; font-weight: bold; color: #1565c0; }}
.metric-label {{ color: #78909c; font-size: 14px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #e0e0e0; padding: 10px; text-align: left; }}
th {{ background: #f5f5f5; font-weight: 600; }}
.bar {{ height: 20px; background: linear-gradient(90deg, #1e88e5, #42a5f5); border-radius: 3px; }}
.warn {{ color: #f57c00; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
</style></head><body>
<h1>⚡ Reliability Analysis Report</h1>
<p style="color: #78909c;">IEC TR 62380 Analysis | Generated by KiCad Reliability Calculator v2.0.0</p>

<div class="card">
<h2>System Summary</h2>
<div class="grid">
<div><div class="metric-label">Mission Duration</div><div class="metric">{years:.1f} years</div></div>
<div><div class="metric-label">Failure Rate</div><div class="metric">{self.system_lambda*1e9:.2f} FIT</div></div>
<div><div class="metric-label">Point Estimate R(t)</div><div class="metric">{r:.6f}</div></div>
</div></div>
"""
        if self.mc_result:
            mc = self.mc_result
            html += f"""
<div class="card">
<h2>Monte Carlo Analysis</h2>
<div class="grid">
<div><div class="metric-label">Mean Reliability</div><div class="metric">{mc.mean:.6f}</div></div>
<div><div class="metric-label">Standard Deviation</div><div class="metric">{mc.std:.6f}</div></div>
<div><div class="metric-label">90% Confidence Interval</div><div class="metric">[{mc.percentile_5:.6f}, {mc.percentile_95:.6f}]</div></div>
<div><div class="metric-label">Simulations</div><div class="metric">{mc.n_simulations:,}</div></div>
</div></div>
"""
        
        if self.sobol_results:
            html += """
<div class="card">
<h2>Sensitivity Analysis (Sobol Indices)</h2>
<table>
<tr><th>Parameter</th><th>S₁ (First-Order)</th><th>Sₜ (Total-Order)</th><th>Interactions</th><th>Visual</th></tr>
"""
            for name, res in sorted(self.sobol_results.items(), key=lambda x: -x[1].S_total):
                interact = res.S_total - res.S_first
                warn = ' class="warn"' if interact > 0.05 else ''
                bar_w = min(100, res.S_total * 100)
                html += f"""<tr>
<td>{name}</td>
<td>{res.S_first:.4f}</td>
<td>{res.S_total:.4f}</td>
<td{warn}>{interact:.4f}</td>
<td><div class="bar" style="width: {bar_w}%"></div></td>
</tr>"""
            html += "</table></div>"
        
        html += "</body></html>"
        return html
    
    def _generate_csv_report(self) -> str:
        lines = ["Type,Parameter,Value"]
        
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        lines.append(f"System,Lambda_FIT,{self.system_lambda*1e9:.6f}")
        lines.append(f"System,Reliability,{r:.6f}")
        lines.append(f"System,Mission_Hours,{self.mission_hours:.0f}")
        
        if self.mc_result:
            mc = self.mc_result
            lines.append(f"MonteCarlo,Mean,{mc.mean:.6f}")
            lines.append(f"MonteCarlo,StdDev,{mc.std:.6f}")
            lines.append(f"MonteCarlo,Percentile_5,{mc.percentile_5:.6f}")
            lines.append(f"MonteCarlo,Percentile_95,{mc.percentile_95:.6f}")
            lines.append(f"MonteCarlo,N_Simulations,{mc.n_simulations}")
        
        if self.sobol_results:
            for name, res in self.sobol_results.items():
                lines.append(f"Sobol_S1,{name},{res.S_first:.6f}")
                lines.append(f"Sobol_ST,{name},{res.S_total:.6f}")
        
        return "\n".join(lines)
    
    def get_results(self) -> SystemAnalysisResult:
        """Return all analysis results."""
        return SystemAnalysisResult(
            system_lambda=self.system_lambda,
            system_reliability=reliability_from_lambda(self.system_lambda, self.mission_hours),
            mission_hours=self.mission_hours,
            monte_carlo=self.mc_result,
            sensitivity=self.sobol_results if self.sobol_results else None,
        )
