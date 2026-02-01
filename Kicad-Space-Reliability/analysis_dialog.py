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

from .monte_carlo import MonteCarloResult, quick_monte_carlo
from .sensitivity_analysis import SobolResult, SobolAnalyzer
from .reliability_math import reliability_from_lambda


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
    PRIMARY = wx.Colour(37, 99, 235)     # Blue
    SUCCESS = wx.Colour(34, 197, 94)     # Green
    WARNING = wx.Colour(245, 158, 11)    # Amber
    DANGER = wx.Colour(239, 68, 68)      # Red
    
    # Chart colors (colorblind-safe)
    CHART = [
        wx.Colour(59, 130, 246),   # Blue
        wx.Colour(16, 185, 129),   # Green
        wx.Colour(245, 158, 11),   # Amber
        wx.Colour(239, 68, 68),    # Red
        wx.Colour(139, 92, 246),   # Purple
        wx.Colour(236, 72, 153),   # Pink
        wx.Colour(20, 184, 166),   # Teal
        wx.Colour(249, 115, 22),   # Orange
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
        dc.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 16, 12)
        
        if self.samples is None or len(self.samples) == 0:
            dc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            dc.SetTextForeground(Colors.TEXT_LIGHT)
            dc.DrawText("Run analysis to see distribution", w//2 - 100, h//2)
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
                dc.DrawRectangle(int(x), margin_t + chart_h - bar_h, int(bar_width), bar_h)
        
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
        dc.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
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
        
        dc.SetFont(wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
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
    
    def set_data(self, data: List[Tuple[str, float]], max_value: float = None, x_label: str = "Value"):
        """Set data as list of (name, value) tuples."""
        self.data = [(name, val, i % len(Colors.CHART)) for i, (name, val) in enumerate(data)]
        self.max_value = max_value if max_value else (max(d[1] for d in self.data) if self.data else 1.0)
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
        dc.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(Colors.TEXT_DARK)
        dc.DrawText(self.title, 16, 12)
        
        if not self.data:
            dc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            dc.SetTextForeground(Colors.TEXT_LIGHT)
            dc.DrawText("No data available", w//2 - 50, h//2)
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
        dc.SetFont(wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
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
        dc.SetFont(wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
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
        
        dc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
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
        dc.SetFont(wx.Font(8, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
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
            
            val_lbl = wx.StaticText(self, label=text)
            val_lbl.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
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
    
    def __init__(self, parent, system_lambda: float, mission_hours: float,
                 sheet_data: Dict[str, Dict] = None,
                 block_structure: Dict = None,
                 title: str = "Reliability Analysis Suite"):
        
        display = wx.Display(0)
        rect = display.GetClientArea()
        w = min(1350, int(rect.Width * 0.85))
        h = min(900, int(rect.Height * 0.88))
        
        super().__init__(parent, title=title, size=(w, h),
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        
        self.SetMinSize((1000, 700))
        self.SetBackgroundColour(Colors.BG_LIGHT)
        
        self.system_lambda = system_lambda
        self.mission_hours = mission_hours
        self.sheet_data = sheet_data or {}
        self.block_structure = block_structure or {}
        
        self.mc_result: Optional[MonteCarloResult] = None
        self.sobol_result: Optional[SobolResult] = None
        
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
        title.SetFont(wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        title.SetForegroundColour(Colors.TEXT_WHITE)
        sizer.Add(title, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 14)
        
        # System info
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        info = wx.StaticText(panel, label=f"Lambda = {self.system_lambda*1e9:.2f} FIT  |  R = {r:.6f}  |  {years:.1f} years")
        info.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
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
        
        ctrl.Add(wx.StaticText(ctrl_panel, label="Simulations:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.mc_n = wx.SpinCtrl(ctrl_panel, min=1000, max=100000, initial=10000, size=(100, -1))
        ctrl.Add(self.mc_n, 0, wx.ALL, 8)
        
        ctrl.Add(wx.StaticText(ctrl_panel, label="Uncertainty (%):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.mc_unc = wx.SpinCtrlDouble(ctrl_panel, min=5, max=100, initial=25, inc=5, size=(80, -1))
        ctrl.Add(self.mc_unc, 0, wx.ALL, 8)
        
        self.btn_mc = wx.Button(ctrl_panel, label="Run Monte Carlo")
        self.btn_mc.SetBackgroundColour(Colors.PRIMARY)
        self.btn_mc.SetForegroundColour(Colors.TEXT_WHITE)
        self.btn_mc.Bind(wx.EVT_BUTTON, self._on_run_mc)
        ctrl.Add(self.btn_mc, 0, wx.ALL, 8)
        
        ctrl_panel.SetSizer(ctrl)
        main.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 8)
        
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
        
        ctrl.Add(wx.StaticText(ctrl_panel, label="Samples (N):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)
        self.sobol_n = wx.SpinCtrl(ctrl_panel, min=256, max=4096, initial=1024, size=(100, -1))
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
        self.interaction_info = wx.StaticText(panel, label="Run analysis to see parameter sensitivities and interactions.")
        self.interaction_info.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.interaction_info.SetForegroundColour(Colors.TEXT_MEDIUM)
        main.Add(self.interaction_info, 0, wx.ALL, 12)
        
        panel.SetSizer(main)
        return panel
    
    def _create_contributions_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)
        
        # Chart
        self.contrib_chart = HorizontalBarPanel(panel, "Failure Rate Contributions")
        main.Add(self.contrib_chart, 1, wx.EXPAND | wx.ALL, 8)
        
        # Table
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
    
    def _create_report_tab(self) -> wx.Panel:
        panel = wx.Panel(self.notebook)
        panel.SetBackgroundColour(Colors.BG_LIGHT)
        main = wx.BoxSizer(wx.VERTICAL)
        
        # Report text
        self.report_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE)
        self.report_text.SetBackgroundColour(Colors.BG_WHITE)
        self.report_text.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
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
    
    def _on_run_mc(self, event):
        self.status.SetLabel("Running Monte Carlo...")
        self.btn_mc.Disable()
        wx.Yield()
        
        try:
            n = self.mc_n.GetValue()
            unc = self.mc_unc.GetValue()
            
            self.mc_result = quick_monte_carlo(
                self.system_lambda, self.mission_hours,
                uncertainty_percent=unc, n_simulations=n
            )
            
            # Update histogram
            self.histogram.set_data(
                self.mc_result.samples,
                self.mc_result.mean,
                self.mc_result.percentile_5,
                self.mc_result.percentile_95
            )
            
            # Update convergence
            self.convergence.set_data(self.mc_result.samples)
            
            # Update stats
            ci_lo, ci_hi = self.mc_result.confidence_interval(0.90)
            self.stats_card.set_stats({
                'mean': self.mc_result.mean,
                'std': self.mc_result.std,
                'median': self.mc_result.percentile_50,
                'p5': self.mc_result.percentile_5,
                'p95': self.mc_result.percentile_95,
                'ci_width': ci_hi - ci_lo,
                'cv': self.mc_result.std / self.mc_result.mean if self.mc_result.mean > 0 else 0,
                'n_sims': n,
                'converged': self.mc_result.converged,
            })
            
            self.status.SetLabel(f"Monte Carlo complete: {n:,} simulations")
            self._update_report()
            
        except Exception as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
            self.status.SetLabel(f"Error: {e}")
        finally:
            self.btn_mc.Enable()
    
    def _on_run_sobol(self, event):
        if not self.sheet_data:
            wx.MessageBox("No sheet data for sensitivity analysis.", "No Data", wx.OK | wx.ICON_WARNING)
            return
        
        self.status.SetLabel("Running Sobol analysis...")
        self.btn_sobol.Disable()
        wx.Yield()
        
        try:
            n = self.sobol_n.GetValue()
            
            # Build parameter bounds from sheet lambdas (Dict not list!)
            param_bounds = {}
            for path, data in self.sheet_data.items():
                lam = data.get('lambda', 0)
                if lam > 0:
                    name = path.rstrip('/').split('/')[-1] or 'Root'
                    # Ensure unique names
                    base_name = name
                    i = 1
                    while name in param_bounds:
                        name = f"{base_name}_{i}"
                        i += 1
                    param_bounds[name] = (lam * 0.7, lam * 1.3)
            
            if len(param_bounds) < 2:
                wx.MessageBox("Need at least 2 sheets with non-zero lambda for sensitivity analysis.",
                             "Insufficient Data", wx.OK | wx.ICON_WARNING)
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
            first_data = list(zip(self.sobol_result.parameter_names, self.sobol_result.S_first))
            first_data.sort(key=lambda x: -x[1])
            self.sobol_first_chart.set_data(first_data, max_value=1.0, x_label="First-Order Index")
            
            total_data = list(zip(self.sobol_result.parameter_names, self.sobol_result.S_total))
            total_data.sort(key=lambda x: -x[1])
            self.sobol_total_chart.set_data(total_data, max_value=1.0, x_label="Total-Order Index")
            
            # Interaction info
            significant = [self.sobol_result.parameter_names[i] for i in self.sobol_result.significant_interactions]
            
            if significant:
                info = f"Warning: Significant interactions detected in: {', '.join(significant)}"
                self.interaction_info.SetForegroundColour(Colors.WARNING)
            else:
                info = "OK: No significant parameter interactions detected."
                self.interaction_info.SetForegroundColour(Colors.SUCCESS)
            self.interaction_info.SetLabel(info)
            
            self.status.SetLabel(f"Sensitivity analysis complete: {len(param_bounds)} parameters")
            self._update_report()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            wx.MessageBox(str(e), "Analysis Error", wx.OK | wx.ICON_ERROR)
            self.status.SetLabel(f"Error: {e}")
        finally:
            self.btn_sobol.Enable()
    
    def _update_contributions(self):
        if not self.sheet_data:
            return
        
        # Gather contributions
        contribs = []
        total_lam = 0
        for path, data in self.sheet_data.items():
            lam = data.get('lambda', 0)
            if lam > 0:
                name = path.rstrip('/').split('/')[-1] or 'Root'
                contribs.append((name, lam))
                total_lam += lam
        
        if total_lam == 0:
            return
        
        contribs.sort(key=lambda x: -x[1])
        
        # Update chart
        chart_data = [(name, lam / total_lam) for name, lam in contribs]
        self.contrib_chart.set_data(chart_data, max_value=1.0, x_label="Relative Contribution")
        
        # Update list
        self.contrib_list.DeleteAllItems()
        cumulative = 0
        for i, (name, lam) in enumerate(contribs):
            pct = lam / total_lam * 100
            cumulative += pct
            idx = self.contrib_list.InsertItem(i, name)
            self.contrib_list.SetItem(idx, 1, f"{lam*1e9:.2f}")
            self.contrib_list.SetItem(idx, 2, f"{pct:.1f}%")
            self.contrib_list.SetItem(idx, 3, f"{cumulative:.1f}%")
    
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
        lines.append(f"  Mission Duration:      {years:.2f} years ({self.mission_hours:.0f} hours)")
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
            
            ranked = sorted(zip(sr.parameter_names, sr.S_first, sr.S_total), key=lambda x: -x[2])
            for name, s1, st in ranked:
                interact = st - s1
                flag = "  *" if interact > 0.1 * st and st > 0.01 else ""
                lines.append(f"  {name:<25} {s1:>10.4f} {st:>10.4f} {interact:>10.4f}{flag}")
            
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
                lam = data.get('lambda', 0)
                if lam > 0:
                    name = path.rstrip('/').split('/')[-1] or 'Root'
                    contribs.append((name, lam, path))
                    total_lam += lam
            
            contribs.sort(key=lambda x: -x[1])
            
            if total_lam > 0:
                cumul = 0
                for name, lam, path in contribs[:20]:
                    pct = lam / total_lam * 100
                    cumul += pct
                    lines.append(f"  {name:<25} {lam*1e9:>8.2f} FIT  ({pct:>5.1f}%)  cum: {cumul:>5.1f}%")
                
                if len(contribs) > 20:
                    lines.append(f"  ... and {len(contribs) - 20} more")
            lines.append("")
        
        # Component details
        if self.sheet_data:
            lines.append("COMPONENT DETAILS BY SHEET")
            lines.append("-" * 50)
            
            for path in sorted(self.sheet_data.keys()):
                data = self.sheet_data[path]
                sheet_name = path.rstrip('/').split('/')[-1] or 'Root'
                sheet_lam = data.get('lambda', 0)
                sheet_r = data.get('r', 1)
                
                lines.append(f"")
                lines.append(f"  [{sheet_name}]")
                lines.append(f"  Path: {path}")
                lines.append(f"  Sheet Lambda: {sheet_lam*1e9:.2f} FIT, R: {sheet_r:.6f}")
                
                components = data.get('components', [])
                if components:
                    lines.append(f"  Components ({len(components)}):")
                    for c in components[:15]:
                        ref = c.get('ref', '?')
                        val = c.get('value', '')[:15]
                        cls = c.get('class', '')[:20]
                        c_lam = c.get('lambda', 0)
                        c_r = c.get('r', 1)
                        lines.append(f"    {ref:<8} {val:<15} {cls:<20} L={c_lam*1e9:>6.2f} FIT  R={c_r:.6f}")
                    if len(components) > 15:
                        lines.append(f"    ... and {len(components) - 15} more")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("  Generated by KiCad Reliability Calculator v2.0.0")
        lines.append("  IEC TR 62380 Methodology")
        lines.append("=" * 70)
        
        self.report_text.SetValue("\n".join(lines))
    
    def _on_export_html(self, event):
        dlg = wx.FileDialog(self, "Export HTML", wildcard="HTML (*.html)|*.html",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            html = self._generate_html()
            with open(dlg.GetPath(), 'w', encoding='utf-8') as f:
                f.write(html)
            self.status.SetLabel(f"Exported: {dlg.GetPath()}")
        dlg.Destroy()
    
    def _on_export_csv(self, event):
        dlg = wx.FileDialog(self, "Export CSV", wildcard="CSV (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            csv = self._generate_csv()
            with open(dlg.GetPath(), 'w', encoding='utf-8') as f:
                f.write(csv)
            self.status.SetLabel(f"Exported: {dlg.GetPath()}")
        dlg.Destroy()
    
    def _generate_html(self) -> str:
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        years = self.mission_hours / (365 * 24)
        
        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Reliability Analysis Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
       max-width: 1100px; margin: 0 auto; padding: 24px; background: #f8fafc; color: #1e293b; }}
h1 {{ color: #1e40af; margin-bottom: 8px; }}
h2 {{ color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 32px; }}
h3 {{ color: #475569; margin-top: 16px; }}
.subtitle {{ color: #64748b; margin-bottom: 24px; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; 
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.metric {{ font-size: 24px; font-weight: 700; color: #2563eb; }}
.metric-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
tr:hover {{ background: #f8fafc; }}
.mono {{ font-family: 'SF Mono', Monaco, monospace; font-size: 13px; }}
.bar {{ height: 8px; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 4px; }}
.warn {{ color: #f59e0b; }}
</style></head><body>
<h1>Reliability Analysis Report</h1>
<p class="subtitle">IEC TR 62380 Methodology - KiCad Reliability Calculator v2.0.0</p>

<div class="card">
<h2 style="margin-top: 0;">System Summary</h2>
<div class="grid">
<div><div class="metric-label">Mission Duration</div><div class="metric">{years:.1f} years</div></div>
<div><div class="metric-label">Failure Rate</div><div class="metric">{self.system_lambda*1e9:.2f} FIT</div></div>
<div><div class="metric-label">Reliability R(t)</div><div class="metric">{r:.6f}</div></div>
</div></div>
"""
        
        if self.mc_result:
            mc = self.mc_result
            ci_lo, ci_hi = mc.confidence_interval(0.90)
            html += f"""
<div class="card">
<h2 style="margin-top: 0;">Monte Carlo Analysis</h2>
<div class="grid">
<div><div class="metric-label">Mean</div><div class="metric">{mc.mean:.6f}</div></div>
<div><div class="metric-label">Std Dev</div><div class="metric">{mc.std:.6f}</div></div>
<div><div class="metric-label">90% CI</div><div class="metric mono">[{ci_lo:.6f}, {ci_hi:.6f}]</div></div>
<div><div class="metric-label">Simulations</div><div class="metric">{mc.n_simulations:,}</div></div>
</div></div>
"""
        
        if self.sobol_result:
            sr = self.sobol_result
            html += """<div class="card"><h2 style="margin-top: 0;">Sensitivity Analysis</h2>
<table><tr><th>Parameter</th><th>S1 (First)</th><th>ST (Total)</th><th>Interaction</th><th></th></tr>"""
            ranked = sorted(zip(sr.parameter_names, sr.S_first, sr.S_total), key=lambda x: -x[2])
            for name, s1, st in ranked:
                interact = st - s1
                warn = ' class="warn"' if interact > 0.1 * st and st > 0.01 else ''
                bar_w = min(100, st * 100)
                html += f"""<tr><td>{name}</td><td class="mono">{s1:.4f}</td><td class="mono">{st:.4f}</td>
<td{warn} class="mono">{interact:.4f}</td><td><div class="bar" style="width:{bar_w}%"></div></td></tr>"""
            html += "</table></div>"
        
        # Contributions
        if self.sheet_data:
            html += """<div class="card"><h2 style="margin-top: 0;">Failure Rate Contributions</h2>
<table><tr><th>Sheet</th><th>Lambda (FIT)</th><th>Contribution</th><th></th></tr>"""
            
            contribs = []
            total = sum(d.get('lambda', 0) for d in self.sheet_data.values())
            for path, data in self.sheet_data.items():
                lam = data.get('lambda', 0)
                if lam > 0:
                    name = path.rstrip('/').split('/')[-1] or 'Root'
                    contribs.append((name, lam))
            
            for name, lam in sorted(contribs, key=lambda x: -x[1])[:15]:
                pct = lam / total * 100 if total > 0 else 0
                bar_w = min(100, pct)
                html += f"""<tr><td>{name}</td><td class="mono">{lam*1e9:.2f}</td>
<td>{pct:.1f}%</td><td><div class="bar" style="width:{bar_w}%"></div></td></tr>"""
            html += "</table></div>"
        
        # Component details
        if self.sheet_data:
            html += """<h2>Component Details</h2>"""
            for path in sorted(self.sheet_data.keys()):
                data = self.sheet_data[path]
                name = path.rstrip('/').split('/')[-1] or 'Root'
                lam = data.get('lambda', 0)
                r_val = data.get('r', 1)
                comps = data.get('components', [])
                
                html += f"""<div class="card"><h3 style="margin-top:0">{name}</h3>
<p><strong>Path:</strong> {path}</p>
<p><strong>Lambda:</strong> {lam*1e9:.2f} FIT | <strong>R:</strong> {r_val:.6f}</p>"""
                
                if comps:
                    html += """<table><tr><th>Ref</th><th>Value</th><th>Type</th><th>Lambda (FIT)</th><th>R</th></tr>"""
                    for c in comps[:30]:
                        html += f"""<tr><td>{c.get('ref','?')}</td><td>{c.get('value','')[:20]}</td>
<td>{c.get('class','')[:25]}</td><td class="mono">{c.get('lambda',0)*1e9:.2f}</td>
<td class="mono">{c.get('r',1):.6f}</td></tr>"""
                    if len(comps) > 30:
                        html += f"<tr><td colspan='5'>... and {len(comps)-30} more</td></tr>"
                    html += "</table>"
                html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _generate_csv(self) -> str:
        lines = ["Section,Item,Parameter,Value"]
        
        r = reliability_from_lambda(self.system_lambda, self.mission_hours)
        lines.append(f"System,Summary,Lambda_FIT,{self.system_lambda*1e9:.6f}")
        lines.append(f"System,Summary,Reliability,{r:.6f}")
        lines.append(f"System,Summary,Mission_Hours,{self.mission_hours:.0f}")
        
        if self.mc_result:
            mc = self.mc_result
            lines.append(f"MonteCarlo,Summary,Mean,{mc.mean:.6f}")
            lines.append(f"MonteCarlo,Summary,StdDev,{mc.std:.6f}")
            lines.append(f"MonteCarlo,Summary,P5,{mc.percentile_5:.6f}")
            lines.append(f"MonteCarlo,Summary,P95,{mc.percentile_95:.6f}")
        
        if self.sobol_result:
            for i, name in enumerate(self.sobol_result.parameter_names):
                lines.append(f"Sensitivity,{name},S_first,{self.sobol_result.S_first[i]:.6f}")
                lines.append(f"Sensitivity,{name},S_total,{self.sobol_result.S_total[i]:.6f}")
        
        for path, data in self.sheet_data.items():
            name = path.rstrip('/').split('/')[-1] or 'Root'
            lines.append(f"Sheet,{name},Path,{path}")
            lines.append(f"Sheet,{name},Lambda_FIT,{data.get('lambda',0)*1e9:.6f}")
            lines.append(f"Sheet,{name},Reliability,{data.get('r',1):.6f}")
            for c in data.get('components', []):
                ref = c.get('ref', '?')
                lines.append(f"Component,{ref},Sheet,{name}")
                lines.append(f"Component,{ref},Value,{c.get('value', '')}")
                lines.append(f"Component,{ref},Class,{c.get('class', '')}")
                lines.append(f"Component,{ref},Lambda_FIT,{c.get('lambda',0)*1e9:.6f}")
                lines.append(f"Component,{ref},Reliability,{c.get('r',1):.6f}")
        
        return "\n".join(lines)
