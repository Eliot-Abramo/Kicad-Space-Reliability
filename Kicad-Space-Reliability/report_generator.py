"""
Enhanced Report Generator with Inline SVG Charts
=================================================
Professional reliability reports with embedded interactive visualizations.
Supports tornado charts, design margin scenarios, configurable CI,
and component override display.

Author:  Eliot Abramo
"""

import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import base64


@dataclass
class ReportData:
    """Container for all report data."""
    project_name: str
    mission_hours: float
    mission_years: float
    n_cycles: int
    delta_t: float

    system_reliability: float
    system_lambda: float
    system_mttf_hours: float

    sheets: Dict[str, Dict]
    blocks: List[Dict]

    monte_carlo: Optional[Dict] = None
    sensitivity: Optional[Dict] = None
    sheet_mc: Optional[Dict] = None
    criticality: Optional[List] = None
    tornado: Optional[Dict] = None
    design_margin: Optional[Dict] = None

    # v3.1.0 co-design fields
    mission_profile: Optional[Dict] = None
    budget: Optional[Dict] = None
    derating: Optional[Dict] = None
    swap_analysis: Optional[Dict] = None
    growth_timeline: Optional[Dict] = None
    correlated_mc: Optional[Dict] = None

    generated_at: str = None

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


# === SVG Chart Generators ===

def _svg_histogram(samples: list, mean: float, p5: float, p95: float,
                   width: int = 600, height: int = 300, title: str = "Distribution",
                   ci_lower: float = None, ci_upper: float = None,
                   ci_label: str = "90% CI") -> str:
    if not samples or len(samples) < 2:
        return ""
    n_bins = 35
    s_min, s_max = min(samples), max(samples)
    s_range = s_max - s_min
    if s_range < 1e-15:
        s_range = 1e-6
    bin_width = s_range / n_bins
    bins = [0] * n_bins
    for v in samples:
        idx = min(int((v - s_min) / bin_width), n_bins - 1)
        bins[idx] += 1
    max_count = max(bins) if max(bins) > 0 else 1

    ml, mr, mt, mb = 60, 30, 40, 50
    cw, ch = width - ml - mr, height - mt - mb

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="24" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>\n'

    for i in range(5):
        y = mt + ch * i // 4
        svg += f'<line x1="{ml}" y1="{y}" x2="{ml+cw}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>\n'

    bw = max(1, cw // n_bins - 1)
    for i, cnt in enumerate(bins):
        if cnt > 0:
            x = ml + i * cw // n_bins
            bh = int((cnt / max_count) * ch)
            svg += f'<rect x="{x}" y="{mt+ch-bh}" width="{bw}" height="{bh}" fill="#3b82f6" rx="1"/>\n'

    def vx(v):
        return ml + (v - s_min) / s_range * cw

    # CI band (shaded area)
    if ci_lower is not None and ci_upper is not None:
        x1, x2 = vx(ci_lower), vx(ci_upper)
        svg += f'<rect x="{x1:.1f}" y="{mt}" width="{x2-x1:.1f}" height="{ch}" fill="#22c55e" opacity="0.12"/>\n'
        svg += f'<line x1="{x1:.1f}" y1="{mt}" x2="{x1:.1f}" y2="{mt+ch}" stroke="#22c55e" stroke-width="2" stroke-dasharray="5,3"/>\n'
        svg += f'<line x1="{x2:.1f}" y1="{mt}" x2="{x2:.1f}" y2="{mt+ch}" stroke="#22c55e" stroke-width="2" stroke-dasharray="5,3"/>\n'

    # Mean line
    xm = vx(mean)
    svg += f'<line x1="{xm:.1f}" y1="{mt}" x2="{xm:.1f}" y2="{mt+ch}" stroke="#ef4444" stroke-width="2"/>\n'

    # Percentile lines
    for pv in [p5, p95]:
        xp = vx(pv)
        svg += f'<line x1="{xp:.1f}" y1="{mt}" x2="{xp:.1f}" y2="{mt+ch}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,3"/>\n'

    # Legend
    lx = width - mr - 160
    ly = mt + 5
    svg += f'<line x1="{lx}" y1="{ly+6}" x2="{lx+20}" y2="{ly+6}" stroke="#ef4444" stroke-width="2"/>\n'
    svg += f'<text x="{lx+25}" y="{ly+10}" font-size="10" fill="#475569">Mean</text>\n'
    svg += f'<line x1="{lx}" y1="{ly+20}" x2="{lx+20}" y2="{ly+20}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,3"/>\n'
    svg += f'<text x="{lx+25}" y="{ly+24}" font-size="10" fill="#475569">5th/95th %ile</text>\n'
    if ci_lower is not None:
        svg += f'<rect x="{lx}" y="{ly+30}" width="20" height="10" fill="#22c55e" opacity="0.3" rx="2"/>\n'
        svg += f'<text x="{lx+25}" y="{ly+39}" font-size="10" fill="#475569">{ci_label}</text>\n'

    # X-axis
    for i in range(5):
        val = s_min + s_range * i / 4
        x = ml + cw * i // 4
        svg += f'<text x="{x}" y="{mt+ch+20}" font-size="9" text-anchor="middle" fill="#64748b">{val:.4f}</text>\n'
    svg += f'<text x="{ml+cw//2}" y="{height-5}" font-size="10" text-anchor="middle" fill="#64748b">Reliability R(t)</text>\n'
    svg += '</svg>\n'
    return svg


def _svg_convergence(samples: list, width: int = 600, height: int = 300, title: str = "Convergence") -> str:
    if not samples or len(samples) < 10:
        return ""
    import numpy as np
    arr = np.array(samples)
    cumsum = np.cumsum(arr)
    running_mean = cumsum / np.arange(1, len(arr) + 1)
    step = max(1, len(running_mean) // 80)
    pts = [(i, running_mean[i]) for i in range(0, len(running_mean), step)]
    if len(running_mean) - 1 not in [p[0] for p in pts]:
        pts.append((len(running_mean) - 1, running_mean[-1]))
    vals = [p[1] for p in pts]
    v_min, v_max = min(vals), max(vals)
    vr = v_max - v_min
    if vr < 1e-12:
        vr = abs(v_max) * 0.1 if v_max != 0 else 0.01
    v_min -= vr * 0.1
    v_max += vr * 0.1
    vr = v_max - v_min
    n_max = len(arr)
    ml, mr, mt, mb = 60, 30, 40, 50
    cw, ch = width - ml - mr, height - mt - mb
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="24" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>\n'
    for i in range(5):
        y = mt + ch * i // 4
        svg += f'<line x1="{ml}" y1="{y}" x2="{ml+cw}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>\n'
    path_d = ""
    for n, v in pts:
        x = ml + (n / n_max) * cw
        y = mt + ch - ((v - v_min) / vr) * ch
        if not path_d:
            path_d = f"M{x:.1f},{y:.1f}"
        else:
            path_d += f" L{x:.1f},{y:.1f}"
    svg += f'<path d="{path_d}" fill="none" stroke="#3b82f6" stroke-width="2"/>\n'
    fy = mt + ch - ((running_mean[-1] - v_min) / vr) * ch
    svg += f'<line x1="{ml}" y1="{fy:.1f}" x2="{ml+cw}" y2="{fy:.1f}" stroke="#22c55e" stroke-width="1" stroke-dasharray="5,3"/>\n'
    svg += f'<text x="{ml+cw//2}" y="{height-5}" font-size="10" text-anchor="middle" fill="#64748b">Simulations</text>\n'
    svg += '</svg>\n'
    return svg


def _svg_bar_chart(data: list, width: int = 600, height: int = 400, title: str = "Chart",
                   x_label: str = "Value", max_value: float = None,
                   colors: list = None) -> str:
    if not data:
        return ""
    default_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
                       "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#06b6d4",
                       "#84cc16", "#d946ef"]
    colors = colors or default_colors
    mv = max_value or max(abs(v) for _, v in data) or 1
    ml, mr, mt, mb = 120, 30, 40, 50
    cw, ch = width - ml - mr, height - mt - mb
    n = min(len(data), 15)
    bh = min(22, max(12, (ch - 10) // n))
    sp = max(2, (ch - n * bh) // (n + 1))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="24" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>\n'
    for i in range(5):
        x = ml + cw * i // 4
        svg += f'<line x1="{x}" y1="{mt}" x2="{x}" y2="{mt+ch}" stroke="#f1f5f9" stroke-width="1"/>\n'
    for i, (name, val) in enumerate(data[:n]):
        y = mt + sp + i * (bh + sp)
        bw = max(2, int((abs(val) / mv) * cw))
        c = colors[i % len(colors)]
        svg += f'<rect x="{ml}" y="{y}" width="{bw}" height="{bh}" fill="{c}" rx="3"/>\n'
        dn = name[:20]
        svg += f'<text x="{ml-5}" y="{y+bh//2+4}" font-size="9" text-anchor="end" fill="#374151">{dn}</text>\n'
        vt = f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}"
        if bw > 50:
            svg += f'<text x="{ml+6}" y="{y+bh//2+4}" font-size="9" fill="white">{vt}</text>\n'
        else:
            svg += f'<text x="{ml+bw+6}" y="{y+bh//2+4}" font-size="9" fill="#374151">{vt}</text>\n'
    for i in range(5):
        val = mv * i / 4
        x = ml + cw * i // 4
        svg += f'<text x="{x}" y="{mt+ch+20}" font-size="9" text-anchor="middle" fill="#64748b">{val:.2f}</text>\n'
    svg += f'<text x="{ml+cw//2}" y="{height-5}" font-size="10" text-anchor="middle" fill="#64748b">{x_label}</text>\n'
    svg += '</svg>\n'
    return svg


def _svg_tornado(entries: list, base_value: float, width: int = 700, height: int = 400,
                 title: str = "Tornado Sensitivity") -> str:
    """Generate SVG tornado chart."""
    if not entries:
        return ""
    n = min(len(entries), 12)
    ml, mr, mt, mb = 140, 40, 45, 55
    cw, ch = width - ml - mr, height - mt - mb
    bh = min(24, max(14, (ch - 10) // n))
    sp = max(3, (ch - n * bh) // (n + 1))

    max_swing = max(max(abs(e.get("delta_low", 0)), abs(e.get("delta_high", 0))) for e in entries[:n])
    if max_swing < 1e-9:
        max_swing = 1.0
    center_x = ml + cw // 2

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="26" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>\n'

    # Center line (base value)
    svg += f'<line x1="{center_x}" y1="{mt}" x2="{center_x}" y2="{mt+ch}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,3"/>\n'

    for i, e in enumerate(entries[:n]):
        y = mt + sp + i * (bh + sp)
        dl = e.get("delta_low", 0)
        dh = e.get("delta_high", 0)

        # Low side (left of center = improvement = green)
        if abs(dl) > 1e-9:
            w_low = int((abs(dl) / max_swing) * (cw / 2))
            x_low = center_x - w_low
            svg += f'<rect x="{x_low}" y="{y}" width="{w_low}" height="{bh}" fill="#22c55e" rx="2" opacity="0.85"/>\n'

        # High side (right of center = degradation = red)
        if abs(dh) > 1e-9:
            w_high = int((abs(dh) / max_swing) * (cw / 2))
            svg += f'<rect x="{center_x}" y="{y}" width="{w_high}" height="{bh}" fill="#ef4444" rx="2" opacity="0.85"/>\n'

        # Label
        name = e.get("name", "?")[:22]
        svg += f'<text x="{ml-5}" y="{y+bh//2+4}" font-size="9" text-anchor="end" fill="#374151">{name}</text>\n'

        # Value labels
        swing = e.get("swing", 0)
        svg += f'<text x="{center_x + int((abs(dh)/max_swing)*(cw/2)) + 5}" y="{y+bh//2+4}" font-size="8" fill="#64748b">{swing:.1f}</text>\n'

    # X-axis labels
    for sign, label_text in [(-1, f"-{max_swing:.1f}"), (0, f"Base: {base_value:.1f}"), (1, f"+{max_swing:.1f}")]:
        x = center_x + sign * (cw // 2)
        svg += f'<text x="{x}" y="{mt+ch+20}" font-size="9" text-anchor="middle" fill="#64748b">{label_text}</text>\n'

    # Legend
    lx = width - mr - 180
    svg += f'<rect x="{lx}" y="{mt+2}" width="12" height="12" fill="#22c55e" rx="2" opacity="0.85"/>\n'
    svg += f'<text x="{lx+16}" y="{mt+12}" font-size="9" fill="#475569">Improvement (lower FIT)</text>\n'
    svg += f'<rect x="{lx}" y="{mt+18}" width="12" height="12" fill="#ef4444" rx="2" opacity="0.85"/>\n'
    svg += f'<text x="{lx+16}" y="{mt+28}" font-size="9" fill="#475569">Degradation (higher FIT)</text>\n'

    svg += f'<text x="{ml+cw//2}" y="{height-5}" font-size="10" text-anchor="middle" fill="#64748b">System Failure Rate Impact (FIT)</text>\n'
    svg += '</svg>\n'
    return svg


def _svg_growth_timeline(snapshots: list, target_fit: float = None,
                         width: int = 700, height: int = 350,
                         title: str = "Reliability Growth Timeline") -> str:
    """Generate SVG line chart showing system FIT across design revisions."""
    if not snapshots or len(snapshots) < 2:
        return ""
    fits = [s.get("system_fit", 0) for s in snapshots]
    labels = [s.get("version_label", f"v{i}") for i, s in enumerate(snapshots)]
    n = len(fits)
    v_min = min(fits)
    v_max = max(fits)
    if target_fit is not None:
        v_min = min(v_min, target_fit)
        v_max = max(v_max, target_fit)
    v_range = v_max - v_min
    if v_range < 1e-9:
        v_range = max(abs(v_max) * 0.1, 0.01)
    v_min -= v_range * 0.15
    v_max += v_range * 0.15
    v_range = v_max - v_min

    ml, mr, mt, mb = 70, 30, 40, 60
    cw, ch = width - ml - mr, height - mt - mb

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="26" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>\n'

    # Grid lines
    for i in range(5):
        y = mt + ch * i // 4
        val = v_max - (v_range * i / 4)
        svg += f'<line x1="{ml}" y1="{y}" x2="{ml+cw}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>\n'
        svg += f'<text x="{ml-8}" y="{y+4}" font-size="9" text-anchor="end" fill="#64748b">{val:.1f}</text>\n'

    # Target line
    if target_fit is not None:
        ty = mt + ch - ((target_fit - v_min) / v_range) * ch
        svg += f'<line x1="{ml}" y1="{ty:.1f}" x2="{ml+cw}" y2="{ty:.1f}" stroke="#22c55e" stroke-width="2" stroke-dasharray="6,4"/>\n'
        svg += f'<text x="{ml+cw+5}" y="{ty+4:.1f}" font-size="9" fill="#22c55e">Target</text>\n'

    # Data line
    path_d = ""
    points = []
    for i, fit in enumerate(fits):
        x = ml + (i / max(n - 1, 1)) * cw
        y = mt + ch - ((fit - v_min) / v_range) * ch
        points.append((x, y))
        if not path_d:
            path_d = f"M{x:.1f},{y:.1f}"
        else:
            path_d += f" L{x:.1f},{y:.1f}"
    svg += f'<path d="{path_d}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>\n'

    # Data points and labels
    for i, (x, y) in enumerate(points):
        color = "#22c55e" if (target_fit and fits[i] <= target_fit) else "#3b82f6"
        svg += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="white" stroke-width="2"/>\n'
        svg += f'<text x="{x:.1f}" y="{y-12:.1f}" font-size="9" text-anchor="middle" fill="#374151">{fits[i]:.1f}</text>\n'

    # X-axis version labels
    for i, label in enumerate(labels):
        x = ml + (i / max(n - 1, 1)) * cw
        svg += f'<text x="{x:.1f}" y="{mt+ch+20}" font-size="9" text-anchor="middle" fill="#64748b">{label[:12]}</text>\n'

    svg += f'<text x="{ml+cw//2}" y="{height-5}" font-size="10" text-anchor="middle" fill="#64748b">Design Revision</text>\n'

    # Legend
    lx = width - mr - 160
    ly = mt + 5
    svg += f'<line x1="{lx}" y1="{ly+6}" x2="{lx+20}" y2="{ly+6}" stroke="#3b82f6" stroke-width="2.5"/>\n'
    svg += f'<text x="{lx+25}" y="{ly+10}" font-size="10" fill="#475569">System FIT</text>\n'
    if target_fit is not None:
        svg += f'<line x1="{lx}" y1="{ly+22}" x2="{lx+20}" y2="{ly+22}" stroke="#22c55e" stroke-width="2" stroke-dasharray="6,4"/>\n'
        svg += f'<text x="{lx+25}" y="{ly+26}" font-size="10" fill="#475569">Target FIT</text>\n'

    svg += '</svg>\n'
    return svg


def _svg_budget_utilization(components: list, width: int = 700, height: int = 400,
                            title: str = "Budget Utilization by Component") -> str:
    """Generate SVG horizontal bar chart showing budget utilization per component."""
    if not components:
        return ""
    n = min(len(components), 20)
    ml, mr, mt, mb = 80, 50, 40, 50
    cw, ch = width - ml - mr, height - mt - mb
    bh = min(20, max(12, (ch - 10) // n))
    sp = max(2, (ch - n * bh) // (n + 1))

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="26" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>\n'

    # 100% reference line
    svg += f'<line x1="{ml+cw}" y1="{mt}" x2="{ml+cw}" y2="{mt+ch}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,3"/>\n'

    for i, comp in enumerate(components[:n]):
        y = mt + sp + i * (bh + sp)
        util = comp.get("utilization_pct", 0)
        ref = comp.get("ref", "?")[:10]
        passed = comp.get("passed", True)

        # Background bar (budget = 100%)
        svg += f'<rect x="{ml}" y="{y}" width="{cw}" height="{bh}" fill="#f1f5f9" rx="3"/>\n'

        # Utilization bar
        bar_w = min(cw, max(2, int((util / 100.0) * cw)))
        color = "#22c55e" if passed else "#ef4444"
        svg += f'<rect x="{ml}" y="{y}" width="{bar_w}" height="{bh}" fill="{color}" rx="3" opacity="0.8"/>\n'

        # Label
        svg += f'<text x="{ml-5}" y="{y+bh//2+4}" font-size="9" text-anchor="end" fill="#374151">{ref}</text>\n'

        # Value
        vt = f"{util:.0f}%"
        if bar_w > 40:
            svg += f'<text x="{ml+6}" y="{y+bh//2+4}" font-size="9" fill="white" font-weight="bold">{vt}</text>\n'
        else:
            svg += f'<text x="{ml+bar_w+5}" y="{y+bh//2+4}" font-size="9" fill="#374151">{vt}</text>\n'

    # X-axis
    for pct in [0, 25, 50, 75, 100]:
        x = ml + int(pct / 100.0 * cw)
        svg += f'<text x="{x}" y="{mt+ch+20}" font-size="9" text-anchor="middle" fill="#64748b">{pct}%</text>\n'
    svg += f'<text x="{ml+cw//2}" y="{height-5}" font-size="10" text-anchor="middle" fill="#64748b">Budget Utilization</text>\n'

    # Legend
    lx = width - mr - 180
    ly = mt + 5
    svg += f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="#22c55e" rx="2" opacity="0.8"/>\n'
    svg += f'<text x="{lx+16}" y="{ly+10}" font-size="9" fill="#475569">Within Budget</text>\n'
    svg += f'<rect x="{lx}" y="{ly+16}" width="12" height="12" fill="#ef4444" rx="2" opacity="0.8"/>\n'
    svg += f'<text x="{lx+16}" y="{ly+26}" font-size="9" fill="#475569">Over Budget</text>\n'

    svg += '</svg>\n'
    return svg


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generate professional reliability reports with embedded SVG charts."""

    def __init__(self, logo_path: Optional[str] = None, logo_mime: str = None):
        self.logo_path = logo_path
        self.logo_mime = logo_mime or "image/png"
        self.logo_base64 = self._encode_logo() if logo_path else None

    def _encode_logo(self) -> Optional[str]:
        try:
            p = Path(self.logo_path) if self.logo_path else None
            if p and p.exists():
                # Detect MIME from extension if not provided
                ext = p.suffix.lower()
                mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                            ".svg": "image/svg+xml", ".bmp": "image/bmp", ".gif": "image/gif"}
                self.logo_mime = mime_map.get(ext, "image/png")
                with open(p, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"[ReportGenerator] Logo encoding failed: {e}")
        return None

    def _css(self) -> str:
        return """
        :root { --primary:#2563eb; --success:#22c55e; --warning:#f59e0b;
                --danger:#ef4444; --bg:#f8fafc; --card:#fff; --text:#1e293b;
                --text2:#64748b; --border:#e2e8f0; }
        * { box-sizing:border-box; margin:0; padding:0; }
        body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
               background:var(--bg); color:var(--text); line-height:1.6; padding:32px; }
        .container { max-width:1200px; margin:0 auto; }
        .header { background:linear-gradient(135deg,var(--primary),#1e40af);
                  color:white; padding:32px 40px; border-radius:12px; margin-bottom:24px;
                  display:flex; align-items:center; justify-content:space-between; }
        .header h1 { font-size:2em; margin-bottom:6px; }
        .header .sub { opacity:0.9; font-size:0.95em; }
        .header-logo { max-height:70px; max-width:160px; margin-left:20px; object-fit:contain; }
        .card { background:var(--card); border-radius:10px; padding:24px;
                margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
        .card h2 { color:var(--primary); margin-bottom:14px; font-size:1.3em;
                   border-bottom:2px solid var(--border); padding-bottom:8px; }
        .card h3 { color:var(--text); margin:18px 0 10px; font-size:1.05em; }
        .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; }
        .metric { background:#f1f5f9; padding:18px; border-radius:8px; text-align:center; }
        .metric .v { font-size:1.8em; font-weight:bold; color:var(--primary); }
        .metric .l { color:var(--text2); font-size:0.85em; margin-top:2px; }
        .metric.ok .v { color:var(--success); }
        .metric.warn .v { color:var(--warning); }
        .metric.bad .v { color:var(--danger); }
        table { width:100%; border-collapse:collapse; margin:14px 0; font-size:0.9em; }
        th,td { padding:10px 12px; text-align:left; border-bottom:1px solid var(--border); }
        th { background:#f1f5f9; font-weight:600; color:var(--text2); position:sticky; top:0; }
        tr:hover { background:#f8fafc; }
        .mono { font-family:'SF Mono',Monaco,'Courier New',monospace; font-size:0.9em; }
        .badge { display:inline-block; padding:3px 10px; border-radius:16px; font-size:0.8em; font-weight:500; }
        .badge-ok { background:#dcfce7; color:#166534; }
        .badge-warn { background:#fef3c7; color:#92400e; }
        .badge-bad { background:#fecaca; color:#991b1b; }
        .badge-override { background:#dbeafe; color:#1e40af; }
        .chart-row { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0; }
        .chart-single { margin:16px 0; }
        .footer { text-align:center; color:var(--text2); padding:20px; font-size:0.82em; }
        @media print { body { padding:16px; } .card { break-inside:avoid; box-shadow:none; border:1px solid var(--border); } }
        @media (max-width:768px) { .chart-row { grid-template-columns:1fr; } }
        """

    def generate_html(self, data: ReportData) -> str:
        fit = data.system_lambda * 1e9
        mttf_years = data.system_mttf_hours / 8760 if data.system_mttf_hours < float('inf') else float('inf')
        sc = "ok" if data.system_reliability >= 0.99 else "warn" if data.system_reliability >= 0.95 else "bad"
        st = {"ok": "Excellent", "warn": "Acceptable", "bad": "Review Required"}[sc]

        html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Reliability Report - {data.project_name}</title>
<style>{self._css()}</style>
</head><body><div class="container">
<div class="header"><div>
<h1>Reliability Analysis Report</h1>
<div class="sub">Project: <strong>{data.project_name}</strong> | {datetime.fromisoformat(data.generated_at).strftime('%Y-%m-%d %H:%M')} | IEC TR 62380:2004</div>
</div>"""
        if self.logo_base64:
            html += f'<img src="data:{self.logo_mime};base64,{self.logo_base64}" alt="Logo" class="header-logo">'
        html += f"""</div>

<div class="card"><h2>System Summary</h2>
<div class="grid">
<div class="metric {sc}"><div class="v">{data.system_reliability:.6f}</div><div class="l">System Reliability</div></div>
<div class="metric"><div class="v">{fit:.1f}</div><div class="l">Failure Rate (FIT)</div></div>
<div class="metric"><div class="v">{mttf_years:.1f}</div><div class="l">MTTF (years)</div></div>
<div class="metric"><div class="v">{data.mission_years:.1f}</div><div class="l">Mission (years)</div></div>
</div>
<p style="margin-top:16px"><span class="badge badge-{sc}">{st}</span>
Mission: {data.n_cycles} cycles/yr, dT = {data.delta_t} degC</p></div>
"""

        if data.monte_carlo:
            html += self._mc_section(data.monte_carlo)
        if data.tornado:
            html += self._tornado_section(data.tornado)
        if data.design_margin:
            html += self._design_margin_section(data.design_margin)
        if data.sensitivity:
            html += self._sensitivity_section(data.sensitivity)
        if data.criticality:
            html += self._criticality_section(data.criticality)

        # v3.1.0 co-design sections
        if data.mission_profile:
            html += self._mission_profile_section(data.mission_profile)
        if data.budget:
            html += self._budget_section(data.budget)
        if data.derating:
            html += self._derating_section(data.derating)
        if data.swap_analysis:
            html += self._swap_section(data.swap_analysis)
        if data.growth_timeline:
            html += self._growth_section(data.growth_timeline)
        if data.correlated_mc:
            html += self._correlated_mc_section(data.correlated_mc)

        html += self._contributions_section(data.sheets)
        if data.sheet_mc:
            html += self._sheet_mc_section(data.sheet_mc)
        html += self._sheets_section(data.sheets)

        html += """
<div class="footer">
<p>Designed and developed by Eliot Abramo | KiCad Reliability Plugin v3.1.0 | IEC TR 62380:2004</p>
<p>This report is for engineering reference. Verify critical calculations independently.</p>
</div></div></body></html>"""
        return html

    def _mc_section(self, mc: Dict) -> str:
        samples = mc.get('samples', [])
        mean = mc.get('mean', 0)
        p5 = mc.get('percentile_5', 0)
        p50 = mc.get('percentile_50', 0)
        p95 = mc.get('percentile_95', 0)
        std = mc.get('std', 0)
        n = mc.get('n_simulations', 0)
        conv = mc.get('converged', False)
        cl = mc.get('confidence_level', 0.90)
        ci_lo = mc.get('ci_lower', p5)
        ci_hi = mc.get('ci_upper', p95)
        cl_pct = cl * 100

        html = f"""<div class="card"><h2>Monte Carlo Uncertainty Analysis</h2>
<div class="grid">
<div class="metric"><div class="v">{mean:.6f}</div><div class="l">Mean Reliability</div></div>
<div class="metric"><div class="v">{std:.6f}</div><div class="l">Standard Deviation</div></div>
<div class="metric"><div class="v">{ci_lo:.6f}</div><div class="l">{cl_pct:.0f}% CI Lower</div></div>
<div class="metric"><div class="v">{ci_hi:.6f}</div><div class="l">{cl_pct:.0f}% CI Upper</div></div>
</div>
<p style="margin-top:12px;color:var(--text2)">Based on {n:,} simulations {'(converged)' if conv else '(max iterations)'} | {cl_pct:.0f}% Confidence Interval: [{ci_lo:.6f}, {ci_hi:.6f}]</p>
"""
        if isinstance(samples, list) and len(samples) > 10:
            ci_label = f"{cl_pct:.0f}% CI"
            html += '<div class="chart-row">'
            html += _svg_histogram(samples, mean, p5, p95, title="Reliability Distribution",
                                   ci_lower=ci_lo, ci_upper=ci_hi, ci_label=ci_label)
            html += _svg_convergence(samples, title="Mean Convergence")
            html += '</div>'
        html += '</div>\n'
        return html

    def _tornado_section(self, tornado: Dict) -> str:
        entries = tornado.get("entries", [])
        if not entries:
            return ""
        base = tornado.get("base_lambda_fit", 0)
        pct = tornado.get("perturbation_pct", 20)

        html = f"""<div class="card"><h2>Sensitivity Analysis (Tornado)</h2>
<p>One-at-a-time perturbation of +/-{pct:.0f}% showing system failure rate impact.</p>
"""
        html += '<div class="chart-single">'
        html += _svg_tornado(entries, base, title=f"System FIT Sensitivity (+/-{pct:.0f}%)")
        html += '</div>'

        html += '<table><thead><tr><th>Parameter</th><th>Base (FIT)</th><th>Low (FIT)</th><th>High (FIT)</th><th>Swing (FIT)</th></tr></thead><tbody>\n'
        for e in entries[:15]:
            html += f'<tr><td>{e["name"]}</td><td class="mono">{e["base_value"]:.2f}</td>'
            html += f'<td class="mono">{e["low_value"]:.2f}</td><td class="mono">{e["high_value"]:.2f}</td>'
            html += f'<td class="mono">{e["swing"]:.2f}</td></tr>\n'
        html += '</tbody></table></div>\n'
        return html

    def _design_margin_section(self, dm: Dict) -> str:
        scenarios = dm.get("scenarios", [])
        if not scenarios:
            return ""
        bl = dm.get("baseline_lambda_fit", 0)
        br = dm.get("baseline_reliability", 1)

        html = """<div class="card"><h2>Design Margin Analysis (What-If Scenarios)</h2>
<p>Impact of environmental and design parameter changes on system reliability.</p>
<table><thead><tr><th>Scenario</th><th>Description</th><th>Lambda (FIT)</th><th>R(t)</th><th>Delta Lambda</th><th>Delta R</th></tr></thead><tbody>
"""
        for s in scenarios:
            dp = s.get("delta_lambda_pct", 0)
            dr = s.get("delta_reliability", 0)
            cls = "badge-bad" if dp > 5 else "badge-warn" if dp > 0 else "badge-ok"
            html += f'<tr><td><strong>{s["name"]}</strong></td><td>{s["description"]}</td>'
            html += f'<td class="mono">{s["lambda_fit"]:.2f}</td><td class="mono">{s["reliability"]:.6f}</td>'
            html += f'<td><span class="badge {cls}">{dp:+.1f}%</span></td>'
            html += f'<td class="mono">{dr:+.6f}</td></tr>\n'

        html += f'</tbody></table>\n'
        html += f'<p style="color:var(--text2);margin-top:8px">Baseline: {bl:.2f} FIT, R = {br:.6f}</p></div>\n'
        return html

    def _sensitivity_section(self, sens: Dict) -> str:
        params = sens.get("parameters", [])
        s_first = sens.get("S_first", [])
        s_total = sens.get("S_total", [])
        if not params:
            return ""
        ranked = sorted(zip(params, s_first, s_total), key=lambda x: -x[2])
        html = """<div class="card"><h2>Sobol Sensitivity Indices</h2>
<table><thead><tr><th>Parameter</th><th>S (First)</th><th>S (Total)</th><th>Interaction</th><th>Influence</th></tr></thead><tbody>
"""
        for name, sf, st_val in ranked[:15]:
            interact = st_val - sf
            inf = "High" if st_val > 0.3 else "Medium" if st_val > 0.1 else "Low"
            ic = "badge-bad" if st_val > 0.3 else "badge-warn" if st_val > 0.1 else "badge-ok"
            html += f'<tr><td>{name}</td><td class="mono">{sf:.4f}</td><td class="mono">{st_val:.4f}</td>'
            html += f'<td class="mono">{interact:.4f}</td><td><span class="badge {ic}">{inf}</span></td></tr>\n'
        html += '</tbody></table>\n'
        first_data = [(n, sf) for n, sf, _ in ranked[:12]]
        total_data = [(n, st_val) for n, _, st_val in ranked[:12]]
        html += '<div class="chart-row">'
        html += _svg_bar_chart(first_data, title="First-Order (S1)", x_label="S", max_value=1.0)
        html += _svg_bar_chart(total_data, title="Total-Order (ST)", x_label="S", max_value=1.0,
                               colors=["#10b981","#14b8a6","#06b6d4","#0ea5e9","#3b82f6","#6366f1","#8b5cf6","#a855f7","#d946ef","#ec4899"])
        html += '</div></div>\n'
        return html

    def _criticality_section(self, criticality: List) -> str:
        html = '<div class="card"><h2>Component Parameter Criticality</h2>\n'
        html += '<p>Parameter sensitivity showing which inputs most influence each component\'s failure rate.</p>\n'
        for entry in criticality:
            ref = entry.get("reference", "?")
            comp_type = entry.get("component_type", "Unknown")
            base_fit = entry.get("base_lambda_fit", 0)
            fields = entry.get("fields", [])
            if not fields:
                continue
            html += f'<h3>{ref} ({comp_type}) -- Base: {base_fit:.2f} FIT</h3>\n'
            html += '<table><thead><tr><th>Parameter</th><th>Value</th><th>Elasticity</th><th>Impact (%)</th><th>Direction</th></tr></thead><tbody>\n'
            for f in fields:
                elast = f.get("elasticity", 0)
                impact = f.get("impact_pct", 0)
                d = "increases lambda" if elast > 0 else "decreases lambda" if elast < 0 else "--"
                html += f'<tr><td>{f.get("name","")}</td><td>{f.get("value","")}</td>'
                html += f'<td>{elast:+.3f}</td><td>{impact:.1f}%</td><td>{d}</td></tr>\n'
            html += '</tbody></table>\n'
        html += '</div>\n'
        return html

    # === v3.1.0 Co-Design Report Sections ===

    def _mission_profile_section(self, mp: Dict) -> str:
        """Mission profile phasing section."""
        phases = mp.get("phases", [])
        if not phases:
            return ""
        profile_name = mp.get("name", "Custom")
        mission_years = mp.get("mission_years", 0)
        phased_lambda = mp.get("phased_lambda_fit", None)
        single_lambda = mp.get("single_phase_lambda_fit", None)
        delta_pct = mp.get("delta_percent", None)

        html = f"""<div class="card"><h2>Mission Profile Analysis</h2>
<p>Profile: <strong>{profile_name}</strong> | Mission Duration: {mission_years:.1f} years | {len(phases)} phases</p>
"""
        if phased_lambda is not None and single_lambda is not None and delta_pct is not None:
            cls = "badge-warn" if delta_pct > 2 else "badge-ok"
            html += f"""<div class="grid">
<div class="metric"><div class="v">{phased_lambda:.2f}</div><div class="l">Phased Model (FIT)</div></div>
<div class="metric"><div class="v">{single_lambda:.2f}</div><div class="l">Single-Phase (FIT)</div></div>
<div class="metric"><div class="v"><span class="badge {cls}">{delta_pct:+.1f}%</span></div><div class="l">Phasing Impact</div></div>
</div>
"""
        html += """<table><thead><tr><th>Phase</th><th>Duration (%)</th><th>T_ambient (degC)</th>
<th>T_junction (degC)</th><th>Cycles</th><th>Delta_T (degC)</th><th>tau_on</th></tr></thead><tbody>
"""
        for p in phases:
            html += f'<tr><td><strong>{p.get("name","?")}</strong></td>'
            html += f'<td class="mono">{p.get("duration_frac",0)*100:.1f}%</td>'
            html += f'<td class="mono">{p.get("t_ambient",25):.0f}</td>'
            html += f'<td class="mono">{p.get("t_junction") or 85:.0f}</td>'
            html += f'<td class="mono">{p.get("n_cycles",0)}</td>'
            html += f'<td class="mono">{p.get("delta_t",0):.1f}</td>'
            html += f'<td class="mono">{p.get("tau_on",1.0):.2f}</td></tr>\n'
        html += '</tbody></table></div>\n'
        return html

    def _budget_section(self, budget: Dict) -> str:
        """Reliability budget allocation section."""
        sheets = budget.get("sheet_budgets", [])
        strategy = budget.get("strategy", "proportional")
        target_fit = budget.get("target_fit", 0)
        actual_fit = budget.get("actual_fit", 0)
        margin_fit = budget.get("margin_fit", 0)
        target_r = budget.get("target_reliability", 0)
        design_margin = budget.get("design_margin_pct", 10)
        n_over = budget.get("components_over_budget", 0)
        n_total = budget.get("total_components", 0)

        status_cls = "ok" if n_over == 0 else "bad"
        status_txt = "All Within Budget" if n_over == 0 else f"{n_over} Over Budget"

        html = f"""<div class="card"><h2>Reliability Budget Allocation</h2>
<div class="grid">
<div class="metric"><div class="v">{target_r:.4f}</div><div class="l">Target R(t)</div></div>
<div class="metric"><div class="v">{target_fit:.1f}</div><div class="l">Max Allowable (FIT)</div></div>
<div class="metric"><div class="v">{actual_fit:.2f}</div><div class="l">Actual System (FIT)</div></div>
<div class="metric {status_cls}"><div class="v">{status_txt}</div><div class="l">Budget Status</div></div>
</div>
<p style="margin-top:12px;color:var(--text2)">Strategy: <strong>{strategy.title()}</strong> | Design Margin: {design_margin:.0f}% | System Margin: {margin_fit:.2f} FIT</p>
"""
        # Component budget table
        all_comps = []
        for sb in sheets:
            for cb in sb.get("component_budgets", []):
                all_comps.append(cb)
        if all_comps:
            # Sort by utilization descending
            all_comps.sort(key=lambda c: c.get("utilization_pct", 0), reverse=True)
            html += """<table><thead><tr><th>Reference</th><th>Type</th><th>Actual (FIT)</th>
<th>Budget (FIT)</th><th>Margin (FIT)</th><th>Utilization</th><th>Status</th></tr></thead><tbody>
"""
            for cb in all_comps[:30]:
                util = cb.get("utilization_pct", 0)
                passed = cb.get("passed", True)
                badge = "badge-ok" if passed else "badge-bad"
                status = "PASS" if passed else "FAIL"
                html += f'<tr><td><strong>{cb.get("ref","?")}</strong></td>'
                html += f'<td>{cb.get("component_type","")}</td>'
                html += f'<td class="mono">{cb.get("actual_fit",0):.3f}</td>'
                html += f'<td class="mono">{cb.get("budget_fit",0):.3f}</td>'
                html += f'<td class="mono">{cb.get("margin_fit",0):+.3f}</td>'
                html += f'<td class="mono">{util:.1f}%</td>'
                html += f'<td><span class="badge {badge}">{status}</span></td></tr>\n'
            html += '</tbody></table>\n'

            # Budget utilization chart
            html += '<div class="chart-single">'
            html += _svg_budget_utilization(all_comps[:15])
            html += '</div>'

        # Recommendations
        recs = budget.get("recommendations", [])
        if recs:
            html += '<h3>Recommendations</h3><ul style="margin:8px 0 0 16px;color:var(--text2)">\n'
            for rec in recs:
                html += f'<li>{rec}</li>\n'
            html += '</ul>\n'

        html += '</div>\n'
        return html

    def _derating_section(self, derating: Dict) -> str:
        """Derating guidance section."""
        recs = derating.get("recommendations", [])
        system_actual = derating.get("system_actual_fit", 0)
        system_target = derating.get("system_target_fit", 0)
        system_gap = derating.get("system_gap_fit", 0)
        n_feasible = derating.get("n_feasible", 0)

        gap_cls = "ok" if system_gap <= 0 else "bad"
        gap_txt = "Within Target" if system_gap <= 0 else "Above Target"

        html = f"""<div class="card"><h2>Derating Guidance</h2>
<div class="grid">
<div class="metric"><div class="v">{system_actual:.2f}</div><div class="l">Actual System (FIT)</div></div>
<div class="metric"><div class="v">{system_target:.1f}</div><div class="l">Target (FIT)</div></div>
<div class="metric {gap_cls}"><div class="v">{system_gap:+.2f}</div><div class="l">Gap (FIT)</div></div>
<div class="metric"><div class="v">{n_feasible}</div><div class="l">Feasible Actions</div></div>
</div>
<p style="margin-top:12px"><span class="badge badge-{gap_cls}">{gap_txt}</span>
{'Recommendations below show optimization opportunities.' if system_gap <= 0 else 'Derating actions required to meet target.'}</p>
"""
        if recs:
            html += """<table><thead><tr><th>#</th><th>Reference</th><th>Parameter</th>
<th>Current</th><th>Required</th><th>Change</th><th>FIT Saved</th>
<th>Feasibility</th><th>Actions</th></tr></thead><tbody>
"""
            for i, r in enumerate(recs[:20], 1):
                feas = r.get("feasibility", "unknown")
                feas_cls = {"easy": "badge-ok", "moderate": "badge-warn",
                            "difficult": "badge-bad", "infeasible": "badge-bad"}.get(feas, "")
                actions = "; ".join(r.get("actions", [])[:2])
                html += f'<tr><td>{i}</td><td><strong>{r.get("reference","?")}</strong></td>'
                html += f'<td>{r.get("parameter","")}</td>'
                html += f'<td class="mono">{r.get("current_value","")}</td>'
                html += f'<td class="mono">{r.get("required_value","")}</td>'
                html += f'<td class="mono">{r.get("change_pct",0):+.1f}%</td>'
                html += f'<td class="mono">{r.get("fit_saved",0):.3f}</td>'
                html += f'<td><span class="badge {feas_cls}">{feas.title()}</span></td>'
                html += f'<td style="font-size:0.85em">{actions}</td></tr>\n'
            html += '</tbody></table>\n'
        html += '</div>\n'
        return html

    def _swap_section(self, swap: Dict) -> str:
        """Component swap analysis section."""
        improvements = swap.get("improvements", [])
        total_analyzed = swap.get("total_analyzed", 0)
        best_single = swap.get("best_single_improvement", None)

        html = f"""<div class="card"><h2>Component Swap Analysis</h2>
<p>Analyzed {total_analyzed} possible swaps across all components.</p>
"""
        if best_single:
            html += f"""<div class="grid">
<div class="metric ok"><div class="v">{best_single.get("delta_system_fit",0):+.2f}</div><div class="l">Best Single Improvement (FIT)</div></div>
<div class="metric"><div class="v">{best_single.get("ref","?")}</div><div class="l">Best Component</div></div>
<div class="metric"><div class="v">{best_single.get("swap_desc","")}</div><div class="l">Best Swap</div></div>
</div>
"""
        if improvements:
            html += """<table><thead><tr><th>#</th><th>Reference</th><th>Type</th>
<th>Parameter</th><th>Swap</th><th>Before (FIT)</th><th>After (FIT)</th>
<th>Delta</th><th>System Impact</th></tr></thead><tbody>
"""
            for i, imp in enumerate(improvements[:20], 1):
                delta_pct = imp.get("delta_percent", 0)
                cls = "badge-ok" if delta_pct < 0 else "badge-bad"
                html += f'<tr><td>{i}</td><td><strong>{imp.get("ref","?")}</strong></td>'
                html += f'<td>{imp.get("component_type","")}</td>'
                html += f'<td>{imp.get("param_name","")}</td>'
                html += f'<td>{imp.get("new_value","")}</td>'
                html += f'<td class="mono">{imp.get("fit_before",0):.2f}</td>'
                html += f'<td class="mono">{imp.get("fit_after",0):.2f}</td>'
                html += f'<td><span class="badge {cls}">{delta_pct:+.1f}%</span></td>'
                html += f'<td class="mono">{imp.get("delta_system_fit",0):+.2f} FIT</td></tr>\n'
            html += '</tbody></table>\n'
        html += '</div>\n'
        return html

    def _growth_section(self, growth: Dict) -> str:
        """Reliability growth tracking section."""
        snapshots = growth.get("snapshots", [])
        target_fit = growth.get("target_fit", None)
        comparisons = growth.get("comparisons", [])

        html = '<div class="card"><h2>Reliability Growth Tracking</h2>\n'

        if len(snapshots) >= 2:
            html += '<div class="chart-single">'
            html += _svg_growth_timeline(snapshots, target_fit=target_fit)
            html += '</div>'

        if snapshots:
            html += """<table><thead><tr><th>Version</th><th>Date</th><th>Components</th>
<th>System FIT</th><th>R(t)</th><th>Notes</th></tr></thead><tbody>
"""
            for s in snapshots:
                html += f'<tr><td><strong>{s.get("version_label","?")}</strong></td>'
                html += f'<td>{s.get("timestamp","")[:10]}</td>'
                html += f'<td>{s.get("n_components",0)}</td>'
                html += f'<td class="mono">{s.get("system_fit",0):.2f}</td>'
                html += f'<td class="mono">{s.get("system_reliability",0):.6f}</td>'
                html += f'<td>{s.get("notes","")}</td></tr>\n'
            html += '</tbody></table>\n'

        # Latest comparison
        if comparisons:
            latest = comparisons[-1]
            html += f"""<h3>Latest Revision: {latest.get("from_version","")} to {latest.get("to_version","")}</h3>
<div class="grid">
<div class="metric"><div class="v">{latest.get("system_delta_fit",0):+.2f}</div><div class="l">FIT Change</div></div>
<div class="metric"><div class="v">{latest.get("components_added",0)}</div><div class="l">Added</div></div>
<div class="metric"><div class="v">{latest.get("components_removed",0)}</div><div class="l">Removed</div></div>
<div class="metric"><div class="v">{latest.get("components_improved",0)}</div><div class="l">Improved</div></div>
</div>
"""
            changes = latest.get("top_changes", [])
            if changes:
                html += '<table><thead><tr><th>Component</th><th>Change</th><th>FIT Delta</th></tr></thead><tbody>\n'
                for c in changes[:10]:
                    html += f'<tr><td>{c.get("ref","?")}</td><td>{c.get("change_type","")}</td>'
                    html += f'<td class="mono">{c.get("delta_fit",0):+.3f}</td></tr>\n'
                html += '</tbody></table>\n'

        html += '</div>\n'
        return html

    def _correlated_mc_section(self, cmc: Dict) -> str:
        """Correlated Monte Carlo section."""
        n_groups = cmc.get("n_groups", 0)
        n_sims = cmc.get("n_simulations", 0)
        indep_mean = cmc.get("independent_mean", 0)
        indep_std = cmc.get("independent_std", 0)
        corr_mean = cmc.get("correlated_mean", 0)
        corr_std = cmc.get("correlated_std", 0)
        std_ratio = cmc.get("std_ratio", 1.0)
        variance_impact = cmc.get("variance_impact", "comparable")
        groups = cmc.get("groups", [])

        impact_cls = {"narrower": "badge-ok", "comparable": "badge-ok",
                      "wider": "badge-warn", "much wider": "badge-bad"}.get(variance_impact, "")

        html = f"""<div class="card"><h2>Correlated Monte Carlo Analysis</h2>
<p>Correlation-aware uncertainty propagation with {n_groups} group(s), {n_sims:,} simulations.</p>
<div class="grid">
<div class="metric"><div class="v">{indep_mean:.6f}</div><div class="l">Independent Mean R</div></div>
<div class="metric"><div class="v">{corr_mean:.6f}</div><div class="l">Correlated Mean R</div></div>
<div class="metric"><div class="v">{indep_std:.6f}</div><div class="l">Independent Std</div></div>
<div class="metric"><div class="v">{corr_std:.6f}</div><div class="l">Correlated Std</div></div>
</div>
<div class="grid" style="margin-top:12px">
<div class="metric"><div class="v">{std_ratio:.3f}x</div><div class="l">Std Ratio (Corr/Indep)</div></div>
<div class="metric"><div class="v"><span class="badge {impact_cls}">{variance_impact.title()}</span></div><div class="l">Variance Impact</div></div>
</div>
"""
        if std_ratio > 1.05:
            html += '<p style="margin-top:12px;color:var(--text2)">Correlation increases uncertainty bounds. '
            html += 'Independent Monte Carlo underestimates tail risk for this design.</p>\n'
        elif std_ratio < 0.95:
            html += '<p style="margin-top:12px;color:var(--text2)">Correlation reduces uncertainty. '
            html += 'Independent Monte Carlo overestimates variability for this design.</p>\n'
        else:
            html += '<p style="margin-top:12px;color:var(--text2)">Correlation has minimal impact on uncertainty bounds.</p>\n'

        if groups:
            html += '<h3>Correlation Groups</h3>\n'
            html += '<table><thead><tr><th>Group</th><th>Components</th><th>rho</th><th>Description</th></tr></thead><tbody>\n'
            for g in groups:
                members = ", ".join(g.get("component_refs", [])[:8])
                if len(g.get("component_refs", [])) > 8:
                    members += f" (+{len(g['component_refs'])-8} more)"
                html += f'<tr><td><strong>{g.get("name","?")}</strong></td>'
                html += f'<td>{members}</td>'
                html += f'<td class="mono">{g.get("rho",0):.2f}</td>'
                html += f'<td>{g.get("description","")}</td></tr>\n'
            html += '</tbody></table>\n'

        # Confidence intervals comparison
        ci_indep_lo = cmc.get("independent_ci_lower", 0)
        ci_indep_hi = cmc.get("independent_ci_upper", 0)
        ci_corr_lo = cmc.get("correlated_ci_lower", 0)
        ci_corr_hi = cmc.get("correlated_ci_upper", 0)
        if ci_indep_lo and ci_corr_lo:
            html += '<h3>Confidence Interval Comparison</h3>\n'
            html += '<table><thead><tr><th>Model</th><th>5th %ile</th><th>Mean</th><th>95th %ile</th><th>Width</th></tr></thead><tbody>\n'
            html += f'<tr><td>Independent</td><td class="mono">{ci_indep_lo:.6f}</td>'
            html += f'<td class="mono">{indep_mean:.6f}</td><td class="mono">{ci_indep_hi:.6f}</td>'
            html += f'<td class="mono">{ci_indep_hi - ci_indep_lo:.6f}</td></tr>\n'
            html += f'<tr><td>Correlated</td><td class="mono">{ci_corr_lo:.6f}</td>'
            html += f'<td class="mono">{corr_mean:.6f}</td><td class="mono">{ci_corr_hi:.6f}</td>'
            html += f'<td class="mono">{ci_corr_hi - ci_corr_lo:.6f}</td></tr>\n'
            html += '</tbody></table>\n'

        html += '</div>\n'
        return html

    def _contributions_section(self, sheets: Dict) -> str:
        contribs = []
        total_lam = 0
        for path, data in sheets.items():
            lam = data.get('lambda', 0)
            if lam > 0:
                contribs.append((path.rstrip('/').split('/')[-1] or 'Root', lam, path))
                total_lam += lam
        if total_lam == 0:
            return ""
        contribs.sort(key=lambda x: -x[1])
        html = """<div class="card"><h2>Failure Rate Contributions</h2>
<table><thead><tr><th>Sheet</th><th>Lambda (FIT)</th><th>Contribution</th><th>Cumulative</th><th></th></tr></thead><tbody>
"""
        cum = 0
        for name, lam, _ in contribs[:20]:
            pct = lam / total_lam * 100
            cum += pct
            html += f'<tr><td>{name}</td><td class="mono">{lam*1e9:.2f}</td><td>{pct:.1f}%</td><td>{cum:.1f}%</td>'
            html += f'<td style="width:120px"><div style="height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden">'
            html += f'<div style="height:100%;width:{min(100,pct)}%;background:linear-gradient(90deg,#3b82f6,#60a5fa);border-radius:4px"></div></div></td></tr>\n'
        html += '</tbody></table>\n'
        chart_data = [(n, l/total_lam) for n, l, _ in contribs[:12]]
        html += '<div class="chart-single">'
        html += _svg_bar_chart(chart_data, title="Relative Contributions", x_label="Fraction", max_value=1.0)
        html += '</div></div>\n'
        return html

    def _sheet_mc_section(self, sheet_mc: Dict) -> str:
        if not sheet_mc:
            return ""
        html = """<div class="card"><h2>Per-Sheet Monte Carlo Analysis</h2>
<table><thead><tr><th>Sheet</th><th>Mean R</th><th>Std</th><th>CI Lower</th><th>CI Upper</th><th>Sims</th></tr></thead><tbody>
"""
        for path, result in sorted(sheet_mc.items()):
            mc = result if isinstance(result, dict) else {}
            name = path.rstrip('/').split('/')[-1] or 'Root'
            cl = mc.get("confidence_level", 0.90)
            html += f'<tr><td>{name}</td><td class="mono">{mc.get("mean",0):.6f}</td>'
            html += f'<td class="mono">{mc.get("std",0):.6f}</td>'
            html += f'<td class="mono">{mc.get("ci_lower", mc.get("percentile_5",0)):.6f}</td>'
            html += f'<td class="mono">{mc.get("ci_upper", mc.get("percentile_95",0)):.6f}</td>'
            html += f'<td>{mc.get("n_simulations",0):,}</td></tr>\n'
        html += '</tbody></table></div>\n'
        return html

    def _sheets_section(self, sheets: Dict) -> str:
        html = '<div class="card"><h2>Component Details by Sheet</h2>\n'
        for path, sheet_data in sorted(sheets.items()):
            sf = sheet_data.get("lambda", 0) * 1e9
            sr = sheet_data.get("r", 1.0)
            components = sheet_data.get("components", [])
            sn = path.rstrip("/").split("/")[-1] or "Root"
            html += f'<h3>{sn}</h3>\n'
            html += f'<p style="color:var(--text2);margin-bottom:10px">Path: <code>{path}</code> | lambda = {sf:.2f} FIT | R = {sr:.6f} | {len(components)} components</p>\n'
            html += '<table><thead><tr><th>Ref</th><th>Value</th><th>Type</th><th>Lambda (FIT)</th><th>R</th><th>Source</th></tr></thead><tbody>\n'
            for comp in components:
                cf = comp.get("lambda", 0) * 1e9
                cr = comp.get("r", 1.0)
                override = comp.get("override_lambda")
                src = '<span class="badge badge-override">Override</span>' if override is not None else "Calculated"
                html += f'<tr><td><strong>{comp.get("ref","?")}</strong></td>'
                html += f'<td>{comp.get("value","")}</td><td>{comp.get("class","Unknown")}</td>'
                html += f'<td class="mono">{cf:.2f}</td><td class="mono">{cr:.6f}</td><td>{src}</td></tr>\n'
            html += '</tbody></table>\n'
        html += '</div>\n'
        return html

    # Other formats

    def generate_markdown(self, data: ReportData) -> str:
        fit = data.system_lambda * 1e9
        mttf_years = data.system_mttf_hours / 8760 if data.system_mttf_hours < float('inf') else float('inf')
        md = f"""# Reliability Analysis Report

**Project:** {data.project_name}
**Generated:** {datetime.fromisoformat(data.generated_at).strftime('%Y-%m-%d %H:%M')}
**Standard:** IEC TR 62380:2004

## System Summary

| Metric | Value |
|--------|-------|
| System Reliability | {data.system_reliability:.6f} |
| Failure Rate | {fit:.2f} FIT |
| MTTF | {mttf_years:.1f} years |
| Mission | {data.mission_years:.1f} years |

"""
        if data.monte_carlo:
            mc = data.monte_carlo
            cl = mc.get('confidence_level', 0.90) * 100
            md += f"""## Monte Carlo Analysis

| Metric | Value |
|--------|-------|
| Mean | {mc.get('mean',0):.6f} |
| Std Dev | {mc.get('std',0):.6f} |
| {cl:.0f}% CI Lower | {mc.get('ci_lower',0):.6f} |
| {cl:.0f}% CI Upper | {mc.get('ci_upper',0):.6f} |
| Simulations | {mc.get('n_simulations',0):,} |

"""
        if data.tornado:
            md += "## Tornado Sensitivity\n\n| Parameter | Swing (FIT) |\n|---|---|\n"
            for e in data.tornado.get("entries", [])[:10]:
                md += f"| {e['name']} | {e['swing']:.2f} |\n"
            md += "\n"

        if data.design_margin:
            md += "## Design Margin Scenarios\n\n| Scenario | Lambda (FIT) | Delta |\n|---|---|---|\n"
            for s in data.design_margin.get("scenarios", []):
                md += f"| {s['name']} | {s['lambda_fit']:.2f} | {s['delta_lambda_pct']:+.1f}% |\n"
            md += "\n"

        # v3.1.0 co-design sections
        if data.mission_profile:
            mp = data.mission_profile
            md += f"## Mission Profile: {mp.get('name','Custom')}\n\n"
            md += "| Phase | Duration (%) | T_amb (degC) | T_junc (degC) | Cycles | dT (degC) | tau_on |\n|---|---|---|---|---|---|---|\n"
            for p in mp.get("phases", []):
                md += f"| {p.get('name','')} | {p.get('duration_frac',0)*100:.1f}% | {p.get('t_ambient',25):.0f} | {p.get('t_junction') or 85:.0f} | {p.get('n_cycles',0)} | {p.get('delta_t',0):.1f} | {p.get('tau_on',1):.2f} |\n"
            md += "\n"

        if data.budget:
            b = data.budget
            md += f"## Reliability Budget Allocation\n\n"
            md += f"Strategy: **{b.get('strategy','').title()}** | Target R: {b.get('target_reliability',0):.4f} | Max FIT: {b.get('target_fit',0):.1f} | Actual: {b.get('actual_fit',0):.2f} FIT\n\n"
            md += "| Ref | Type | Actual (FIT) | Budget (FIT) | Utilization | Status |\n|---|---|---|---|---|---|\n"
            for sb in b.get("sheet_budgets", []):
                for cb in sb.get("component_budgets", []):
                    st = "PASS" if cb.get("passed", True) else "FAIL"
                    md += f"| {cb.get('ref','')} | {cb.get('component_type','')} | {cb.get('actual_fit',0):.3f} | {cb.get('budget_fit',0):.3f} | {cb.get('utilization_pct',0):.1f}% | {st} |\n"
            md += "\n"

        if data.derating:
            d = data.derating
            md += f"## Derating Guidance\n\nSystem: {d.get('system_actual_fit',0):.2f} FIT | Target: {d.get('system_target_fit',0):.1f} FIT | Gap: {d.get('system_gap_fit',0):+.2f} FIT\n\n"
            md += "| # | Ref | Parameter | Current | Required | Change | FIT Saved | Feasibility |\n|---|---|---|---|---|---|---|---|\n"
            for i, r in enumerate(d.get("recommendations", [])[:15], 1):
                md += f"| {i} | {r.get('reference','')} | {r.get('parameter','')} | {r.get('current_value','')} | {r.get('required_value','')} | {r.get('change_pct',0):+.1f}% | {r.get('fit_saved',0):.3f} | {r.get('feasibility','')} |\n"
            md += "\n"

        if data.swap_analysis:
            sw = data.swap_analysis
            md += "## Component Swap Analysis\n\n"
            md += "| # | Ref | Parameter | Swap | Before (FIT) | After (FIT) | Delta | System Impact |\n|---|---|---|---|---|---|---|---|\n"
            for i, imp in enumerate(sw.get("improvements", [])[:15], 1):
                md += f"| {i} | {imp.get('ref','')} | {imp.get('param_name','')} | {imp.get('new_value','')} | {imp.get('fit_before',0):.2f} | {imp.get('fit_after',0):.2f} | {imp.get('delta_percent',0):+.1f}% | {imp.get('delta_system_fit',0):+.2f} FIT |\n"
            md += "\n"

        if data.growth_timeline:
            gt = data.growth_timeline
            md += "## Reliability Growth Timeline\n\n"
            md += "| Version | Date | Components | System FIT | R(t) | Notes |\n|---|---|---|---|---|---|\n"
            for s in gt.get("snapshots", []):
                md += f"| {s.get('version_label','')} | {s.get('timestamp','')[:10]} | {s.get('n_components',0)} | {s.get('system_fit',0):.2f} | {s.get('system_reliability',0):.6f} | {s.get('notes','')} |\n"
            md += "\n"

        if data.correlated_mc:
            cm = data.correlated_mc
            md += f"## Correlated Monte Carlo\n\n"
            md += f"| Model | Mean R | Std | 5th %ile | 95th %ile |\n|---|---|---|---|---|\n"
            md += f"| Independent | {cm.get('independent_mean',0):.6f} | {cm.get('independent_std',0):.6f} | {cm.get('independent_ci_lower',0):.6f} | {cm.get('independent_ci_upper',0):.6f} |\n"
            md += f"| Correlated | {cm.get('correlated_mean',0):.6f} | {cm.get('correlated_std',0):.6f} | {cm.get('correlated_ci_lower',0):.6f} | {cm.get('correlated_ci_upper',0):.6f} |\n"
            md += f"\nStd Ratio: {cm.get('std_ratio',1):.3f}x | Variance Impact: {cm.get('variance_impact','comparable')}\n\n"

        for path, sd in sorted(data.sheets.items()):
            name = path.rstrip('/').split('/')[-1] or 'Root'
            md += f"### {name}\n\n| Ref | Value | Type | Lambda (FIT) | R |\n|---|---|---|---|---|\n"
            for c in sd.get('components',[])[:20]:
                md += f"| {c.get('ref','?')} | {c.get('value','')} | {c.get('class','')} | {c.get('lambda',0)*1e9:.2f} | {c.get('r',1):.6f} |\n"
            md += "\n"
        md += "\n---\n*Designed and developed by Eliot Abramo | KiCad Reliability Plugin v3.1.0*\n"
        return md

    def generate_csv(self, data: ReportData) -> str:
        lines = ["Sheet,Reference,Value,Type,Lambda_FIT,Reliability,Source"]
        for path, sd in sorted(data.sheets.items()):
            for c in sd.get("components", []):
                src = "override" if c.get("override_lambda") is not None else "calculated"
                lines.append(f'"{path}","{c.get("ref","")}","{c.get("value","")}","{c.get("class","")}",{c.get("lambda",0)*1e9:.4f},{c.get("r",1):.8f},{src}')
        return "\n".join(lines)

    def generate_json(self, data: ReportData) -> str:
        output = {
            "meta": {"project": data.project_name, "generated": data.generated_at, "standard": "IEC TR 62380:2004"},
            "mission": {"hours": data.mission_hours, "years": data.mission_years, "thermal_cycles": data.n_cycles, "delta_t": data.delta_t},
            "system": {"reliability": data.system_reliability, "lambda": data.system_lambda, "lambda_fit": data.system_lambda*1e9, "mttf_hours": data.system_mttf_hours},
            "sheets": {p: {k: v for k, v in sd.items() if k != 'components'} for p, sd in data.sheets.items()},
        }
        if data.monte_carlo:
            mc = dict(data.monte_carlo); mc.pop('samples', None)
            output["monte_carlo"] = mc
        if data.sensitivity:
            output["sensitivity"] = data.sensitivity
        if data.tornado:
            output["tornado"] = data.tornado
        if data.design_margin:
            output["design_margin"] = data.design_margin
        if data.criticality:
            output["criticality"] = data.criticality
        # v3.1.0 co-design data
        if data.mission_profile:
            output["mission_profile"] = data.mission_profile
        if data.budget:
            output["budget"] = data.budget
        if data.derating:
            output["derating"] = data.derating
        if data.swap_analysis:
            output["swap_analysis"] = data.swap_analysis
        if data.growth_timeline:
            output["growth_timeline"] = data.growth_timeline
        if data.correlated_mc:
            cm = dict(data.correlated_mc)
            cm.pop("independent_samples", None)
            cm.pop("correlated_samples", None)
            output["correlated_mc"] = cm
        return json.dumps(output, indent=2, default=str)

    def generate(self, data: ReportData, format: str = "html") -> str:
        return {"html": self.generate_html, "markdown": self.generate_markdown,
                "md": self.generate_markdown, "csv": self.generate_csv,
                "json": self.generate_json}.get(format.lower(), self.generate_html)(data)

    @staticmethod
    def html_to_pdf(html_content: str, output_path: str):
        """Convert HTML report to PDF using reportlab.

        Falls back through multiple strategies:
        1. weasyprint (best quality, may not be installed)
        2. reportlab with manual layout (always available)
        """
        # Strategy 1: Try weasyprint for high-fidelity HTML->PDF
        try:
            from weasyprint import HTML as WeasyprintHTML
            WeasyprintHTML(string=html_content).write_pdf(output_path)
            return
        except ImportError:
            pass

        # Strategy 2: reportlab PDF generation from structured data
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=20*mm, bottomMargin=20*mm
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            'ReportTitle', parent=styles['Title'],
            fontSize=18, spaceAfter=6, textColor=HexColor('#1e40af')
        ))
        styles.add(ParagraphStyle(
            'SectionHead', parent=styles['Heading2'],
            fontSize=13, spaceBefore=14, spaceAfter=6,
            textColor=HexColor('#2563eb'),
            borderWidth=0, borderPadding=0,
        ))
        styles.add(ParagraphStyle(
            'MetricLabel', parent=styles['Normal'],
            fontSize=8, textColor=HexColor('#64748b'), alignment=TA_CENTER
        ))
        styles.add(ParagraphStyle(
            'MetricValue', parent=styles['Normal'],
            fontSize=14, textColor=HexColor('#2563eb'),
            alignment=TA_CENTER, fontName='Helvetica-Bold'
        ))
        styles.add(ParagraphStyle(
            'TableCell', parent=styles['Normal'], fontSize=8, leading=10
        ))
        styles.add(ParagraphStyle(
            'FooterStyle', parent=styles['Normal'],
            fontSize=7, textColor=HexColor('#64748b'), alignment=TA_CENTER
        ))

        story = []

        # --- Extract data from HTML using simple regex ---
        import re

        def _extract_text(pattern, text, default=""):
            m = re.search(pattern, text, re.DOTALL)
            return m.group(1).strip() if m else default

        project = _extract_text(r'Project:\s*<strong>(.*?)</strong>', html_content, "Reliability Report")
        date_str = _extract_text(r'</strong>\s*\|\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})', html_content, "")

        # Title
        story.append(Paragraph("Reliability Analysis Report", styles['ReportTitle']))
        story.append(Paragraph(
            f"Project: <b>{project}</b> | {date_str} | IEC TR 62380:2004",
            styles['Normal']
        ))
        story.append(Spacer(1, 8*mm))

        # Extract metric values from HTML
        metric_pattern = r'<div class="v">(.*?)</div>\s*<div class="l">(.*?)</div>'
        metrics = re.findall(metric_pattern, html_content)

        if metrics:
            story.append(Paragraph("System Summary", styles['SectionHead']))
            metric_data = []
            metric_labels = []
            for val, label in metrics[:4]:
                metric_data.append(Paragraph(f'<b>{val}</b>', styles['MetricValue']))
                metric_labels.append(Paragraph(label, styles['MetricLabel']))
            if metric_data:
                t = Table([metric_data, metric_labels], colWidths=[doc.width/len(metric_data)]*len(metric_data))
                t.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BACKGROUND', (0,0), (-1,0), HexColor('#f1f5f9')),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                    ('BOX', (0,0), (-1,-1), 0.5, HexColor('#e2e8f0')),
                ]))
                story.append(t)
                story.append(Spacer(1, 4*mm))

        # Extract tables from HTML
        table_pattern = r'<h2>(.*?)</h2>.*?<table>(.*?)</table>'
        tables = re.findall(table_pattern, html_content, re.DOTALL)

        for title, table_html in tables:
            title_clean = re.sub(r'<[^>]+>', '', title)
            story.append(Paragraph(title_clean, styles['SectionHead']))

            # Parse rows
            header_match = re.findall(r'<th[^>]*>(.*?)</th>', table_html, re.DOTALL)
            headers = [re.sub(r'<[^>]+>', '', h).strip() for h in header_match]

            row_pattern = r'<tr>(.*?)</tr>'
            rows_html = re.findall(row_pattern, table_html, re.DOTALL)

            all_rows = []
            if headers:
                all_rows.append([Paragraph(f'<b>{h}</b>', styles['TableCell']) for h in headers])

            for row_html in rows_html:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
                if cells:
                    clean = [Paragraph(re.sub(r'<[^>]+>', '', c).strip(), styles['TableCell']) for c in cells]
                    if len(clean) == len(headers) or not headers:
                        all_rows.append(clean)

            if len(all_rows) > 1:
                n_cols = len(all_rows[0])
                col_w = doc.width / n_cols if n_cols > 0 else doc.width
                try:
                    t = Table(all_rows, colWidths=[col_w]*n_cols, repeatRows=1)
                    style_cmds = [
                        ('FONTSIZE', (0,0), (-1,-1), 7),
                        ('LEADING', (0,0), (-1,-1), 9),
                        ('TOPPADDING', (0,0), (-1,-1), 3),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                        ('GRID', (0,0), (-1,-1), 0.4, HexColor('#e2e8f0')),
                        ('BACKGROUND', (0,0), (-1,0), HexColor('#f1f5f9')),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor('#ffffff'), HexColor('#f8fafc')]),
                    ]
                    t.setStyle(TableStyle(style_cmds))
                    story.append(t)
                except Exception:
                    pass  # Skip malformed tables
            story.append(Spacer(1, 3*mm))

        # Footer
        story.append(Spacer(1, 10*mm))
        story.append(HRFlowable(width="100%", color=HexColor('#e2e8f0')))
        story.append(Paragraph(
            "Designed and developed by Eliot Abramo | KiCad Reliability Plugin v3.1.0 | IEC TR 62380:2004",
            styles['FooterStyle']
        ))
        story.append(Paragraph(
            "This report is for engineering reference. Verify critical calculations independently.",
            styles['FooterStyle']
        ))

        doc.build(story)

    def generate_pdf(self, data: ReportData, output_path: str):
        """Generate PDF report."""
        html = self.generate_html(data)
        self.html_to_pdf(html, output_path)
