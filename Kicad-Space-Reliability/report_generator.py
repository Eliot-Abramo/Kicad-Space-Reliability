"""
Enhanced Report Generator with Inline SVG Charts
=================================================
Professional reliability reports with embedded interactive visualizations.
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
    sheet_mc: Optional[Dict] = None  # per-sheet MC results

    generated_at: str = None

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


# === SVG Chart Generators ===

def _svg_histogram(samples: list, mean: float, p5: float, p95: float,
                   width: int = 600, height: int = 300, title: str = "Distribution") -> str:
    """Generate an SVG histogram with mean and percentile lines."""
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

    # Grid lines
    for i in range(5):
        y = mt + ch * i // 4
        svg += f'<line x1="{ml}" y1="{y}" x2="{ml+cw}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>\n'

    # Bars
    bw = max(1, cw // n_bins - 1)
    for i, count in enumerate(bins):
        if count > 0:
            x = ml + i * cw // n_bins
            bh = int((count / max_count) * ch)
            y = mt + ch - bh
            svg += f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="#3b82f6" opacity="0.85" rx="1"/>\n'

    # Reference lines
    def val_to_x(v):
        return ml + (v - s_min) / s_range * cw

    # Mean
    mx = val_to_x(mean)
    svg += f'<line x1="{mx:.1f}" y1="{mt}" x2="{mx:.1f}" y2="{mt+ch}" stroke="#ef4444" stroke-width="2"/>\n'
    svg += f'<text x="{mx+4:.1f}" y="{mt+14}" font-size="9" fill="#ef4444">Mean={mean:.5f}</text>\n'

    # Percentiles
    if p5 is not None:
        px = val_to_x(p5)
        svg += f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt+ch}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5,3"/>\n'
        svg += f'<text x="{px+3:.1f}" y="{mt+ch-4}" font-size="8" fill="#f59e0b">5%={p5:.5f}</text>\n'
    if p95 is not None:
        px = val_to_x(p95)
        svg += f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt+ch}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="5,3"/>\n'
        svg += f'<text x="{px-80:.1f}" y="{mt+ch-4}" font-size="8" fill="#f59e0b">95%={p95:.5f}</text>\n'

    # Axes
    svg += f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#64748b" stroke-width="1"/>\n'
    for i in range(5):
        v = s_min + s_range * i / 4
        x = ml + cw * i // 4
        svg += f'<text x="{x}" y="{mt+ch+18}" font-size="9" fill="#64748b" text-anchor="middle">{v:.4f}</text>\n'
    svg += f'<text x="{ml+cw//2}" y="{height-6}" font-size="10" fill="#64748b" text-anchor="middle">Reliability R(t)</text>\n'

    svg += '</svg>\n'
    return svg


def _svg_bar_chart(data: List[Tuple[str, float]], width: int = 600, height: int = 350,
                   title: str = "Chart", x_label: str = "Value", max_value: float = None,
                   colors: List[str] = None) -> str:
    """Generate an SVG horizontal bar chart."""
    if not data:
        return ""

    default_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
                      "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16"]
    colors = colors or default_colors
    max_val = max_value or max(d[1] for d in data) or 1.0
    max_val = max(max_val, 0.001)

    ml, mr, mt, mb = 140, 40, 40, 40
    cw, ch = width - ml - mr, height - mt - mb
    n = min(len(data), 15)
    bar_h = min(22, max(12, ch // n - 4))
    spacing = max(3, (ch - n * bar_h) // (n + 1))

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="24" font-size="13" font-weight="bold" fill="#1e293b">{title}</text>\n'

    for i in range(5):
        x = ml + cw * i // 4
        svg += f'<line x1="{x}" y1="{mt}" x2="{x}" y2="{mt+ch}" stroke="#f1f5f9" stroke-width="1"/>\n'

    for i, (name, val) in enumerate(data[:n]):
        y = mt + spacing + i * (bar_h + spacing)
        bw = max(2, int((val / max_val) * cw))
        color = colors[i % len(colors)]

        svg += f'<rect x="{ml}" y="{y}" width="{bw}" height="{bar_h}" fill="{color}" rx="3"/>\n'

        display = name[:20] + "..." if len(name) > 20 else name
        svg += f'<text x="{ml-6}" y="{y+bar_h//2+4}" font-size="9" fill="#1e293b" text-anchor="end">{display}</text>\n'

        vt = f"{val:.4f}" if val < 1 else f"{val:.1f}"
        tx = ml + bw + 6 if bw < cw * 0.7 else ml + 6
        fc = "#1e293b" if bw < cw * 0.7 else "#ffffff"
        svg += f'<text x="{tx}" y="{y+bar_h//2+4}" font-size="9" fill="{fc}">{vt}</text>\n'

    svg += f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#64748b" stroke-width="1"/>\n'
    for i in range(5):
        v = max_val * i / 4
        x = ml + cw * i // 4
        svg += f'<text x="{x}" y="{mt+ch+16}" font-size="8" fill="#64748b" text-anchor="middle">{v:.2f}</text>\n'
    svg += f'<text x="{ml+cw//2}" y="{height-6}" font-size="10" fill="#64748b" text-anchor="middle">{x_label}</text>\n'
    svg += '</svg>\n'
    return svg


def _svg_convergence(samples: list, width: int = 600, height: int = 200, title: str = "Convergence") -> str:
    """Generate SVG running mean convergence plot."""
    if not samples or len(samples) < 10:
        return ""

    import numpy as np
    arr = np.array(samples)
    cumsum = np.cumsum(arr)
    running_mean = cumsum / np.arange(1, len(arr) + 1)

    step = max(1, len(running_mean) // 100)
    points = [(i, float(running_mean[i])) for i in range(0, len(running_mean), step)]
    if len(running_mean) - 1 not in [p[0] for p in points]:
        points.append((len(running_mean) - 1, float(running_mean[-1])))

    vals = [p[1] for p in points]
    v_min, v_max = min(vals), max(vals)
    v_range = v_max - v_min
    if v_range < 1e-12:
        v_range = abs(v_max) * 0.01 if v_max != 0 else 0.01
    v_min -= v_range * 0.1
    v_max += v_range * 0.1
    v_range = v_max - v_min
    n_max = len(arr)

    ml, mr, mt, mb = 60, 20, 35, 35
    cw, ch = width - ml - mr, height - mt - mb

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto;font-family:sans-serif;">\n'
    svg += f'<rect width="{width}" height="{height}" fill="white" rx="8"/>\n'
    svg += f'<text x="{ml}" y="22" font-size="12" font-weight="bold" fill="#1e293b">{title}</text>\n'

    for i in range(5):
        y = mt + ch * i // 4
        svg += f'<line x1="{ml}" y1="{y}" x2="{ml+cw}" y2="{y}" stroke="#f1f5f9" stroke-width="1"/>\n'

    # Path
    path_d = ""
    for idx, (n, v) in enumerate(points):
        x = ml + (n / n_max) * cw
        y = mt + ch - ((v - v_min) / v_range) * ch
        path_d += f"{'M' if idx == 0 else 'L'}{x:.1f},{y:.1f} "

    svg += f'<path d="{path_d}" fill="none" stroke="#3b82f6" stroke-width="2"/>\n'

    # Final value dashed line
    final_y = mt + ch - ((float(running_mean[-1]) - v_min) / v_range) * ch
    svg += f'<line x1="{ml}" y1="{final_y:.1f}" x2="{ml+cw}" y2="{final_y:.1f}" stroke="#22c55e" stroke-width="1" stroke-dasharray="4,3"/>\n'
    svg += f'<text x="{ml+cw+4}" y="{final_y+4:.1f}" font-size="8" fill="#22c55e">{float(running_mean[-1]):.5f}</text>\n'

    svg += f'<text x="{ml+cw//2}" y="{height-6}" font-size="9" fill="#64748b" text-anchor="middle">Simulations</text>\n'
    svg += '</svg>\n'
    return svg


class ReportGenerator:
    """Generate professional reliability reports with embedded SVG charts."""

    def __init__(self, logo_path: Optional[str] = None):
        self.logo_path = logo_path
        self.logo_base64 = self._encode_logo() if logo_path else None

    def _encode_logo(self) -> Optional[str]:
        try:
            if self.logo_path and Path(self.logo_path).exists():
                with open(self.logo_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            pass
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
        .header-logo { max-height:70px; max-width:140px; margin-left:20px; }
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
        .chart-row { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:16px 0; }
        .chart-single { margin:16px 0; }
        .footer { text-align:center; color:var(--text2); padding:20px; font-size:0.82em; }
        .tab-container { margin:16px 0; }
        @media print { body { padding:16px; } .card { break-inside:avoid; box-shadow:none; border:1px solid var(--border); } }
        @media (max-width:768px) { .chart-row { grid-template-columns:1fr; } }
        """

    def generate_html(self, data: ReportData) -> str:
        fit = data.system_lambda * 1e9
        mttf_years = data.system_mttf_hours / 8760 if data.system_mttf_hours < float('inf') else float('inf')

        if data.system_reliability >= 0.99:
            sc, st = "ok", "Excellent"
        elif data.system_reliability >= 0.95:
            sc, st = "warn", "Acceptable"
        else:
            sc, st = "bad", "Review Required"

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
            html += f'<img src="data:image/png;base64,{self.logo_base64}" alt="Logo" class="header-logo">'
        html += f"""</div>

<div class="card"><h2>System Summary</h2>
<div class="grid">
<div class="metric {sc}"><div class="v">{data.system_reliability:.6f}</div><div class="l">System Reliability</div></div>
<div class="metric"><div class="v">{fit:.1f}</div><div class="l">Failure Rate (FIT)</div></div>
<div class="metric"><div class="v">{mttf_years:.1f}</div><div class="l">MTTF (years)</div></div>
<div class="metric"><div class="v">{data.mission_years:.1f}</div><div class="l">Mission (years)</div></div>
</div>
<p style="margin-top:16px"><span class="badge badge-{sc}">{st}</span>
Mission: {data.n_cycles} cycles/yr, ΔT = {data.delta_t}°C</p></div>
"""

        # Monte Carlo section with SVG charts
        if data.monte_carlo:
            mc = data.monte_carlo
            html += self._mc_section(mc)

        # Sensitivity section with SVG charts
        if data.sensitivity:
            html += self._sensitivity_section(data.sensitivity)

        # Failure rate contributions
        html += self._contributions_section(data.sheets)

        # Per-sheet MC results
        if data.sheet_mc:
            html += self._sheet_mc_section(data.sheet_mc)

        # Sheet-by-sheet details
        html += self._sheets_section(data.sheets)

        html += """
<div class="footer">
<p>Generated by KiCad Reliability Calculator v2.0 | IEC TR 62380:2004</p>
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

        html = f"""<div class="card"><h2>Monte Carlo Uncertainty Analysis</h2>
<div class="grid">
<div class="metric"><div class="v">{mean:.6f}</div><div class="l">Mean Reliability</div></div>
<div class="metric"><div class="v">{std:.6f}</div><div class="l">Standard Deviation</div></div>
<div class="metric"><div class="v">{p5:.6f}</div><div class="l">5th Percentile</div></div>
<div class="metric"><div class="v">{p95:.6f}</div><div class="l">95th Percentile</div></div>
</div>
<p style="margin-top:12px;color:var(--text2)">Based on {n:,} simulations {'(converged)' if conv else '(max iterations)'}</p>
"""
        if isinstance(samples, list) and len(samples) > 10:
            html += '<div class="chart-row">'
            html += _svg_histogram(samples, mean, p5, p95, title="Reliability Distribution")
            html += _svg_convergence(samples, title="Mean Convergence")
            html += '</div>'
        html += '</div>\n'
        return html

    def _sensitivity_section(self, sens: Dict) -> str:
        params = sens.get("parameters", [])
        s_first = sens.get("S_first", [])
        s_total = sens.get("S_total", [])
        if not params:
            return ""

        ranked = sorted(zip(params, s_first, s_total), key=lambda x: -x[2])

        html = """<div class="card"><h2>Sensitivity Analysis (Sobol Indices)</h2>
<table><thead><tr><th>Parameter</th><th>S₁ (First)</th><th>Sₜ (Total)</th><th>Interaction</th><th>Influence</th></tr></thead><tbody>
"""
        for name, sf, st_val in ranked[:15]:
            interact = st_val - sf
            influence = "High" if st_val > 0.3 else "Medium" if st_val > 0.1 else "Low"
            inf_cls = "badge-bad" if st_val > 0.3 else "badge-warn" if st_val > 0.1 else "badge-ok"
            html += f'<tr><td>{name}</td><td class="mono">{sf:.4f}</td><td class="mono">{st_val:.4f}</td>'
            html += f'<td class="mono">{interact:.4f}</td><td><span class="badge {inf_cls}">{influence}</span></td></tr>\n'

        html += '</tbody></table>\n'

        # SVG charts
        first_data = [(n, sf) for n, sf, _ in ranked[:12]]
        total_data = [(n, st_val) for n, _, st_val in ranked[:12]]
        html += '<div class="chart-row">'
        html += _svg_bar_chart(first_data, title="First-Order Indices (S₁)", x_label="S₁", max_value=1.0)
        html += _svg_bar_chart(total_data, title="Total-Order Indices (Sₜ)", x_label="Sₜ", max_value=1.0,
                               colors=["#10b981", "#14b8a6", "#06b6d4", "#0ea5e9", "#3b82f6",
                                       "#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899"])
        html += '</div></div>\n'
        return html

    def _contributions_section(self, sheets: Dict) -> str:
        contribs = []
        total_lam = 0
        for path, data in sheets.items():
            lam = data.get('lambda', 0)
            if lam > 0:
                name = path.rstrip('/').split('/')[-1] or 'Root'
                contribs.append((name, lam, path))
                total_lam += lam
        if total_lam == 0:
            return ""

        contribs.sort(key=lambda x: -x[1])

        html = """<div class="card"><h2>Failure Rate Contributions</h2>
<table><thead><tr><th>Sheet</th><th>λ (FIT)</th><th>Contribution</th><th>Cumulative</th><th></th></tr></thead><tbody>
"""
        cum = 0
        for name, lam, _ in contribs[:20]:
            pct = lam / total_lam * 100
            cum += pct
            bar_w = min(100, pct)
            html += f'<tr><td>{name}</td><td class="mono">{lam*1e9:.2f}</td><td>{pct:.1f}%</td><td>{cum:.1f}%</td>'
            html += f'<td style="width:120px"><div style="height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden">'
            html += f'<div style="height:100%;width:{bar_w}%;background:linear-gradient(90deg,#3b82f6,#60a5fa);border-radius:4px"></div></div></td></tr>\n'
        html += '</tbody></table>\n'

        chart_data = [(n, l / total_lam) for n, l, _ in contribs[:12]]
        html += '<div class="chart-single">'
        html += _svg_bar_chart(chart_data, title="Relative Failure Rate Contributions", x_label="Fraction of Total λ", max_value=1.0)
        html += '</div></div>\n'
        return html

    def _sheet_mc_section(self, sheet_mc: Dict) -> str:
        if not sheet_mc:
            return ""

        html = """<div class="card"><h2>Per-Sheet Monte Carlo Analysis</h2>
<table><thead><tr><th>Sheet</th><th>Mean R</th><th>Std</th><th>5th %ile</th><th>95th %ile</th><th>Sims</th></tr></thead><tbody>
"""
        for path, result in sorted(sheet_mc.items()):
            mc = result if isinstance(result, dict) else result.get('mc_result', result) if isinstance(result, dict) else {}
            if hasattr(result, 'mc_result'):
                mc = result.mc_result
                mc = {'mean': mc.mean, 'std': mc.std, 'percentile_5': mc.percentile_5,
                      'percentile_95': mc.percentile_95, 'n_simulations': mc.n_simulations}
            name = path.rstrip('/').split('/')[-1] or 'Root'
            html += f'<tr><td>{name}</td><td class="mono">{mc.get("mean",0):.6f}</td>'
            html += f'<td class="mono">{mc.get("std",0):.6f}</td>'
            html += f'<td class="mono">{mc.get("percentile_5",0):.6f}</td>'
            html += f'<td class="mono">{mc.get("percentile_95",0):.6f}</td>'
            html += f'<td>{mc.get("n_simulations",0):,}</td></tr>\n'
        html += '</tbody></table></div>\n'
        return html

    def _sheets_section(self, sheets: Dict) -> str:
        html = '<div class="card"><h2>Component Details by Sheet</h2>\n'

        for path, sheet_data in sorted(sheets.items()):
            sheet_fit = sheet_data.get("lambda", 0) * 1e9
            sheet_r = sheet_data.get("r", 1.0)
            components = sheet_data.get("components", [])
            sheet_name = path.rstrip("/").split("/")[-1] or "Root"

            html += f'<h3>{sheet_name}</h3>\n'
            html += f'<p style="color:var(--text2);margin-bottom:10px">Path: <code>{path}</code> | λ = {sheet_fit:.2f} FIT | R = {sheet_r:.6f} | {len(components)} components</p>\n'
            html += '<table><thead><tr><th>Ref</th><th>Value</th><th>Type</th><th>λ (FIT)</th><th>R</th></tr></thead><tbody>\n'

            for comp in components:
                c_fit = comp.get("lambda", 0) * 1e9
                c_r = comp.get("r", 1.0)
                html += f'<tr><td><strong>{comp.get("ref", "?")}</strong></td>'
                html += f'<td>{comp.get("value", "")}</td>'
                html += f'<td>{comp.get("class", "Unknown")}</td>'
                html += f'<td class="mono">{c_fit:.2f}</td>'
                html += f'<td class="mono">{c_r:.6f}</td></tr>\n'

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
            md += f"""## Monte Carlo Analysis

| Metric | Value |
|--------|-------|
| Mean | {mc.get('mean',0):.6f} |
| Std Dev | {mc.get('std',0):.6f} |
| 5th %ile | {mc.get('percentile_5',0):.6f} |
| 95th %ile | {mc.get('percentile_95',0):.6f} |
| Simulations | {mc.get('n_simulations',0):,} |

"""
        if data.sensitivity:
            sens = data.sensitivity
            md += "## Sensitivity Analysis\n\n| Parameter | S1 | ST |\n|---|---|---|\n"
            ranked = sorted(zip(sens.get("parameters",[]), sens.get("S_first",[]), sens.get("S_total",[])), key=lambda x: -x[2])
            for name, sf, st_val in ranked[:10]:
                md += f"| {name} | {sf:.4f} | {st_val:.4f} |\n"
            md += "\n"

        for path, sd in sorted(data.sheets.items()):
            name = path.rstrip('/').split('/')[-1] or 'Root'
            md += f"### {name}\n\nλ = {sd.get('lambda',0)*1e9:.2f} FIT | R = {sd.get('r',1):.6f}\n\n"
            md += "| Ref | Value | Type | λ (FIT) | R |\n|---|---|---|---|---|\n"
            for c in sd.get('components',[])[:20]:
                md += f"| {c.get('ref','?')} | {c.get('value','')} | {c.get('class','')} | {c.get('lambda',0)*1e9:.2f} | {c.get('r',1):.6f} |\n"
            md += "\n"
        md += "\n---\n*Generated by KiCad Reliability Calculator v2.0*\n"
        return md

    def generate_csv(self, data: ReportData) -> str:
        lines = ["Sheet,Reference,Value,Type,Lambda_FIT,Reliability"]
        for path, sd in sorted(data.sheets.items()):
            for c in sd.get("components", []):
                lines.append(f'"{path}","{c.get("ref","")}","{c.get("value","")}","{c.get("class","")}",{c.get("lambda",0)*1e9:.4f},{c.get("r",1):.8f}')
        return "\n".join(lines)

    def generate_json(self, data: ReportData) -> str:
        output = {
            "meta": {"project": data.project_name, "generated": data.generated_at, "standard": "IEC TR 62380:2004"},
            "mission": {"hours": data.mission_hours, "years": data.mission_years,
                        "thermal_cycles": data.n_cycles, "delta_t": data.delta_t},
            "system": {"reliability": data.system_reliability, "lambda": data.system_lambda,
                       "lambda_fit": data.system_lambda * 1e9, "mttf_hours": data.system_mttf_hours},
            "sheets": {p: {k: v for k, v in sd.items() if k != 'components'} for p, sd in data.sheets.items()},
        }
        if data.monte_carlo:
            mc = dict(data.monte_carlo)
            mc.pop('samples', None)
            output["monte_carlo"] = mc
        if data.sensitivity:
            output["sensitivity"] = data.sensitivity
        return json.dumps(output, indent=2, default=str)

    def generate(self, data: ReportData, format: str = "html") -> str:
        return {"html": self.generate_html, "markdown": self.generate_markdown,
                "md": self.generate_markdown, "csv": self.generate_csv,
                "json": self.generate_json}.get(format.lower(), self.generate_html)(data)
