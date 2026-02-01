"""
Enhanced Report Generator - Industrial Grade
=============================================
Professional reliability reports with interactive charts, block-level tabs,
comprehensive data display, and print-ready formatting.

Features:
- Tabbed navigation per block/sheet
- SVG charts (histograms, Pareto, pie charts, bar charts)
- Complete component data with all IEC TR 62380 parameters
- Monte Carlo distribution visualizations
- Sensitivity tornado diagrams
- Contribution waterfall charts
- Collapsible sections
- Print-optimized CSS
"""

import json
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ComponentReportData:
    """Detailed component data for reporting."""
    ref: str
    value: str
    component_type: str
    lambda_fit: float
    reliability: float
    contribution_pct: float
    
    # IEC TR 62380 parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Breakdown
    lambda_die: float = 0.0
    lambda_package: float = 0.0
    lambda_eos: float = 0.0
    
    # Monte Carlo results (if available)
    mc_mean: Optional[float] = None
    mc_std: Optional[float] = None
    mc_p5: Optional[float] = None
    mc_p95: Optional[float] = None


@dataclass
class BlockReportData:
    """Block/sheet data for reporting."""
    id: str
    name: str
    path: str
    connection_type: str  # series, parallel, k_of_n
    k_value: int = 2
    
    lambda_total: float = 0.0
    reliability: float = 1.0
    contribution_pct: float = 0.0
    
    components: List[ComponentReportData] = field(default_factory=list)
    child_blocks: List[str] = field(default_factory=list)
    
    # Per-block Monte Carlo
    mc_result: Optional[Dict] = None
    
    # Per-block sensitivity
    sensitivity_result: Optional[Dict] = None


@dataclass 
class ReportData:
    """Complete report data container."""
    project_name: str
    mission_hours: float
    mission_years: float
    n_cycles: int
    delta_t: float
    
    system_reliability: float
    system_lambda: float
    system_mttf_hours: float
    
    blocks: List[BlockReportData] = field(default_factory=list)
    
    # System-level Monte Carlo
    monte_carlo: Optional[Dict] = None
    mc_samples: Optional[List[float]] = None
    
    # System-level sensitivity
    sensitivity: Optional[Dict] = None
    
    # Legacy compatibility
    sheets: Dict[str, Dict] = field(default_factory=dict)
    
    generated_at: str = None
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


class SVGChartGenerator:
    """Generate SVG charts for reports."""
    
    @staticmethod
    def histogram(samples: List[float], title: str = "Distribution", 
                  width: int = 500, height: int = 250, n_bins: int = 30,
                  highlight_percentiles: bool = True) -> str:
        """Generate SVG histogram from samples."""
        if not samples or len(samples) < 2:
            return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#999">No data</text></svg>'
        
        samples = [s for s in samples if math.isfinite(s)]
        if len(samples) < 2:
            return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#999">No valid data</text></svg>'
        
        min_val, max_val = min(samples), max(samples)
        if max_val - min_val < 1e-10:
            max_val = min_val + 1e-6
        
        bin_width = (max_val - min_val) / n_bins
        bins = [0] * n_bins
        
        for s in samples:
            idx = min(int((s - min_val) / bin_width), n_bins - 1)
            bins[idx] += 1
        
        max_count = max(bins) if bins else 1
        
        # Margins
        ml, mr, mt, mb = 50, 20, 40, 50
        cw, ch = width - ml - mr, height - mt - mb
        bar_w = cw / n_bins
        
        # Calculate percentiles
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        p5 = sorted_samples[int(n * 0.05)]
        p50 = sorted_samples[int(n * 0.50)]
        p95 = sorted_samples[int(n * 0.95)]
        mean = sum(samples) / n
        std = (sum((s-mean)**2 for s in samples)/n)**0.5
        
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="barGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#4FC3F7"/>
      <stop offset="100%" style="stop-color:#0288D1"/>
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>
  <text x="{width//2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{title}</text>
'''
        
        # Grid lines
        for i in range(5):
            y = mt + ch * i / 4
            svg += f'  <line x1="{ml}" y1="{y}" x2="{ml+cw}" y2="{y}" stroke="#E0E0E0" stroke-dasharray="3,3"/>\n'
        
        # Bars
        for i, count in enumerate(bins):
            bar_h = (count / max_count) * ch if max_count > 0 else 0
            x = ml + i * bar_w
            y = mt + ch - bar_h
            svg += f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-1:.1f}" height="{bar_h:.1f}" fill="url(#barGrad)" rx="2"/>\n'
        
        # Percentile markers
        if highlight_percentiles:
            for val, label, color in [(p5, "5%", "#FF5722"), (p50, "50%", "#4CAF50"), (p95, "95%", "#FF5722"), (mean, "μ", "#2196F3")]:
                x = ml + ((val - min_val) / (max_val - min_val)) * cw
                svg += f'  <line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ch}" stroke="{color}" stroke-width="2" stroke-dasharray="5,3"/>\n'
                svg += f'  <text x="{x:.1f}" y="{mt-5}" text-anchor="middle" font-size="10" fill="{color}">{label}</text>\n'
        
        # X-axis labels
        for i in range(5):
            val = min_val + (max_val - min_val) * i / 4
            x = ml + cw * i / 4
            svg += f'  <text x="{x:.1f}" y="{mt+ch+20}" text-anchor="middle" font-size="10" fill="#666">{val:.4f}</text>\n'
        
        # Stats box
        svg += f'''  <rect x="{width-150}" y="{mt}" width="140" height="80" fill="white" stroke="#DDD" rx="4"/>
  <text x="{width-145}" y="{mt+18}" font-size="10" fill="#333">Mean: {mean:.6f}</text>
  <text x="{width-145}" y="{mt+33}" font-size="10" fill="#333">Std: {std:.6f}</text>
  <text x="{width-145}" y="{mt+48}" font-size="10" fill="#FF5722">P5: {p5:.6f}</text>
  <text x="{width-145}" y="{mt+63}" font-size="10" fill="#FF5722">P95: {p95:.6f}</text>
'''
        
        svg += '</svg>'
        return svg
    
    @staticmethod
    def pareto_chart(data: List[Tuple[str, float]], title: str = "Pareto Chart",
                     width: int = 600, height: int = 300) -> str:
        """Generate Pareto chart (bars + cumulative line)."""
        if not data:
            return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#999">No data</text></svg>'
        
        # Sort descending and take top 15
        data = sorted(data, key=lambda x: -x[1])[:15]
        total = sum(d[1] for d in data)
        if total <= 0:
            total = 1
        
        ml, mr, mt, mb = 120, 60, 40, 80
        cw, ch = width - ml - mr, height - mt - mb
        bar_w = cw / len(data) * 0.7
        gap = cw / len(data) * 0.3
        
        max_val = data[0][1] if data else 1
        
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>
  <text x="{width//2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{title}</text>
'''
        
        # Bars and cumulative line
        cumulative = 0
        line_points = []
        
        colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#C62828', '#00838F', '#5D4037', '#455A64']
        
        for i, (name, val) in enumerate(data):
            x = ml + i * (bar_w + gap)
            bar_h = (val / max_val) * ch
            y = mt + ch - bar_h
            
            color = colors[i % len(colors)]
            svg += f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>\n'
            
            # Label (rotated)
            label = name[:12] + "…" if len(name) > 12 else name
            svg += f'  <text x="{x + bar_w/2:.1f}" y="{mt+ch+15}" text-anchor="end" font-size="9" fill="#333" transform="rotate(-45 {x + bar_w/2:.1f} {mt+ch+15})">{label}</text>\n'
            
            # Cumulative
            cumulative += val / total
            cx = x + bar_w / 2
            cy = mt + ch * (1 - cumulative)
            line_points.append(f"{cx:.1f},{cy:.1f}")
            
            # Percentage on bar
            pct = (val / total) * 100
            svg += f'  <text x="{x + bar_w/2:.1f}" y="{y-5}" text-anchor="middle" font-size="9" fill="#333">{pct:.1f}%</text>\n'
        
        # Cumulative line
        if line_points:
            svg += f'  <polyline points="{" ".join(line_points)}" fill="none" stroke="#E53935" stroke-width="2.5"/>\n'
            for pt in line_points:
                svg += f'  <circle cx="{pt.split(",")[0]}" cy="{pt.split(",")[1]}" r="4" fill="#E53935"/>\n'
        
        # Right Y-axis (cumulative %)
        for i in range(5):
            y = mt + ch * (1 - i/4)
            svg += f'  <text x="{ml+cw+10}" y="{y+4}" font-size="9" fill="#E53935">{i*25}%</text>\n'
        
        svg += f'  <text x="{ml+cw+40}" y="{height//2}" font-size="10" fill="#E53935" transform="rotate(90 {ml+cw+40} {height//2})">Cumulative</text>\n'
        
        svg += '</svg>'
        return svg
    
    @staticmethod
    def tornado_diagram(data: List[Tuple[str, float, float, float]], title: str = "Sensitivity Tornado",
                        width: int = 600, height: int = 350) -> str:
        """Generate tornado diagram for sensitivity.
        data: List of (param_name, low_impact, nominal, high_impact)
        """
        if not data:
            return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#999">No data</text></svg>'
        
        # Sort by total range
        data = sorted(data, key=lambda x: -(abs(x[3] - x[1])))[:12]
        
        ml, mr, mt, mb = 140, 20, 40, 40
        cw, ch = width - ml - mr, height - mt - mb
        bar_h = min(22, (ch - 20) / len(data))
        gap = 4
        
        # Find min/max for scaling
        all_vals = [d[1] for d in data] + [d[3] for d in data]
        nominal = data[0][2] if data else 0
        
        # Center at nominal
        max_range = max(abs(max(all_vals) - nominal), abs(min(all_vals) - nominal))
        if max_range < 1e-10:
            max_range = 1
        
        center_x = ml + cw / 2
        
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>
  <text x="{width//2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{title}</text>
  <line x1="{center_x}" y1="{mt}" x2="{center_x}" y2="{mt+ch}" stroke="#333" stroke-width="2"/>
'''
        
        for i, (name, low, nom, high) in enumerate(data):
            y = mt + 10 + i * (bar_h + gap)
            
            # Calculate bar positions
            x_low = center_x + ((low - nominal) / max_range) * (cw / 2)
            x_high = center_x + ((high - nominal) / max_range) * (cw / 2)
            
            # Draw bar (low side in red, high side in blue)
            if x_low < center_x:
                svg += f'  <rect x="{x_low:.1f}" y="{y}" width="{center_x - x_low:.1f}" height="{bar_h}" fill="#EF5350" rx="3"/>\n'
            if x_high > center_x:
                svg += f'  <rect x="{center_x}" y="{y}" width="{x_high - center_x:.1f}" height="{bar_h}" fill="#42A5F5" rx="3"/>\n'
            
            # Label
            label = name[:18] + "…" if len(name) > 18 else name
            svg += f'  <text x="{ml-5}" y="{y + bar_h/2 + 4}" text-anchor="end" font-size="10" fill="#333">{label}</text>\n'
            
            # Values at ends
            svg += f'  <text x="{x_low - 5:.1f}" y="{y + bar_h/2 + 3}" text-anchor="end" font-size="8" fill="#C62828">{low:.4f}</text>\n'
            svg += f'  <text x="{x_high + 5:.1f}" y="{y + bar_h/2 + 3}" text-anchor="start" font-size="8" fill="#1565C0">{high:.4f}</text>\n'
        
        # Legend
        svg += f'''  <rect x="{ml}" y="{height-30}" width="15" height="12" fill="#EF5350" rx="2"/>
  <text x="{ml+20}" y="{height-21}" font-size="10" fill="#333">Low (-20%)</text>
  <rect x="{ml+100}" y="{height-30}" width="15" height="12" fill="#42A5F5" rx="2"/>
  <text x="{ml+120}" y="{height-21}" font-size="10" fill="#333">High (+20%)</text>
  <text x="{center_x}" y="{height-21}" text-anchor="middle" font-size="10" fill="#333">Nominal: {nominal:.4f}</text>
'''
        
        svg += '</svg>'
        return svg
    
    @staticmethod
    def pie_chart(data: List[Tuple[str, float]], title: str = "Distribution",
                  width: int = 400, height: int = 300) -> str:
        """Generate pie chart."""
        if not data:
            return f'<svg width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle" fill="#999">No data</text></svg>'
        
        total = sum(d[1] for d in data)
        if total <= 0:
            total = 1
        
        cx, cy = width // 2 - 50, height // 2 + 10
        radius = min(width, height) // 2 - 50
        
        colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2', '#C62828', '#00838F', '#5D4037', '#455A64', '#AFB42B', '#00ACC1']
        
        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" fill="#FAFAFA" rx="8"/>
  <text x="{width//2}" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{title}</text>
'''
        
        angle = -90  # Start at top
        for i, (name, val) in enumerate(data[:10]):
            pct = val / total
            sweep = pct * 360
            
            # Arc path
            start_rad = math.radians(angle)
            end_rad = math.radians(angle + sweep)
            
            x1 = cx + radius * math.cos(start_rad)
            y1 = cy + radius * math.sin(start_rad)
            x2 = cx + radius * math.cos(end_rad)
            y2 = cy + radius * math.sin(end_rad)
            
            large_arc = 1 if sweep > 180 else 0
            
            color = colors[i % len(colors)]
            svg += f'  <path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{radius},{radius} 0 {large_arc},1 {x2:.1f},{y2:.1f} Z" fill="{color}" stroke="white" stroke-width="2"/>\n'
            
            angle += sweep
        
        # Legend
        legend_x = width - 120
        for i, (name, val) in enumerate(data[:8]):
            y = 50 + i * 20
            color = colors[i % len(colors)]
            label = name[:12] + "…" if len(name) > 12 else name
            pct = (val / total) * 100
            svg += f'  <rect x="{legend_x}" y="{y}" width="12" height="12" fill="{color}" rx="2"/>\n'
            svg += f'  <text x="{legend_x+16}" y="{y+10}" font-size="9" fill="#333">{label} ({pct:.1f}%)</text>\n'
        
        svg += '</svg>'
        return svg


class ReportGenerator:
    """Generate comprehensive industrial-grade reliability reports."""
    
    def __init__(self):
        self.svg = SVGChartGenerator()
    
    def _get_css(self) -> str:
        return """
:root {
    --primary: #1976D2;
    --primary-dark: #1565C0;
    --success: #43A047;
    --warning: #FB8C00;
    --error: #E53935;
    --bg: #F5F5F5;
    --card: #FFFFFF;
    --text: #212121;
    --text-secondary: #757575;
    --border: #E0E0E0;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}

.container { max-width: 1400px; margin: 0 auto; padding: 20px; }

.header {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    padding: 30px 40px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(25, 118, 210, 0.3);
}
.header h1 { font-size: 2em; margin-bottom: 8px; }
.header .subtitle { opacity: 0.9; font-size: 1em; }
.header .meta { margin-top: 16px; display: flex; gap: 30px; flex-wrap: wrap; }
.header .meta-item { display: flex; align-items: center; gap: 8px; }
.header .meta-item .label { opacity: 0.8; font-size: 0.85em; }
.header .meta-item .value { font-weight: 600; font-size: 1.1em; }

.card {
    background: var(--card);
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    overflow: hidden;
}
.card-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.card-header h2 { font-size: 1.2em; color: var(--text); display: flex; align-items: center; gap: 10px; }
.card-body { padding: 24px; }

.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
}
.metric {
    background: linear-gradient(135deg, #F8F9FA, #ECEFF1);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    transition: transform 0.2s;
}
.metric:hover { transform: translateY(-2px); }
.metric .value { font-size: 1.8em; font-weight: 700; color: var(--primary); }
.metric .label { color: var(--text-secondary); font-size: 0.85em; margin-top: 4px; }
.metric.success .value { color: var(--success); }
.metric.warning .value { color: var(--warning); }
.metric.error .value { color: var(--error); }

.tabs {
    display: flex;
    border-bottom: 2px solid var(--border);
    margin-bottom: 20px;
    overflow-x: auto;
    scrollbar-width: thin;
}
.tab {
    padding: 12px 24px;
    cursor: pointer;
    border: none;
    background: none;
    font-size: 0.95em;
    color: var(--text-secondary);
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
    white-space: nowrap;
}
.tab:hover { color: var(--primary); background: rgba(25, 118, 210, 0.05); }
.tab.active {
    color: var(--primary);
    border-bottom-color: var(--primary);
    font-weight: 600;
}
.tab-content { display: none; }
.tab-content.active { display: block; }

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
}
th, td {
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
th {
    background: #F8F9FA;
    font-weight: 600;
    color: var(--text-secondary);
    position: sticky;
    top: 0;
}
tr:hover { background: #FAFAFA; }
.fit { font-family: 'SF Mono', Monaco, 'Consolas', monospace; font-size: 0.9em; }
.right { text-align: right; }

.table-container {
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
}

.chart-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    justify-content: center;
}
.chart { background: white; border-radius: 8px; padding: 10px; }

.collapsible {
    cursor: pointer;
    user-select: none;
}
.collapsible::before {
    content: '▼';
    display: inline-block;
    margin-right: 8px;
    transition: transform 0.2s;
}
.collapsible.collapsed::before { transform: rotate(-90deg); }
.collapsible-content { overflow: hidden; transition: max-height 0.3s; }
.collapsible-content.collapsed { max-height: 0 !important; }

.block-card {
    border-left: 4px solid var(--primary);
    margin-bottom: 16px;
}
.block-card.series { border-left-color: #43A047; }
.block-card.parallel { border-left-color: #FB8C00; }
.block-card.k_of_n { border-left-color: #7B1FA2; }

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 500;
}
.badge-success { background: #E8F5E9; color: var(--success); }
.badge-warning { background: #FFF3E0; color: var(--warning); }
.badge-error { background: #FFEBEE; color: var(--error); }
.badge-info { background: #E3F2FD; color: var(--primary); }

.footer {
    text-align: center;
    color: var(--text-secondary);
    padding: 30px;
    font-size: 0.85em;
    border-top: 1px solid var(--border);
    margin-top: 30px;
}

@media print {
    body { padding: 0; background: white; }
    .container { max-width: none; padding: 10px; }
    .card { break-inside: avoid; box-shadow: none; border: 1px solid var(--border); }
    .tabs { display: none; }
    .tab-content { display: block !important; page-break-before: always; }
    .header { background: var(--primary); -webkit-print-color-adjust: exact; }
    .no-print { display: none; }
}

@media (max-width: 768px) {
    .header .meta { flex-direction: column; gap: 12px; }
    .summary-grid { grid-template-columns: repeat(2, 1fr); }
    .tabs { flex-wrap: nowrap; }
    .tab { padding: 10px 16px; font-size: 0.85em; }
}
"""
    
    def _get_js(self) -> str:
        return """
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            const tabGroup = this.closest('.card').querySelectorAll('.tab');
            const contents = this.closest('.card').querySelectorAll('.tab-content');
            
            tabGroup.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            this.classList.add('active');
            const target = document.getElementById(this.dataset.tab);
            if (target) target.classList.add('active');
        });
    });
    
    document.querySelectorAll('.collapsible').forEach(el => {
        el.addEventListener('click', function() {
            this.classList.toggle('collapsed');
            const content = this.nextElementSibling;
            if (content && content.classList.contains('collapsible-content')) {
                content.classList.toggle('collapsed');
            }
        });
    });
    
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', function() {
            const table = this.closest('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const idx = Array.from(this.parentNode.children).indexOf(this);
            const asc = this.dataset.order !== 'asc';
            
            rows.sort((a, b) => {
                const av = a.children[idx].textContent;
                const bv = b.children[idx].textContent;
                const an = parseFloat(av) || av;
                const bn = parseFloat(bv) || bv;
                return asc ? (an > bn ? 1 : -1) : (an < bn ? 1 : -1);
            });
            
            this.dataset.order = asc ? 'asc' : 'desc';
            rows.forEach(r => tbody.appendChild(r));
        });
    });
});
"""
    
    def generate_html(self, data: ReportData) -> str:
        """Generate comprehensive HTML report with tabs and charts."""
        fit = data.system_lambda * 1e9
        mttf_years = data.system_mttf_hours / 8760 if data.system_mttf_hours < float('inf') else float('inf')
        
        if data.system_reliability >= 0.99:
            status_class, status_text = "success", "Excellent"
        elif data.system_reliability >= 0.95:
            status_class, status_text = "warning", "Acceptable"
        else:
            status_class, status_text = "error", "Needs Review"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reliability Report - {data.project_name}</title>
    <style>{self._get_css()}</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>⚡ Reliability Analysis Report</h1>
        <div class="subtitle">IEC TR 62380:2004 Compliant Analysis</div>
        <div class="meta">
            <div class="meta-item">
                <span class="label">Project:</span>
                <span class="value">{data.project_name}</span>
            </div>
            <div class="meta-item">
                <span class="label">Generated:</span>
                <span class="value">{datetime.fromisoformat(data.generated_at).strftime('%Y-%m-%d %H:%M')}</span>
            </div>
            <div class="meta-item">
                <span class="label">Mission:</span>
                <span class="value">{data.mission_years:.1f} years ({data.mission_hours:.0f} h)</span>
            </div>
            <div class="meta-item">
                <span class="label">Profile:</span>
                <span class="value">{data.n_cycles} cycles/yr, ΔT={data.delta_t}°C</span>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">
            <h2>📊 System Summary</h2>
            <span class="badge badge-{status_class}">{status_text}</span>
        </div>
        <div class="card-body">
            <div class="summary-grid">
                <div class="metric {status_class}">
                    <div class="value">{data.system_reliability:.6f}</div>
                    <div class="label">System Reliability</div>
                </div>
                <div class="metric">
                    <div class="value">{fit:.2f}</div>
                    <div class="label">Failure Rate (FIT)</div>
                </div>
                <div class="metric">
                    <div class="value">{data.system_lambda:.2e}</div>
                    <div class="label">λ (per hour)</div>
                </div>
                <div class="metric">
                    <div class="value">{mttf_years:.1f}</div>
                    <div class="label">MTTF (years)</div>
                </div>
                <div class="metric">
                    <div class="value">{len(data.blocks) or len(data.sheets)}</div>
                    <div class="label">Blocks Analyzed</div>
                </div>
                <div class="metric">
                    <div class="value">{sum(len(b.components) for b in data.blocks) if data.blocks else sum(len(s.get('components',[])) for s in data.sheets.values())}</div>
                    <div class="label">Total Components</div>
                </div>
            </div>
        </div>
    </div>
"""
        
        if data.monte_carlo:
            html += self._generate_mc_section(data)
        
        if data.sensitivity:
            html += self._generate_sensitivity_section(data)
        
        html += self._generate_block_tabs(data)
        html += self._generate_contributions_section(data)
        html += self._generate_full_component_list(data)
        
        html += f"""
    <div class="footer">
        <p><strong>KiCad Reliability Calculator v2.0</strong> | IEC TR 62380:2004 Compliant</p>
        <p>This report is for engineering reference only. Critical calculations should be independently verified.</p>
        <p>Generated: {datetime.fromisoformat(data.generated_at).strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</div>
<script>{self._get_js()}</script>
</body>
</html>
"""
        return html
    
    def _generate_mc_section(self, data: ReportData) -> str:
        mc = data.monte_carlo
        html = f"""
    <div class="card">
        <div class="card-header">
            <h2>🎲 Monte Carlo Uncertainty Analysis</h2>
            <span class="badge badge-info">{mc.get('n_simulations', 0):,} simulations</span>
        </div>
        <div class="card-body">
            <div class="summary-grid">
                <div class="metric">
                    <div class="value">{mc.get('mean', 0):.6f}</div>
                    <div class="label">Mean Reliability</div>
                </div>
                <div class="metric">
                    <div class="value">{mc.get('std', 0):.2e}</div>
                    <div class="label">Standard Deviation</div>
                </div>
                <div class="metric">
                    <div class="value">{mc.get('percentile_5', 0):.6f}</div>
                    <div class="label">5th Percentile</div>
                </div>
                <div class="metric">
                    <div class="value">{mc.get('percentile_95', 0):.6f}</div>
                    <div class="label">95th Percentile</div>
                </div>
            </div>
"""
        
        if data.mc_samples and len(data.mc_samples) > 10:
            histogram = self.svg.histogram(data.mc_samples, title="Reliability Distribution", width=700, height=280)
            html += f'''
            <div class="chart-container" style="margin-top: 24px;">
                <div class="chart">{histogram}</div>
            </div>
'''
        
        html += """
            <p style="margin-top: 16px; color: var(--text-secondary); font-size: 0.9em;">
                Monte Carlo simulation propagates parameter uncertainties through IEC TR 62380 physics models.
            </p>
        </div>
    </div>
"""
        return html
    
    def _generate_sensitivity_section(self, data: ReportData) -> str:
        sens = data.sensitivity
        params = sens.get("parameters", [])
        s_first = sens.get("S_first", [])
        s_total = sens.get("S_total", [])
        
        if not params:
            return ""
        
        nominal = data.system_reliability
        tornado_data = []
        for name, sf, st in sorted(zip(params, s_first, s_total), key=lambda x: -x[2])[:10]:
            impact = st * 0.1 * nominal
            tornado_data.append((name, nominal - impact, nominal, nominal + impact))
        
        tornado_svg = self.svg.tornado_diagram(tornado_data, "Parameter Sensitivity", width=700, height=350)
        
        html = f"""
    <div class="card">
        <div class="card-header">
            <h2>📈 Sensitivity Analysis (Sobol Indices)</h2>
        </div>
        <div class="card-body">
            <div class="chart-container">
                <div class="chart">{tornado_svg}</div>
            </div>
            
            <div class="table-container" style="margin-top: 24px; max-height: 300px;">
                <table>
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th class="right">S₁ (First-Order)</th>
                            <th class="right">Sₜ (Total-Order)</th>
                            <th class="right">Interaction</th>
                            <th>Influence</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        ranked = sorted(zip(params, s_first, s_total), key=lambda x: -x[2])
        for name, sf, st in ranked[:15]:
            interaction = st - sf
            pct = st * 100
            bar_w = min(pct * 2, 100)
            html += f"""
                        <tr>
                            <td><strong>{name}</strong></td>
                            <td class="fit right">{sf:.4f}</td>
                            <td class="fit right">{st:.4f}</td>
                            <td class="fit right">{interaction:.4f}</td>
                            <td>
                                <div style="background:#E3F2FD; border-radius:4px; height:16px; width:100px;">
                                    <div style="background:var(--primary); height:100%; width:{bar_w}%; border-radius:4px;"></div>
                                </div>
                            </td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
"""
        return html
    
    def _generate_block_tabs(self, data: ReportData) -> str:
        blocks = data.blocks if data.blocks else []
        if not blocks and data.sheets:
            for path, sheet_data in data.sheets.items():
                components = []
                total_lam = sheet_data.get("lambda", 0)
                for c in sheet_data.get("components", []):
                    lam = c.get("lambda", 0)
                    pct = (lam / total_lam * 100) if total_lam > 0 else 0
                    components.append(ComponentReportData(
                        ref=c.get("ref", "?"),
                        value=c.get("value", ""),
                        component_type=c.get("class", "Unknown"),
                        lambda_fit=lam * 1e9,
                        reliability=c.get("r", 1.0),
                        contribution_pct=pct,
                        parameters=c.get("params", {})
                    ))
                blocks.append(BlockReportData(
                    id=path,
                    name=path.rstrip("/").split("/")[-1] or "Root",
                    path=path,
                    connection_type="series",
                    lambda_total=total_lam,
                    reliability=sheet_data.get("r", 1.0),
                    components=components
                ))
        
        if not blocks:
            return ""
        
        html = """
    <div class="card">
        <div class="card-header">
            <h2>🔧 Block Analysis</h2>
        </div>
        <div class="card-body">
            <div class="tabs">
"""
        
        for i, block in enumerate(blocks[:20]):
            active = "active" if i == 0 else ""
            html += f'                <button class="tab {active}" data-tab="block-{i}">{block.name}</button>\n'
        
        html += """            </div>
"""
        
        for i, block in enumerate(blocks[:20]):
            active = "active" if i == 0 else ""
            fit = block.lambda_total * 1e9 if hasattr(block, 'lambda_total') else 0
            
            html += f'''
            <div id="block-{i}" class="tab-content {active}">
                <div class="block-card card {block.connection_type}">
                    <div class="card-body">
                        <div class="summary-grid" style="margin-bottom: 20px;">
                            <div class="metric">
                                <div class="value">{block.reliability:.6f}</div>
                                <div class="label">Block Reliability</div>
                            </div>
                            <div class="metric">
                                <div class="value">{fit:.2f}</div>
                                <div class="label">λ (FIT)</div>
                            </div>
                            <div class="metric">
                                <div class="value">{len(block.components)}</div>
                                <div class="label">Components</div>
                            </div>
                            <div class="metric">
                                <div class="value">{block.connection_type.upper()}</div>
                                <div class="label">Connection</div>
                            </div>
                        </div>
                        
                        <p style="color: var(--text-secondary); margin-bottom: 16px;">
                            <strong>Path:</strong> <code>{block.path}</code>
                        </p>
'''
            
            if block.components:
                pareto_data = [(c.ref, c.lambda_fit) for c in sorted(block.components, key=lambda x: -x.lambda_fit)[:15]]
                pareto_svg = self.svg.pareto_chart(pareto_data, f"Component Contributions - {block.name}", width=650, height=280)
                html += f'''
                        <div class="chart-container" style="margin-bottom: 20px;">
                            <div class="chart">{pareto_svg}</div>
                        </div>
'''
            
            html += '''
                        <div class="table-container">
                            <table>
                                <thead>
                                    <tr>
                                        <th data-sort="ref">Reference</th>
                                        <th data-sort="value">Value</th>
                                        <th data-sort="type">Type</th>
                                        <th data-sort="lambda" class="right">λ (FIT)</th>
                                        <th data-sort="r" class="right">Reliability</th>
                                        <th data-sort="contrib" class="right">Contribution</th>
                                    </tr>
                                </thead>
                                <tbody>
'''
            
            for comp in sorted(block.components, key=lambda x: -x.lambda_fit):
                html += f'''
                                    <tr>
                                        <td><strong>{comp.ref}</strong></td>
                                        <td>{comp.value}</td>
                                        <td>{comp.component_type}</td>
                                        <td class="fit right">{comp.lambda_fit:.2f}</td>
                                        <td class="fit right">{comp.reliability:.6f}</td>
                                        <td class="right">{comp.contribution_pct:.1f}%</td>
                                    </tr>
'''
            
            html += '''
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
'''
        
        html += """
        </div>
    </div>
"""
        return html
    
    def _generate_contributions_section(self, data: ReportData) -> str:
        all_components = []
        for block in data.blocks:
            for comp in block.components:
                all_components.append((f"{block.name}/{comp.ref}", comp.lambda_fit, comp.component_type))
        
        if not all_components and data.sheets:
            for path, sheet_data in data.sheets.items():
                sheet_name = path.rstrip("/").split("/")[-1] or "Root"
                for c in sheet_data.get("components", []):
                    all_components.append((
                        f"{sheet_name}/{c.get('ref', '?')}",
                        c.get('lambda', 0) * 1e9,
                        c.get('class', 'Unknown')
                    ))
        
        if not all_components:
            return ""
        
        all_components.sort(key=lambda x: -x[1])
        total = sum(c[1] for c in all_components)
        
        pie_data = [(c[0], c[1]) for c in all_components[:10]]
        pie_svg = self.svg.pie_chart(pie_data, "Top 10 Contributors", width=400, height=300)
        
        pareto_data = [(c[0], c[1]) for c in all_components[:20]]
        pareto_svg = self.svg.pareto_chart(pareto_data, "Pareto Analysis - All Components", width=750, height=320)
        
        html = f"""
    <div class="card">
        <div class="card-header">
            <h2>📊 Contribution Analysis</h2>
        </div>
        <div class="card-body">
            <div class="chart-container">
                <div class="chart">{pie_svg}</div>
                <div class="chart">{pareto_svg}</div>
            </div>
            
            <h3 class="collapsible" style="margin-top: 24px; cursor: pointer;">▼ Top 30 Contributors</h3>
            <div class="collapsible-content">
                <div class="table-container" style="max-height: 400px;">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Component</th>
                                <th>Type</th>
                                <th class="right">λ (FIT)</th>
                                <th class="right">Contribution</th>
                                <th class="right">Cumulative</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        cumulative = 0
        for i, (name, lam, ctype) in enumerate(all_components[:30]):
            pct = (lam / total * 100) if total > 0 else 0
            cumulative += pct
            html += f"""
                            <tr>
                                <td>{i+1}</td>
                                <td><strong>{name}</strong></td>
                                <td>{ctype}</td>
                                <td class="fit right">{lam:.2f}</td>
                                <td class="right">{pct:.1f}%</td>
                                <td class="right">{cumulative:.1f}%</td>
                            </tr>
"""
        
        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
"""
        return html
    
    def _generate_full_component_list(self, data: ReportData) -> str:
        all_components = []
        
        for block in data.blocks:
            for comp in block.components:
                all_components.append((block.path, comp))
        
        if not all_components and data.sheets:
            for path, sheet_data in data.sheets.items():
                for c in sheet_data.get("components", []):
                    comp = ComponentReportData(
                        ref=c.get("ref", "?"),
                        value=c.get("value", ""),
                        component_type=c.get("class", "Unknown"),
                        lambda_fit=c.get("lambda", 0) * 1e9,
                        reliability=c.get("r", 1.0),
                        contribution_pct=0,
                        parameters=c.get("params", {})
                    )
                    all_components.append((path, comp))
        
        if not all_components:
            return ""
        
        html = """
    <div class="card">
        <div class="card-header">
            <h2>📋 Complete Component Database</h2>
            <span class="badge badge-info">""" + str(len(all_components)) + """ components</span>
        </div>
        <div class="card-body">
            <div class="table-container" style="max-height: 600px;">
                <table>
                    <thead>
                        <tr>
                            <th data-sort>Sheet</th>
                            <th data-sort>Reference</th>
                            <th data-sort>Value</th>
                            <th data-sort>Type</th>
                            <th data-sort class="right">λ (FIT)</th>
                            <th data-sort class="right">Reliability</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for path, comp in sorted(all_components, key=lambda x: -x[1].lambda_fit):
            sheet_name = path.rstrip("/").split("/")[-1] or "Root"
            html += f"""
                        <tr>
                            <td>{sheet_name}</td>
                            <td><strong>{comp.ref}</strong></td>
                            <td>{comp.value}</td>
                            <td>{comp.component_type}</td>
                            <td class="fit right">{comp.lambda_fit:.2f}</td>
                            <td class="fit right">{comp.reliability:.6f}</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
"""
        return html
    
    def generate_markdown(self, data: ReportData) -> str:
        fit = data.system_lambda * 1e9
        mttf_years = data.system_mttf_hours / 8760 if data.system_mttf_hours < float('inf') else float('inf')
        
        md = f"""# Reliability Analysis Report

**Project:** {data.project_name}  
**Generated:** {datetime.fromisoformat(data.generated_at).strftime('%Y-%m-%d %H:%M')}  
**Standard:** IEC TR 62380:2004

---

## System Summary

| Metric | Value |
|--------|-------|
| **System Reliability** | {data.system_reliability:.6f} |
| **Failure Rate** | {fit:.2f} FIT |
| **MTTF** | {mttf_years:.1f} years |
| **Mission Duration** | {data.mission_years:.1f} years |

"""
        
        if data.monte_carlo:
            mc = data.monte_carlo
            md += f"""
## Monte Carlo Analysis

| Metric | Value |
|--------|-------|
| Mean | {mc.get('mean', 0):.6f} |
| Std Dev | {mc.get('std', 0):.2e} |
| P5-P95 | [{mc.get('percentile_5', 0):.6f}, {mc.get('percentile_95', 0):.6f}] |

"""
        
        for path, sheet_data in sorted(data.sheets.items()):
            sheet_fit = sheet_data.get("lambda", 0) * 1e9
            md += f"""
### {path.rstrip("/").split("/")[-1] or "Root"}

**λ:** {sheet_fit:.2f} FIT | **R:** {sheet_data.get("r", 1.0):.6f}

| Ref | Value | Type | λ (FIT) |
|-----|-------|------|---------|
"""
            for c in sheet_data.get("components", [])[:15]:
                md += f"| {c.get('ref', '?')} | {c.get('value', '')} | {c.get('class', '')} | {c.get('lambda', 0)*1e9:.2f} |\n"
        
        return md
    
    def generate_csv(self, data: ReportData) -> str:
        lines = ["Sheet,Reference,Value,Type,Lambda_FIT,Reliability"]
        for path, sheet_data in sorted(data.sheets.items()):
            for c in sheet_data.get("components", []):
                lines.append(f'"{path}","{c.get("ref","")}","{c.get("value","")}","{c.get("class","")}",{c.get("lambda",0)*1e9:.4f},{c.get("r",1.0):.8f}')
        return "\n".join(lines)
    
    def generate_json(self, data: ReportData) -> str:
        return json.dumps({
            "meta": {"project": data.project_name, "generated": data.generated_at, "standard": "IEC TR 62380:2004"},
            "mission": {"hours": data.mission_hours, "years": data.mission_years, "cycles": data.n_cycles, "delta_t": data.delta_t},
            "system": {"reliability": data.system_reliability, "lambda_fit": data.system_lambda * 1e9, "mttf_hours": data.system_mttf_hours},
            "sheets": data.sheets,
            "monte_carlo": data.monte_carlo,
            "sensitivity": data.sensitivity,
        }, indent=2, default=str)
    
    def generate(self, data: ReportData, format: str = "html") -> str:
        return {"html": self.generate_html, "markdown": self.generate_markdown, "md": self.generate_markdown, "csv": self.generate_csv, "json": self.generate_json}.get(format.lower(), self.generate_html)(data)
