"""
Enhanced Report Generator
=========================
Professional reliability reports in HTML, Markdown, CSV, and JSON formats.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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
    
    sheets: Dict[str, Dict]  # {path: {components, lambda, r}}
    blocks: List[Dict]  # Block diagram structure
    
    monte_carlo: Optional[Dict] = None
    sensitivity: Optional[Dict] = None
    
    generated_at: str = None
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


class ReportGenerator:
    """Generate professional reliability reports."""
    
    def __init__(self):
        self.css_style = self._get_css()
    
    def _get_css(self) -> str:
        return """
        :root {
            --primary: #1976D2;
            --success: #43A047;
            --warning: #FB8C00;
            --error: #E53935;
            --bg: #FAFAFA;
            --card: #FFFFFF;
            --text: #212121;
            --text-secondary: #757575;
            --border: #E0E0E0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: var(--bg); color: var(--text); line-height: 1.6; padding: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, var(--primary), #1565C0);
                  color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }
        .header h1 { font-size: 2.2em; margin-bottom: 10px; }
        .header .subtitle { opacity: 0.9; font-size: 1.1em; }
        .card { background: var(--card); border-radius: 12px; padding: 24px;
                margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card h2 { color: var(--primary); margin-bottom: 16px; font-size: 1.4em;
                   border-bottom: 2px solid var(--border); padding-bottom: 8px; }
        .card h3 { color: var(--text); margin: 20px 0 12px; font-size: 1.1em; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric { background: #F5F5F5; padding: 20px; border-radius: 8px; text-align: center; }
        .metric .value { font-size: 2em; font-weight: bold; color: var(--primary); }
        .metric .label { color: var(--text-secondary); font-size: 0.9em; margin-top: 4px; }
        .metric.success .value { color: var(--success); }
        .metric.warning .value { color: var(--warning); }
        table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.95em; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: #F5F5F5; font-weight: 600; color: var(--text-secondary); }
        tr:hover { background: #FAFAFA; }
        .fit { font-family: 'SF Mono', Monaco, monospace; }
        .bar { height: 8px; background: #E0E0E0; border-radius: 4px; overflow: hidden; }
        .bar-fill { height: 100%; background: var(--primary); border-radius: 4px; }
        .chart { margin: 20px 0; }
        .footer { text-align: center; color: var(--text-secondary); padding: 20px; font-size: 0.85em; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 500; }
        .badge-success { background: #E8F5E9; color: var(--success); }
        .badge-warning { background: #FFF3E0; color: var(--warning); }
        .badge-error { background: #FFEBEE; color: var(--error); }
        @media print {
            body { padding: 20px; }
            .card { break-inside: avoid; box-shadow: none; border: 1px solid var(--border); }
        }
        """
    
    def generate_html(self, data: ReportData) -> str:
        """Generate comprehensive HTML report."""
        fit = data.system_lambda * 1e9
        mttf_years = data.system_mttf_hours / 8760 if data.system_mttf_hours < float('inf') else float('inf')
        
        # Determine reliability status
        if data.system_reliability >= 0.99:
            status_class, status_text = "success", "Excellent"
        elif data.system_reliability >= 0.95:
            status_class, status_text = "warning", "Acceptable"
        else:
            status_class, status_text = "error", "Review Required"
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reliability Analysis Report - {data.project_name}</title>
    <style>{self.css_style}</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>⚡ Reliability Analysis Report</h1>
        <div class="subtitle">
            Project: <strong>{data.project_name}</strong> | 
            Generated: {datetime.fromisoformat(data.generated_at).strftime('%Y-%m-%d %H:%M')} |
            Standard: IEC TR 62380:2004
        </div>
    </div>
    
    <div class="card">
        <h2>📊 System Summary</h2>
        <div class="summary-grid">
            <div class="metric {status_class}">
                <div class="value">{data.system_reliability:.6f}</div>
                <div class="label">System Reliability</div>
            </div>
            <div class="metric">
                <div class="value">{fit:.1f}</div>
                <div class="label">Failure Rate (FIT)</div>
            </div>
            <div class="metric">
                <div class="value">{mttf_years:.1f}</div>
                <div class="label">MTTF (years)</div>
            </div>
            <div class="metric">
                <div class="value">{data.mission_years:.1f}</div>
                <div class="label">Mission (years)</div>
            </div>
        </div>
        <p style="margin-top: 20px;">
            <span class="badge badge-{status_class}">{status_text}</span>
            Mission profile: {data.n_cycles} cycles/year, ΔT = {data.delta_t}°C
        </p>
    </div>
"""
        
        # Monte Carlo results if available
        if data.monte_carlo:
            mc = data.monte_carlo
            html += f"""
    <div class="card">
        <h2>🎲 Monte Carlo Uncertainty Analysis</h2>
        <div class="summary-grid">
            <div class="metric">
                <div class="value">{mc.get('mean', 0):.6f}</div>
                <div class="label">Mean Reliability</div>
            </div>
            <div class="metric">
                <div class="value">{mc.get('std', 0):.6f}</div>
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
        <p style="margin-top: 16px; color: var(--text-secondary);">
            Based on {mc.get('n_simulations', 0):,} simulations
            {"(converged)" if mc.get('converged') else "(max iterations reached)"}
        </p>
    </div>
"""
        
        # Sensitivity results if available
        if data.sensitivity:
            sens = data.sensitivity
            html += """
    <div class="card">
        <h2>📈 Sensitivity Analysis (Sobol Indices)</h2>
        <table>
            <thead>
                <tr><th>Parameter</th><th>First-Order (S₁)</th><th>Total-Order (Sₜ)</th><th>Influence</th></tr>
            </thead>
            <tbody>
"""
            params = sens.get("parameters", [])
            s_first = sens.get("S_first", [])
            s_total = sens.get("S_total", [])
            
            # Sort by total index
            ranked = sorted(zip(params, s_first, s_total), key=lambda x: -x[2])[:10]
            
            for name, sf, st in ranked:
                pct = st * 100
                html += f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td class="fit">{sf:.4f}</td>
                    <td class="fit">{st:.4f}</td>
                    <td>
                        <div class="bar" style="width: 150px;">
                            <div class="bar-fill" style="width: {min(pct, 100):.1f}%;"></div>
                        </div>
                    </td>
                </tr>
"""
            
            html += """
            </tbody>
        </table>
        <p style="color: var(--text-secondary); font-size: 0.9em;">
            Higher values indicate greater influence on system reliability uncertainty.
        </p>
    </div>
"""
        
        # Sheet-by-sheet breakdown
        html += """
    <div class="card">
        <h2>📋 Sheet Analysis</h2>
"""
        
        for path, sheet_data in sorted(data.sheets.items()):
            sheet_fit = sheet_data.get("lambda", 0) * 1e9
            sheet_r = sheet_data.get("r", 1.0)
            components = sheet_data.get("components", [])
            
            sheet_name = path.rstrip("/").split("/")[-1] or "Root"
            
            html += f"""
        <h3>{sheet_name}</h3>
        <p style="color: var(--text-secondary); margin-bottom: 12px;">
            Path: <code>{path}</code> | λ = {sheet_fit:.2f} FIT | R = {sheet_r:.6f}
        </p>
        <table>
            <thead>
                <tr><th>Reference</th><th>Value</th><th>Type</th><th>λ (FIT)</th><th>R</th></tr>
            </thead>
            <tbody>
"""
            
            for comp in components[:20]:  # Limit to 20 per sheet
                c_fit = comp.get("lambda", 0) * 1e9
                c_r = comp.get("r", 1.0)
                html += f"""
                <tr>
                    <td><strong>{comp.get('ref', '?')}</strong></td>
                    <td>{comp.get('value', '')}</td>
                    <td>{comp.get('class', 'Unknown')}</td>
                    <td class="fit">{c_fit:.2f}</td>
                    <td class="fit">{c_r:.6f}</td>
                </tr>
"""
            
            if len(components) > 20:
                html += f'<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">... and {len(components) - 20} more components</td></tr>'
            
            html += """
            </tbody>
        </table>
"""
        
        html += """
    </div>
    
    <div class="footer">
        <p>Generated by KiCad Reliability Calculator v2.0 | IEC TR 62380:2004 Compliant</p>
        <p>This report is for engineering reference only. Verify critical calculations independently.</p>
    </div>
</div>
</body>
</html>
"""
        return html
    
    def generate_markdown(self, data: ReportData) -> str:
        """Generate Markdown report."""
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
| **Thermal Cycles** | {data.n_cycles}/year |
| **Temperature Swing** | {data.delta_t}°C |

"""
        
        if data.monte_carlo:
            mc = data.monte_carlo
            md += f"""
## Monte Carlo Uncertainty Analysis

| Metric | Value |
|--------|-------|
| Mean Reliability | {mc.get('mean', 0):.6f} |
| Standard Deviation | {mc.get('std', 0):.6f} |
| 5th Percentile | {mc.get('percentile_5', 0):.6f} |
| 95th Percentile | {mc.get('percentile_95', 0):.6f} |
| Simulations | {mc.get('n_simulations', 0):,} |

"""
        
        if data.sensitivity:
            sens = data.sensitivity
            md += """
## Sensitivity Analysis (Sobol Indices)

| Parameter | S₁ (First) | Sₜ (Total) |
|-----------|------------|------------|
"""
            params = sens.get("parameters", [])
            s_first = sens.get("S_first", [])
            s_total = sens.get("S_total", [])
            ranked = sorted(zip(params, s_first, s_total), key=lambda x: -x[2])[:10]
            for name, sf, st in ranked:
                md += f"| {name} | {sf:.4f} | {st:.4f} |\n"
        
        md += "\n## Sheet Analysis\n"
        
        for path, sheet_data in sorted(data.sheets.items()):
            sheet_fit = sheet_data.get("lambda", 0) * 1e9
            sheet_r = sheet_data.get("r", 1.0)
            components = sheet_data.get("components", [])
            sheet_name = path.rstrip("/").split("/")[-1] or "Root"
            
            md += f"""
### {sheet_name}

**Path:** `{path}` | **λ:** {sheet_fit:.2f} FIT | **R:** {sheet_r:.6f}

| Ref | Value | Type | λ (FIT) | R |
|-----|-------|------|---------|---|
"""
            for comp in components[:15]:
                c_fit = comp.get("lambda", 0) * 1e9
                md += f"| {comp.get('ref', '?')} | {comp.get('value', '')} | {comp.get('class', '')} | {c_fit:.2f} | {comp.get('r', 1.0):.6f} |\n"
        
        md += """
---

*Generated by KiCad Reliability Calculator v2.0*
"""
        return md
    
    def generate_csv(self, data: ReportData) -> str:
        """Generate CSV report."""
        lines = ["Sheet,Reference,Value,Type,Lambda_FIT,Reliability"]
        
        for path, sheet_data in sorted(data.sheets.items()):
            for comp in sheet_data.get("components", []):
                c_fit = comp.get("lambda", 0) * 1e9
                line = f'"{path}","{comp.get("ref", "")}","{comp.get("value", "")}","{comp.get("class", "")}",{c_fit:.4f},{comp.get("r", 1.0):.8f}'
                lines.append(line)
        
        return "\n".join(lines)
    
    def generate_json(self, data: ReportData) -> str:
        """Generate JSON report."""
        output = {
            "meta": {
                "project": data.project_name,
                "generated": data.generated_at,
                "standard": "IEC TR 62380:2004",
                "version": "2.0.0",
            },
            "mission": {
                "hours": data.mission_hours,
                "years": data.mission_years,
                "thermal_cycles_per_year": data.n_cycles,
                "delta_t_celsius": data.delta_t,
            },
            "system": {
                "reliability": data.system_reliability,
                "lambda_per_hour": data.system_lambda,
                "lambda_fit": data.system_lambda * 1e9,
                "mttf_hours": data.system_mttf_hours,
            },
            "sheets": data.sheets,
        }
        
        if data.monte_carlo:
            output["monte_carlo"] = data.monte_carlo
        
        if data.sensitivity:
            output["sensitivity"] = data.sensitivity
        
        return json.dumps(output, indent=2, default=str)
    
    def generate(self, data: ReportData, format: str = "html") -> str:
        """Generate report in specified format."""
        generators = {
            "html": self.generate_html,
            "markdown": self.generate_markdown,
            "md": self.generate_markdown,
            "csv": self.generate_csv,
            "json": self.generate_json,
        }
        
        gen = generators.get(format.lower(), self.generate_html)
        return gen(data)


if __name__ == "__main__":
    # Quick test
    data = ReportData(
        project_name="Test Project",
        mission_hours=43800,
        mission_years=5.0,
        n_cycles=5256,
        delta_t=3.0,
        system_reliability=0.9847,
        system_lambda=3.5e-9,
        system_mttf_hours=285714285,
        sheets={"/Power/": {"components": [{"ref": "U1", "value": "LM7805", "class": "LDO", "lambda": 1e-9, "r": 0.99}], "lambda": 1e-9, "r": 0.99}},
        blocks=[],
    )
    
    gen = ReportGenerator()
    print(gen.generate(data, "markdown")[:1000])
