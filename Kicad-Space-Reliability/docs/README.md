# KiCad Reliability Calculator Plugin v2.0

**IEC TR 62380 compliant reliability analysis for KiCad projects**

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![KiCad](https://img.shields.io/badge/KiCad-8.x%20%7C%209.x-green)
![Standard](https://img.shields.io/badge/Standard-IEC%20TR%2062380-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Overview

The Reliability Calculator plugin provides comprehensive reliability analysis for electronic systems designed in KiCad. It implements the IEC TR 62380 standard for predicting failure rates of electronic components and systems.

### Key Features

- **Component-Level Analysis**: Calculate failure rates (λ) for ICs, discretes, passives, connectors, and more
- **System Reliability**: Build block diagrams with series, parallel, and k-of-n redundancy configurations
- **Visual Block Editor**: Drag-and-drop interface for defining system topology
- **Monte Carlo Analysis**: Uncertainty quantification with automatic convergence detection
- **Sensitivity Analysis**: Sobol indices to identify critical parameters
- **Professional Reports**: Export to HTML, Markdown, CSV, and JSON formats

### What's New in v2.0

- **Configurable EOS**: 10 interface types with realistic overstress parameters
- **Working Time Ratio (τ_on)**: Support for duty-cycled and intermittent operation
- **Thermal Expansion**: Substrate/package CTE mismatch calculations
- **Enhanced Reports**: Monte Carlo and sensitivity results, component breakdowns
- **Improved UI**: New parameter editors for all IEC TR 62380 features

## Installation

### Method 1: KiCad Plugin Manager (Recommended)

1. Open KiCad
2. Go to **Plugin and Content Manager**
3. Search for "Reliability Calculator"
4. Click **Install**

### Method 2: Manual Installation

Copy the plugin folder to your KiCad plugins directory:

| Platform | Path |
|----------|------|
| **Linux** | `~/.local/share/kicad/9.0/scripting/plugins/` |
| **Windows** | `%APPDATA%\kicad\9.0\scripting\plugins\` |
| **macOS** | `~/Library/Preferences/kicad/9.0/scripting/plugins/` |

```bash
# Linux example
cp -r kicad_reliability_plugin_v2 ~/.local/share/kicad/9.0/scripting/plugins/
```

### Method 3: Standalone Usage

Run without KiCad for development or testing:

```bash
python -m kicad_reliability_plugin_v2 [project_path]
```

## Quick Start

### 1. Launch the Plugin

From KiCad PCB Editor: **Tools → External Plugins → Reliability Calculator**

### 2. Load Your Project

Click **Open...** and select your KiCad project directory containing `.kicad_sch` files.

### 3. Add Sheets to Block Diagram

Select schematic sheets from the left panel and click **Add Selected** to create reliability blocks.

### 4. Configure System Topology

- **Drag** blocks to arrange them
- **Shift+Drag** to select multiple blocks
- **Right-click** selection to group as Series/Parallel/K-of-N
- **Double-click** groups to change connection type

### 5. Edit Component Parameters

- **Double-click** a sheet block to edit all components
- Or select a component in the details panel and click **Edit**

### 6. Run Analysis

Click **Recalculate** to compute system reliability. Export reports via **Export** button.

## Component Parameters

### Mission Profile

| Parameter | Description | Default |
|-----------|-------------|---------|
| Mission Duration | Total operating time | 5 years |
| Thermal Cycles/Year | Number of temperature cycles | 5256 (LEO satellite) |
| ΔT per Cycle | Temperature swing | 3°C |

### Component Fields

Add these fields to your KiCad symbols for accurate calculations:

| Field | Description | Example |
|-------|-------------|---------|
| `Reliability_Class` | Component type | "Integrated Circuit" |
| `T_Junction` | Junction temperature (°C) | 85 |
| `T_Ambient` | Ambient temperature (°C) | 25 |
| `Operating_Power` | Power dissipation (W) | 0.01 |
| `Rated_Power` | Maximum rated power (W) | 0.125 |
| `Interface_Type` | EOS interface category | "Avionics" |
| `Is_Interface` | Component is at interface | Yes/No |
| `Tau_On` | Working time ratio (0-1) | 1.0 |
| `Substrate` | PCB material | "FR4 (Epoxy Glass)" |

### Interface Types (EOS)

| Type | π_i | λ_EOS (FIT) | Use Case |
|------|-----|-------------|----------|
| Not Interface | 0 | 0 | Internal components |
| Computer | 1 | 10 | Data interfaces |
| Telecom (Switching) | 1 | 15 | Network equipment |
| Telecom (Subscriber) | 1 | 70 | End-user equipment |
| Avionics | 1 | 20 | Aircraft systems |
| Power Supply | 1 | 40 | Power distribution |
| Industrial | 1 | 30 | Factory automation |
| Automotive | 1 | 50 | Vehicle electronics |
| Space (LEO) | 1 | 25 | Low Earth orbit |
| Space (GEO) | 1 | 35 | Geostationary orbit |

## Mathematical Model

### Failure Rate Calculation

The total failure rate follows IEC TR 62380:

```
λ_total = λ_die + λ_package + λ_EOS
```

Where:
- **λ_die**: Die contribution with temperature acceleration (Arrhenius)
- **λ_package**: Package/solder joint contribution with thermal cycling
- **λ_EOS**: Electrical overstress contribution for interface circuits

### Temperature Factor (Arrhenius)

```
π_T = exp(E_a × (1/T_ref - 1/(273 + T_j)))
```

### Thermal Cycling Factor

```
π_n = n^0.76           for n ≤ 8760 cycles/year
π_n = 1.7 × n^0.6      for n > 8760 cycles/year
```

### Working Time Ratio

```
λ_effective = λ_base × τ_on
```

Where τ_on ∈ [0, 1] represents the fraction of time the component is operating.

### System Reliability

| Configuration | Formula |
|---------------|---------|
| Series | R = R₁ × R₂ × ... × Rₙ |
| Parallel | R = 1 - (1-R₁) × (1-R₂) × ... × (1-Rₙ) |
| K-of-N | R = Σ C(n,i) × Rⁱ × (1-R)^(n-i) for i=k to n |

## Advanced Features

### Monte Carlo Analysis

Quantify uncertainty in reliability predictions:

```python
from kicad_reliability_plugin_v2 import MonteCarloAnalyzer, quick_monte_carlo

# Quick analysis with ±20% uncertainty
result = quick_monte_carlo(lambda_base=1e-7, uncertainty_pct=20)
print(f"Mean: {result.mean:.2e}, 95th percentile: {result.percentile_95:.2e}")

# Full analysis with custom distributions
analyzer = MonteCarloAnalyzer()
result = analyzer.run_analysis(
    calc_function=my_reliability_function,
    distributions={'temperature': ParameterDistribution('normal', 85, 5)},
    n_samples=10000
)
```

### Sensitivity Analysis

Identify which parameters most affect reliability:

```python
from kicad_reliability_plugin_v2 import SobolAnalyzer, quick_sensitivity

# Sobol sensitivity analysis
analyzer = SobolAnalyzer()
result = analyzer.analyze(
    model_function=my_model,
    param_bounds={
        'temperature': (60, 100),
        'n_cycles': (3000, 8000),
        'delta_t': (1, 10)
    },
    n_samples=1024
)

# Print results
for param, s1 in result.S_first.items():
    print(f"{param}: S1={s1:.3f}, ST={result.S_total[param]:.3f}")
```

### Programmatic Usage

```python
from kicad_reliability_plugin_v2 import (
    lambda_integrated_circuit,
    lambda_capacitor,
    reliability_from_lambda,
    r_series
)

# Calculate IC failure rate
ic_result = lambda_integrated_circuit(
    ic_type="MOS_DIGITAL",
    transistor_count=100000,
    t_junction=85,
    package_type="TQFP-10x10",
    interface_type="Avionics",
    is_interface=True,
    tau_on=0.8,  # 80% duty cycle
    n_cycles=5256,
    delta_t=3.0
)
print(f"IC λ: {ic_result['fit_total']:.2f} FIT")

# Calculate capacitor failure rate
cap_result = lambda_capacitor(
    capacitor_type="Ceramic Class II (X7R/X5R)",
    t_ambient=25,
    n_cycles=5256,
    delta_t=3.0,
    tau_on=1.0
)

# System reliability (5 year mission)
mission_hours = 5 * 365 * 24
r_ic = reliability_from_lambda(ic_result['lambda_total'], mission_hours)
r_cap = reliability_from_lambda(cap_result['lambda_total'], mission_hours)
r_system = r_series([r_ic, r_cap])
print(f"System R: {r_system:.6f}")
```

## Report Formats

### HTML Report

Professional styled report with:
- System summary metrics
- Monte Carlo results (if available)
- Sensitivity analysis (if available)
- Per-sheet component tables
- Color-coded status indicators

### Markdown Report

Portable format suitable for:
- Git repositories
- Documentation systems
- Wiki pages

### CSV Export

Spreadsheet-compatible format with columns:
- Sheet, Reference, Value, Type, Lambda_FIT, Reliability

### JSON Export

Structured data for:
- Programmatic processing
- Database import
- API integration

## Troubleshooting

### Plugin Not Appearing in KiCad

1. Verify installation path is correct for your KiCad version
2. Check KiCad scripting console for errors: **View → Scripting Console**
3. Ensure wxPython is installed: `pip install wxPython`

### Import Errors

```bash
# Install dependencies
pip install numpy scipy wxPython
```

### Schematic Not Parsing

- Ensure `.kicad_sch` files are in KiCad 6+ format
- Check file permissions
- Verify project structure (`.kicad_pro` file present)

## API Reference

See `docs/API.md` for complete API documentation.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- IEC TR 62380:2004 - Reliability data handbook
- MIL-HDBK-217F - Military handbook for reliability prediction
- FIDES Guide 2009 - Reliability methodology for electronic systems

## Changelog

### v2.0.0
- Added configurable EOS parameters with 10 interface types
- Added tau_on (working time ratio) for duty-cycled operation
- Added thermal expansion mismatch with material databases
- Added Monte Carlo uncertainty analysis
- Added Sobol sensitivity analysis
- Enhanced report generation with new metrics
- Improved UI with parameter editors for all IEC TR 62380 features
- Fixed π_n threshold calculation (8760 cycles/year boundary)

### v1.0.0
- Initial release with basic IEC TR 62380 implementation
- Visual block diagram editor
- Basic report generation
