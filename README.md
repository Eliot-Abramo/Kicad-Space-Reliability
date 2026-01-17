# KiCad Space Reliability Calculator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![KiCad: 9.0+](https://img.shields.io/badge/KiCad-9.0+-green.svg)](https://www.kicad.org/)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

**IEC TR 62380 Reliability Prediction for Electronic Assemblies**

A professional-grade KiCad plugin for calculating system reliability based on the IEC TR 62380 standard. Designed for space, aerospace, and high-reliability electronics applications.

## Features

### Visual Block Diagram Editor
- **Zoomable Canvas**: Mouse wheel zoom (25%-300%), centered on cursor
- **Pan Navigation**: Middle-mouse-button drag or arrow keys
- **Fit to View**: Automatically fit all blocks (F key)
- **Drag-and-Drop**: Organize blocks visually on a grid
- **Selection Grouping**: Drag-select multiple blocks, right-click to group
- **Redundancy Configurations**: Series, Parallel, or K-of-N grouping

### Component Field Editor
- **No ECSS Reference Required**: All parameters via dropdown menus with help text
- **Auto-Classification**: Determines component type from reference designators (R*, C*, U*, Q*, D*, etc.)
- **Real-time Preview**: See calculated failure rates as you edit parameters
- **Batch Editing**: Edit all components at once or by schematic sheet

### IEC TR 62380 Calculations
All formulas implemented per the standard:
- **Temperature Factors (πt)**: Arrhenius model with correct activation energies
- **Thermal Cycling Factors (πn)**: Based on annual cycle count
- **Package Factors**: Complete Table 17a/17b implementation
- **Interface/Overstress (λEOS)**: Environment-specific overstress rates

### Centralized Math Module
All calculations in `reliability_math.py` for easy tuning:
- Modify failure rate constants
- Adjust activation energies
- Tune package stress factors
- Add new component types

## Installation

### KiCad Plugin Installation

Copy this folder to your KiCad plugins directory:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/kicad/9.0/scripting/plugins/` |
| Windows | `%APPDATA%\kicad\9.0\scripting\plugins\` |
| macOS | `~/Library/Preferences/kicad/9.0/scripting/plugins/` |

Then restart KiCad.

### Standalone Usage (for testing)

```bash
cd /path/to/Kicad-Space-Reliability
python reliability_launcher.py [project_path]
```

## Usage

### Quick Start
1. Open your KiCad project
2. Launch the plugin from **Tools → External Plugins → Reliability Calculator**
3. Add schematic sheets to the block diagram (left panel)
4. Arrange blocks and group them (Series/Parallel/K-of-N)
5. Double-click components to edit reliability fields
6. View calculated system reliability in the Results panel

### Keyboard Shortcuts (Block Editor)

| Key | Action |
|-----|--------|
| Mouse Wheel | Zoom in/out (centered on cursor) |
| Middle Mouse Button | Pan view |
| Arrow Keys | Pan view |
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `F` | Fit all blocks in view |
| `0` | Reset zoom to 100% |
| `Ctrl+0` | Reset view (zoom + pan) |
| `Delete` | Remove selected block / Ungroup |

### Editing Component Fields
Each component type has specific fields:

**Integrated Circuits:**
- IC Type (Microcontroller, FPGA, Op-Amp, etc.)
- Transistor Count
- Package Type (SOIC, QFP, BGA, etc.)
- Junction Temperature
- Interface Type (for protection circuits)

**Transistors:**
- Technology (BJT, MOSFET, IGBT)
- Power Class (Low ≤5W, High >5W)
- Voltage Stress Ratios (VDS/VGS or VCE)
- Package Type

**Diodes:**
- Type (Signal, Zener, TVS, Schottky, LED)
- Power Class
- Package Type

**Capacitors:**
- Type (Ceramic Class I/II, Tantalum, Aluminum)
- Ambient Temperature
- Ripple Current Ratio (for electrolytics)

**Resistors:**
- Type (SMD, Film, Wirewound)
- Operating/Rated Power
- Ambient Temperature

### Mission Profile Settings
- **Mission Duration**: 1-30 years
- **Annual Thermal Cycles**: LEO satellite default is 5256/year
- **Temperature Swing (ΔT)**: Per-cycle temperature change

## Project Structure

```
Kicad-Space-Reliability/
├── __init__.py              # Plugin registration
├── plugin.py                # KiCad ActionPlugin interface
├── reliability_dialog.py    # Main UI dialog
├── block_editor.py          # Visual block diagram editor (with zoom/pan)
├── reliability_math.py      # IEC TR 62380 calculations (all formulas here)
├── reliability_core.py      # Backward compatibility exports
├── component_editor.py      # Component field editing dialogs
├── schematic_parser.py      # KiCad schematic file parser
├── table_generator.py       # KiCad table/text box generation
├── ecss_fields.py           # ECSS field definitions
├── reliability_launcher.py  # Standalone launcher with project selector
├── bom_reliability.py       # BOM generator integration
├── run_standalone.py        # Development test runner
└── README.md                # This file
```

## Modifying Calculations

All reliability formulas are in `reliability_math.py`. Key sections:

### Adding New Component Types
```python
# In IC_DIE_TABLE, add new IC technologies:
IC_DIE_TABLE["MY_NEW_IC"] = {
    "l1": 1.0e-5,  # Per-transistor rate
    "l2": 15,      # Fixed rate
    "ea": ActivationEnergy.MOS
}

# In IC_TYPE_CHOICES, add user-friendly name:
IC_TYPE_CHOICES["My New IC Type"] = "MY_NEW_IC"
```

### Adjusting Failure Rates
```python
# Modify base rates in lookup tables:
DIODE_BASE_RATES["Signal (<1A)"]["l0"] = 0.05  # Lower base rate

CAPACITOR_PARAMS["Ceramic Class II (X7R/X5R)"]["l0"] = 0.12
```

### Changing Activation Energies
```python
class ActivationEnergy:
    MOS = 3480        # Adjust for different process nodes
    BIPOLAR = 4640
    # Add custom values as needed
```

## IEC TR 62380 Reference

### Key Formulas

**Temperature Factor:**
```
πt = exp(Ea × (1/Tref - 1/(273+Tj)))
```

**Thermal Cycling Factor:**
```
πn = n^0.76          for n ≤ 8760 cycles/year
πn = 1.7 × n^0.6     for n > 8760 cycles/year
```

**IC Failure Rate:**
```
λ = (λdie + λpackage + λEOS) × 10^-9 /h

λdie = (λ1 × N × exp(-0.35×a) + λ2) × πt
λpackage = 2.75×10^-3 × πα × πn × ΔT^0.68 × λ3
```

**System Reliability:**
```
R(t) = exp(-λ × t)
MTTF = 1/λ
```

### Standard Activation Energies
| Technology | Ea (eV) | Ea (K) |
|------------|---------|--------|
| MOS | 0.3 | 3480 |
| Bipolar | 0.4 | 4640 |
| Ceramic Cap | 0.1 | 1160 |
| Passives | 0.15 | 1740 |
| Aluminum Cap | 0.4 | 4640 |

## Export Formats

- **HTML**: Formatted report with tables
- **Markdown**: GitHub-compatible documentation
- **CSV**: Spreadsheet import for further analysis

## Configuration Files

Save/load configurations as JSON files containing:
- Block diagram structure
- Mission profile settings
- All edited component fields

## Dependencies

- Python 3.8+
- wxPython 4.0+
- KiCad 9.0+ (for plugin mode)

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Author

**Eliot Abramo**

Created for professional space electronics reliability analysis.

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style conventions (type hints, docstrings)
2. New features include appropriate documentation
3. IEC TR 62380 calculations are properly documented with section references
4. All functions include docstrings with parameter descriptions

## References

- **IEC TR 62380:2004** - Reliability data handbook – Universal model for reliability prediction
- **ECSS-Q-ST-30-11C** - Derating – EEE components
- **MIL-HDBK-217F** - Reliability Prediction of Electronic Equipment
