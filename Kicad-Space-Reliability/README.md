# KiCad Reliability Calculator Plugin v2.0.0

IEC TR 62380 reliability analysis tool for KiCad with visual block diagram editor.

## Features

### Core Analysis
- **IEC TR 62380 Compliance**: Full implementation of component failure rate models
- **Component Types**: ICs, transistors, diodes, capacitors, resistors, inductors, connectors
- **System Modeling**: Series, parallel, and K-of-N redundancy configurations
- **Visual Editor**: Drag-and-drop block diagram for system topology

### New in v2.0.0
- **Configurable EOS**: 10 interface types (Computer, Telecom, Avionics, Space, etc.)
- **Working Time Ratio (τ_on)**: Support for duty-cycled operation (0-1)
- **Thermal Expansion**: Selectable substrate and package materials
- **Corrected π_n Threshold**: Proper 8760 cycles/year discontinuity handling

### Advanced Analysis
- **Monte Carlo**: Uncertainty quantification with convergence detection
- **Sobol Sensitivity**: First-order and total-order sensitivity indices
- **Export Reports**: HTML, Markdown, CSV, JSON formats

## Installation

### Method 1: KiCad Plugin Manager
1. Download the `.zip` file
2. In KiCad, go to **Plugin and Content Manager**
3. Click **Install from File** and select the zip

### Method 2: Manual Installation
Copy the `kicad_reliability_plugin` folder to:
- **Linux**: `~/.local/share/kicad/9.0/scripting/plugins/`
- **Windows**: `%APPDATA%\kicad\9.0\scripting\plugins\`
- **macOS**: `~/Library/Preferences/kicad/9.0/scripting/plugins/`

## Usage

1. Open your KiCad project
2. In PCB Editor, go to **Tools → External Plugins → Reliability Calculator**
3. Add schematic sheets to the block diagram
4. Group blocks as Series/Parallel/K-of-N
5. Edit component parameters as needed
6. View system reliability results
7. Export reports

## Quick Start

```python
# Standalone usage
from kicad_reliability_plugin import ReliabilityMainDialog
import wx

app = wx.App()
dlg = ReliabilityMainDialog(None, "/path/to/project")
dlg.ShowModal()
dlg.Destroy()
```

## IEC TR 62380 Parameters

### Interface Types (EOS)
| Type | λ_EOS (FIT) |
|------|-------------|
| Computer | 10 |
| Telecom (Switching) | 15 |
| Telecom (Subscriber) | 70 |
| Avionics | 20 |
| Power Supply | 40 |
| Space (LEO) | 25 |
| Space (GEO) | 35 |
| Industrial | 30 |
| Automotive | 45 |
| Consumer | 50 |

### Thermal Expansion (α, ppm/°C)
| Substrate | α |
|-----------|---|
| FR4 (Epoxy Glass) | 16.0 |
| Polyimide Flex | 6.5 |
| Alumina (Ceramic) | 6.5 |
| Aluminum (Metal Core) | 23.0 |
| Rogers (PTFE) | 24.0 |

## File Structure

```
kicad_reliability_plugin/
├── __init__.py           # Plugin entry point
├── plugin.py             # KiCad plugin class
├── reliability_math.py   # IEC TR 62380 calculations
├── reliability_dialog.py # Main UI
├── block_editor.py       # Visual block diagram
├── component_editor.py   # Component parameter editing
├── schematic_parser.py   # KiCad schematic parsing
├── monte_carlo.py        # Uncertainty analysis
├── sensitivity_analysis.py # Sobol indices
├── report_generator.py   # Report generation
└── README.md
```

## License

MIT License

## Version History

- **v2.0.0**: New EOS configuration, τ_on support, thermal expansion, corrected π_n
- **v1.0.0**: Initial release with basic IEC TR 62380 support
