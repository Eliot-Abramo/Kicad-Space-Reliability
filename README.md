# ⚡ KiCad Reliability Calculator

Calculate component and system reliability for your KiCad projects.

![Screenshot](screenshot.png)

## ✨ Features

- **Visual Block Diagram Editor** - Drag & drop to define series/parallel/k-of-n redundancy
- **Automatic Schematic Parsing** - Reads your KiCad 9 hierarchy automatically
- **Component-Level Analysis** - FIDES-based failure rates for all component types
- **Real-Time Calculations** - See reliability update as you modify the topology
- **Export Reports** - HTML, Markdown, or CSV formats

---

## 🚀 Installation (KiCad 9)

### Step 1: Copy Files

Copy the **entire `kicad_reliability_plugin` folder** to your KiCad plugins directory:

| OS | Location |
|----|----------|
| **Linux** | `~/.local/share/kicad/9.0/scripting/plugins/` |
| **Windows** | `%APPDATA%\kicad\9.0\scripting\plugins\` |
| **macOS** | `~/Library/Preferences/kicad/9.0/scripting/plugins/` |

### Step 2: Register as BOM Generator

1. Open **Eeschema** (schematic editor)
2. Go to **Tools → Generate Bill of Materials**
3. Click **+ (Add Plugin)**
4. Browse to: `kicad_reliability_plugin/bom_reliability.py`
5. Set the **Command line** to:
   ```
   python3 "%P/bom_reliability.py" "%I" "%O"
   ```
   Or on Windows:
   ```
   python "%P/bom_reliability.py" "%I" "%O"
   ```
6. Set **Nickname** to: `Reliability Calculator`
7. Click **OK**

### Step 3: Use It!

1. Open your schematic in Eeschema
2. Go to **Tools → Generate Bill of Materials**
3. Select **"Reliability Calculator"**
4. Click **Generate**
5. The Reliability Calculator window opens with your project loaded! 🎉

---

## 🎯 Quick Start

### 1. Add Sheets to the Diagram

- Select sheets from the left panel
- Click **Add Selected** or double-click
- Sheets appear as blocks in the diagram

### 2. Define System Topology

- **Drag a rectangle** around multiple blocks to select them
- **Right-click** → Choose connection type:
  - **SERIES** - All blocks must work (R = R₁ × R₂ × ...)
  - **PARALLEL** - At least one must work (R = 1 - (1-R₁)(1-R₂)...)
  - **K-of-N** - K out of N must work (redundancy)

### 3. View Results

- Click **Calculate System Reliability**
- See per-block and system reliability
- Export reports as needed

---

## 📋 Component Fields

For accurate calculations, add these custom fields to your symbols:

### Required Field

| Field | Description | Examples |
|-------|-------------|----------|
| `Reliability_Class` | Component type | `Resistor`, `Ceramic Capacitor`, `Integrated Circuit` |

### Supported Classes

- `Resistor`
- `Ceramic Capacitor` / `Tantalum Capacitor`
- `Low Power Transistor` / `Power Transistor`
- `Low Power Diode` / `Power Diode`
- `Integrated Circuit` / `MCU`
- `Inductor` / `Transformer`
- `DC-DC Converter` / `LDO Regulator`
- `Crystal` / `Connector` / `Primary Battery`

### Optional Fields (for better accuracy)

| Field | Default | Description |
|-------|---------|-------------|
| `T_Junction` | 85°C | Junction temperature |
| `T_Ambient` | 25°C | Ambient temperature |
| `Operating_Power` | 0.01W | Operating power (resistors) |
| `Rated_Power` | 0.125W | Rated power (resistors) |
| `Package` | auto | Package type (from footprint) |
| `IC_Type` | auto | IC classification |
| `Construction_Year` | 2020 | Year of manufacture |

---

## 🖥️ Standalone Mode

You can also run the calculator without KiCad:

```bash
cd ~/.local/share/kicad/9.0/scripting/plugins/kicad_reliability_plugin
python3 reliability_launcher.py
```

Or with a specific project:
```bash
python3 reliability_launcher.py /path/to/your/kicad/project
```

---

## 📊 Calculation Methodology

Based on FIDES 2009 reliability prediction methodology:

### Component Failure Rate (λ)

Each component type has a model considering:
- Base failure rate (component-specific)
- Temperature acceleration (Arrhenius model)
- Thermal cycling stress
- Electrical stress factors
- Package type

### System Reliability

- **Series**: R_sys = R₁ × R₂ × R₃ × ...
- **Parallel**: R_sys = 1 - (1-R₁)(1-R₂)(1-R₃)...
- **K-of-N**: Binomial probability

### Output

- **R** - Reliability (0 to 1)
- **λ** - Failure rate (failures/hour)
- **MTTF** - Mean Time To Failure = 1/λ

---

## 📁 Files

```
kicad_reliability_plugin/
├── bom_reliability.py      # KiCad BOM generator entry point
├── reliability_launcher.py # Main launcher with project selector
├── reliability_dialog.py   # Main UI
├── reliability_core.py     # Calculation engine
├── schematic_parser.py     # KiCad file parser
├── block_editor.py         # Visual diagram editor
└── README.md               # This file
```

---

## ❓ Troubleshooting

### "Plugin not showing in BOM list"
- Make sure you added it via Tools → Generate BOM → Add Plugin
- Check the command line path is correct

### "No sheets found"
- Ensure your .kicad_sch files are in the project directory
- Check for a .kicad_pro file in the same directory

### "Wrong reliability values"
- Add `Reliability_Class` field to your symbols
- Check temperature and power fields for accuracy

---

## 📄 License

MIT License - Use freely in your projects!
