#!/usr/bin/env python3
"""
KiCad BOM Generator Plugin - Reliability Calculator

This script is called by KiCad's Schematic Editor via Tools > Generate Bill of Materials.
It launches the Reliability Calculator GUI with the current project loaded.

Installation:
1. In Eeschema, go to Tools > Generate Bill of Materials
2. Click "Add Plugin..."  
3. Browse to this file (bom_reliability.py)
4. Set the command line to:
   python3 "PLUGIN_PATH" "%I" "%O"
   
   Or on Windows:
   python "PLUGIN_PATH" "%I" "%O"

5. Give it a nickname like "Reliability Calculator"
6. Click OK

Usage:
- With your schematic open, go to Tools > Generate Bill of Materials
- Select "Reliability Calculator" from the list
- Click "Generate"
- The Reliability Calculator window will open with your project loaded

Author:  Eliot Abramo
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    # Get the plugin directory (where this script lives)
    plugin_dir = Path(__file__).parent.absolute()
    
    # Parse command line arguments from KiCad
    # KiCad passes: script.py <input_xml_netlist> <output_file>
    input_xml = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Find the project directory from the input XML path
    project_dir = None
    if input_xml:
        input_path = Path(input_xml)
        # The XML is usually in the project directory
        project_dir = input_path.parent
        
        # Look for .kicad_pro file to confirm
        pro_files = list(project_dir.glob("*.kicad_pro"))
        if not pro_files:
            # Try parent directory
            project_dir = project_dir.parent
            pro_files = list(project_dir.glob("*.kicad_pro"))
    
    # Launch the main GUI
    launcher = plugin_dir / "reliability_launcher.py"
    
    cmd = [sys.executable, str(launcher)]
    if project_dir:
        cmd.append(str(project_dir))
    
    # Run the GUI (non-blocking so KiCad doesn't freeze)
    if sys.platform == 'win32':
        # On Windows, use START to detach
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # On Linux/Mac, fork to background
        subprocess.Popen(cmd, start_new_session=True)
    
    # Write a simple output file so KiCad doesn't complain
    if output_file:
        with open(output_file, 'w') as f:
            f.write("Reliability Calculator launched.\n")
            f.write(f"Project: {project_dir}\n")
    
    print("Reliability Calculator launched!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
