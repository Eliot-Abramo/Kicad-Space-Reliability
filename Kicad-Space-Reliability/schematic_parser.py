"""
KiCad Schematic Parser
======================
Reads KiCad 9 schematic files and extracts hierarchy and component data.

Author:  Eliot Abramo
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

@dataclass
class Component:
    """A component in the schematic."""
    reference: str
    value: str
    lib_id: str
    sheet_path: str
    footprint: str = ""
    fields: Dict[str, str] = field(default_factory=dict)
    
    def get_field(self, name: str, default: Any = None) -> Any:
        name_lower = name.lower().replace(" ", "_").replace("-", "_")
        for key, val in self.fields.items():
            if key.lower().replace(" ", "_").replace("-", "_") == name_lower:
                return val
        return default
    
    def get_float(self, name: str, default: float = 0.0) -> float:
        val = self.get_field(name)
        if val is None: return default
        try:
            s = str(val).strip().upper()
            mult = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12}
            for suffix, m in mult.items():
                if s.endswith(suffix): return float(s[:-1]) * m
            return float(s)
        except ValueError: return default
    
    def get_int(self, name: str, default: int = 0) -> int:
        return int(self.get_float(name, default))

@dataclass
class Sheet:
    """A hierarchical sheet."""
    name: str
    path: str
    filename: str
    components: List[Component] = field(default_factory=list)
    child_sheets: List[str] = field(default_factory=list)

class SchematicParser:
    """Parser for KiCad 9 schematic files."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        if self.project_path.is_file():
            self.project_dir = self.project_path.parent
            self.project_name = self.project_path.stem
        else:
            self.project_dir = self.project_path
            pro_files = list(self.project_dir.glob("*.kicad_pro"))
            self.project_name = pro_files[0].stem if pro_files else self.project_dir.name
        self.sheets: Dict[str, Sheet] = {}
        self.all_components: List[Component] = []
    
    def parse(self) -> bool:
        root_sch = self.project_dir / f"{self.project_name}.kicad_sch"
        if not root_sch.exists():
            sch_files = list(self.project_dir.glob("*.kicad_sch"))
            if sch_files: root_sch = sch_files[0]
            else: return False
        self._parse_sheet(root_sch, "/")
        return len(self.sheets) > 0
    
    def _parse_sheet(self, sch_path: Path, hierarchy_path: str):
        if not sch_path.exists(): return
        try: content = sch_path.read_text(encoding='utf-8')
        except: return
        
        sheet_name = sch_path.stem
        display_path = hierarchy_path if hierarchy_path != "/" else f"/{sheet_name}/"
        sheet = Sheet(name=sheet_name, path=display_path, filename=str(sch_path))
        sheet.components = self._parse_components(content, display_path)
        self.all_components.extend(sheet.components)
        
        for child_name, child_file in self._parse_child_sheets(content):
            child_path = display_path.rstrip('/') + "/" + child_name + "/"
            sheet.child_sheets.append(child_path)
            child_sch = sch_path.parent / child_file
            self._parse_sheet(child_sch, child_path)
        
        self.sheets[display_path] = sheet
    
    def _parse_components(self, content: str, sheet_path: str) -> List[Component]:
        components = []
        # KiCad 6+: (symbol "Lib:Name" ...)  |  KiCad 5: (symbol (lib_id "Lib:Name") ...)
        symbol_pattern = r'\(symbol\s+(?:"([^"]+)"|\(lib_id\s+"([^"]+)"\))'
        pos = 0
        while True:
            match = re.search(symbol_pattern, content[pos:])
            if not match: break
            start = pos + match.start()
            lib_id = match.group(1) or match.group(2) or ""
            symbol_content = self._extract_sexp(content, start)
            if not symbol_content:
                pos = start + 1
                continue
            props = self._extract_properties(symbol_content)
            # KiCad 6+ may put Reference in (instances (path ... (reference "R1")))
            reference = props.get("Reference") or self._extract_reference_from_instances(symbol_content)
            reference = reference or "?"
            value = props.get("Value", "")
            footprint = props.get("Footprint", "")
            if reference.startswith("#") or lib_id.lower().startswith("power:"):
                pos = start + len(symbol_content)
                continue
            comp = Component(
                reference=reference, value=value, lib_id=lib_id, sheet_path=sheet_path, footprint=footprint,
                fields={k: v for k, v in props.items() if k not in ("Reference", "Value", "Footprint", "Datasheet")}
            )
            components.append(comp)
            pos = start + len(symbol_content)
        return components

    def _extract_reference_from_instances(self, symbol_sexp: str) -> Optional[str]:
        """Extract reference designator from (instances ... (reference "R1"))."""
        match = re.search(r'\(reference\s+"([^"]*)"\)', symbol_sexp)
        return match.group(1) if match else None
    
    def _parse_child_sheets(self, content: str) -> List[Tuple[str, str]]:
        children = []
        sheet_pattern = r'\(sheet\s+'
        pos = 0
        while True:
            match = re.search(sheet_pattern, content[pos:])
            if not match: break
            start = pos + match.start()
            sheet_content = self._extract_sexp(content, start)
            if not sheet_content:
                pos = start + 1
                continue
            props = self._extract_properties(sheet_content)
            sheet_name = props.get("Sheetname") or props.get("Sheet name") or props.get("Name") or ""
            sheet_file = props.get("Sheetfile") or props.get("Sheet file") or props.get("File") or ""
            if sheet_name and sheet_file:
                children.append((sheet_name, sheet_file))
            pos = start + len(sheet_content)
        return children
    
    def _extract_sexp(self, content: str, start: int) -> Optional[str]:
        if content[start] != '(': return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(content)):
            c = content[i]
            if escape: escape = False; continue
            if c == '\\': escape = True; continue
            if c == '"': in_string = not in_string
            elif not in_string:
                if c == '(': depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0: return content[start:i+1]
        return None
    
    def _extract_properties(self, sexp: str) -> Dict[str, str]:
        props = {}
        for match in re.finditer(r'\(property\s+"([^"]+)"\s+"([^"]*)"', sexp):
            props[match.group(1)] = match.group(2)
        return props
    
    def get_sheet_paths(self) -> List[str]:
        return list(self.sheets.keys())
    
    def get_sheet(self, path: str) -> Optional[Sheet]:
        return self.sheets.get(path)
    
    def get_sheet_components(self, path: str) -> List[Component]:
        sheet = self.sheets.get(path)
        return sheet.components if sheet else []


def create_test_data(sheet_names: List[str]) -> SchematicParser:
    """Create mock parser for testing."""
    parser = SchematicParser("/test")
    for path in sheet_names:
        name = path.rstrip('/').split('/')[-1] or "Root"
        sheet = Sheet(name=name, path=path, filename=f"/test/{name}.kicad_sch")
        if "Power" in path or "LDO" in path:
            sheet.components = [
                Component("R1", "10k", "Device:R", path, "0603", {"Reliability_Class": "Resistor"}),
                Component("C1", "100n", "Device:C", path, "0603", {"Reliability_Class": "Ceramic Capacitor"}),
                Component("U1", "LM7805", "Regulator:LM7805", path, "TO-220", {"Reliability_Class": "LDO Regulator"}),
            ]
        elif "MCU" in path:
            sheet.components = [
                Component("U1", "STM32F4", "MCU:STM32F4", path, "TQFP-100", {"Reliability_Class": "Integrated Circuit"}),
                Component("Y1", "8MHz", "Device:Crystal", path, "", {"Reliability_Class": "Crystal"}),
            ]
        else:
            sheet.components = [
                Component("R1", "1k", "Device:R", path, "0603", {"Reliability_Class": "Resistor"}),
                Component("C1", "100n", "Device:C", path, "0603", {"Reliability_Class": "Ceramic Capacitor"}),
            ]
        parser.sheets[path] = sheet
        parser.all_components.extend(sheet.components)
    return parser
