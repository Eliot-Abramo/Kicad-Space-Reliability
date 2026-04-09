"""
Helpers for loading sibling plugin modules in package or flat-module mode.
Author:  Eliot Abramo
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_plugin_paths() -> None:
    base = Path(__file__).resolve().parent
    ui_dir = base / "ui"
    for path in (base, ui_dir):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
