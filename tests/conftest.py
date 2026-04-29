"""pytest conftest - runs before any test, sets up sys.path."""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PLUGIN_ROOT = str(_REPO_ROOT / "plugins")
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)
