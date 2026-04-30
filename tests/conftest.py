"""pytest conftest - runs before any test, sets up sys.path."""

import pathlib
import sys

_DIR = pathlib.Path(__file__).resolve().parent
if _DIR.parent.name == "mutants":
    _REPO_ROOT = _DIR.parents[1]
else:
    _REPO_ROOT = _DIR.parents[0]
_PLUGIN_ROOT = str(_REPO_ROOT / "plugins")
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)
