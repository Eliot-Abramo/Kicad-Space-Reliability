"""pytest conftest - sets up sys.path + hypothesis profile for mutmut."""

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

# Suppress hypothesis health check for mutmut (files are copied, triggering
# differing_executors). This is only relevant when running from mutants/tests/.
from hypothesis import HealthCheck, settings  # noqa: E402
settings.register_profile("mutmut", suppress_health_check=[HealthCheck.differing_executors], phases=["generate"])
settings.load_profile("mutmut")
