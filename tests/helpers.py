"""Shared test helpers — importable directly (unlike conftest.py)."""

import math
import pathlib
import sys

import numpy as np

_TEST_DIR = pathlib.Path(__file__).resolve().parent
# Handle mutmut's mutants/tests/ directory: walk up past 'mutants' if present
if _TEST_DIR.parent.name == "mutants":
    REPO_ROOT = _TEST_DIR.parents[1]
else:
    REPO_ROOT = _TEST_DIR.parents[0]
PLUGIN_ROOT = REPO_ROOT / "plugins"

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def exp_reliability(lam, hours):
    return math.exp(-lam * hours)


def make_component(ref, ctype, lam, params=None):
    return {
        "ref": ref,
        "class": ctype,
        "lambda": lam,
        "params": params or {},
    }


def make_sheet_data(sheets):
    return {
        path: {
            "lambda": sum(c["lambda"] for c in data.get("components", [])),
            "components": data.get("components", []),
        }
        for path, data in sheets.items()
    }


def make_component_input(ref, ctype, params, lam=0.0):
    import monte_carlo

    return monte_carlo.ComponentInput(
        reference=ref,
        component_type=ctype,
        base_params=params,
        nominal_lambda=lam,
        override_lambda=None,
        uncertain_field_names=[],
    )


def make_parameter_spec(name, refs_values, shared=False, distribution="uniform", **bounds):
    import monte_carlo

    kwargs = {
        "name": name,
        "nominal_by_ref": refs_values,
        "distribution": distribution,
        "shared": shared,
    }
    if "delta_low" in bounds and "delta_high" in bounds:
        kwargs["delta_low"] = bounds["delta_low"]
        kwargs["delta_high"] = bounds["delta_high"]
    else:
        kwargs["rel_low"] = bounds.get("rel_low", 0.0)
        kwargs["rel_high"] = bounds.get("rel_high", 0.0)
    return monte_carlo.ParameterSpec(**kwargs)


def make_resistor_params(t_ambient=25.0, v_applied=5.0, v_rated=10.0):
    return {"t_ambient": t_ambient, "v_applied": v_applied, "v_rated": v_rated}


def make_ic_params(t_junction=50.0, package="QFP-48 (7x7mm)", ic_type="MOS Digital (Micro/DSP)", n_pins=48):
    return {
        "t_junction": t_junction,
        "package": package,
        "ic_type": ic_type,
        "n_pins": n_pins,
        "t_ambient": 25.0,
        "n_cycles": 365,
        "delta_t": 10.0,
        "a": 0.5,
    }


def make_resistor_component(ref="R1", lam=100e-9, t_ambient=25.0, v_applied=5.0, v_rated=10.0):
    return make_component(ref, "Resistor", lam, make_resistor_params(t_ambient, v_applied, v_rated))


def sample_shared_series():
    return np.array([0.0, 1.0, -2.0, 0.5, -1.5, 2.0, -0.5, 1.5, -1.0, 0.25])


def sample_drive_series():
    return np.array([1.1, 0.9, 1.2, 1.0, 0.95, 1.05, 1.15, 0.85, 1.0, 1.1])


def fake_lambda_calc(return_total_coeff=1e-6):
    def _calc(_ctype, params):
        total = sum(float(v) for v in params.values()) * return_total_coeff
        return {"lambda_total": total}

    return _calc


def fake_calc_lambda_float(return_total=1e-6):
    def _calc(_ctype, params):  # noqa: ARG001
        return return_total

    return _calc


def find_available(module, names):
    return [n for n in names if hasattr(module, n)]
