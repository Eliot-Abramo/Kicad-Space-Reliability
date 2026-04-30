"""
Unified Sensitivity & Uncertainty Analysis Engine
==================================================
Monte Carlo, Sobol (parameter & component-level), and user-defined what-if.
Uses block structure (series/parallel/K-of-N). This module contains
production utilities plus exploratory global-sensitivity helpers.

Author:  Eliot Abramo
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from ..reliability_math import (
        FIT_PER_LAMBDA,
        LAMBDA_PER_FIT,
        calculate_component_lambda,
        r_k_of_n,
        r_parallel,
        r_series,
        reliability_from_lambda,
    )
except ImportError:
    from reliability_math import (
        FIT_PER_LAMBDA,
        LAMBDA_PER_FIT,
        calculate_component_lambda,
        r_k_of_n,
        r_parallel,
        r_series,
        reliability_from_lambda,
    )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class UncertainParam:
    """User-selected uncertain parameter (component/field specific)."""

    sheet_path: str
    reference: str
    field_name: str
    nominal: float
    low: float
    high: float
    distribution: str = "uniform"  # "uniform" or "pert"


@dataclass
class WhatIfShift:
    """User-defined parameter shift for what-if analysis."""

    sheet_path: str
    reference: str
    field_name: str
    new_value: float


@dataclass
class UncertaintyResult:
    """Monte Carlo uncertainty analysis results."""

    nominal_R: float  # noqa: N815
    nominal_lambda_fit: float
    mean_R: float  # noqa: N815
    median_R: float  # noqa: N815
    std_R: float  # noqa: N815
    ci_low: float
    ci_high: float
    confidence_level: float
    reliability_samples: np.ndarray
    lambda_samples: np.ndarray
    n_simulations: int
    runtime_seconds: float
    jensen_note: str


@dataclass
class SobolEntry:
    """Single Sobol index result."""

    name: str
    sobol_index: float
    sobol_std: float
    rank: int


@dataclass
class SobolResult:
    """Sobol sensitivity analysis results."""

    parameter_level: list[SobolEntry]
    component_level: list[SobolEntry]
    nominal_R: float  # noqa: N815
    n_samples: int
    runtime_seconds: float


@dataclass
class WhatIfResult:
    """What-if scenario result."""

    name: str
    baseline_R: float  # noqa: N815
    baseline_lambda_fit: float
    shifted_R: float  # noqa: N815
    shifted_lambda_fit: float
    delta_R: float  # noqa: N815
    delta_lambda_pct: float


# =============================================================================
# Block structure: compute system R from sheet lambdas
# =============================================================================


def _system_r_from_sheet_lambdas(
    blocks: dict,
    root_id: str | None,
    sheet_lambdas: dict[str, float],
    mission_hours: float,
) -> float:
    """Compute system reliability from sheet lambdas using block structure."""
    if not root_id or not isinstance(blocks, dict) or root_id not in blocks:
        return 1.0
    try:
        mh = float(mission_hours)
        if mh <= 0 or mh != mh:
            mh = 43800
    except (TypeError, ValueError):
        mh = 43800

    def calc(bid: str) -> float:
        b = blocks.get(bid)
        if not b:
            return 1.0
        is_grp = getattr(b, "is_group", False) if hasattr(b, "is_group") else False
        if is_grp:
            children = getattr(b, "children", None) or []
            child_rs = [calc(cid) for cid in children]
            conn = getattr(b, "connection_type", "series")
            k = getattr(b, "k_value", 2)
            if conn == "series":
                return r_series(child_rs)
            if conn == "parallel":
                return r_parallel(child_rs)
            return r_k_of_n(child_rs, k)
        name = getattr(b, "name", "") if hasattr(b, "name") else ""
        lam = sheet_lambdas.get(name, 0.0) if isinstance(sheet_lambdas, dict) else 0.0
        return reliability_from_lambda(lam, mh)

    return calc(root_id)


def _sheet_lambdas_from_components(
    sheet_data: dict,
    component_lambdas: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Aggregate component lambdas into sheet lambdas."""
    result = {}
    if not isinstance(sheet_data, dict):
        return result
    for path, data in sheet_data.items():
        if not isinstance(data, dict):
            result[path] = 0.0
            continue
        comps = data.get("components")
        if comps is None or not isinstance(comps, list):
            result[path] = 0.0
            continue
        total = 0.0
        for comp in comps:
            if not isinstance(comp, dict):
                continue
            ref = comp.get("ref", "?")
            lam = component_lambdas.get((path, ref))
            if lam is not None:
                total += lam
            else:
                with contextlib.suppress(TypeError, ValueError):
                    total += float(comp.get("lambda", 0) or 0)
        result[path] = total
    return result


# =============================================================================
# Monte Carlo
# =============================================================================


def _pert_sample(rng, low: float, high: float, size: int, mode: float | None = None) -> np.ndarray:
    """Sample from PERT (scaled Beta) or Uniform."""
    if mode is not None and low < mode < high:
        gamma = 4.0
        r = high - low
        alpha = 1.0 + gamma * (mode - low) / r
        beta = 1.0 + gamma * (high - mode) / r
        alpha, beta = max(alpha, 1.001), max(beta, 1.001)
        x = rng.beta(alpha, beta, size=size)
        return low + r * x
    return rng.uniform(low, high, size=size)


def run_monte_carlo(  # noqa: C901
    sheet_data: dict,
    blocks: dict,
    root_id: str | None,
    mission_hours: float,
    uncertain_params: list[UncertainParam],
    n_simulations: int = 3000,
    confidence_level: float = 0.90,
    seed: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> UncertaintyResult:
    """
    Monte Carlo uncertainty propagation through block structure.
    Samples user-selected parameters, evaluates IEC formulas, computes R via block topology.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Build param index: (sheet_path, ref, field_name) -> (nominal, low, high, dist)
    param_index: dict[tuple[str, str, str], UncertainParam] = {}
    for p in uncertain_params:
        param_index[(p.sheet_path, p.reference, p.field_name)] = p

    # Pre-generate samples for each param
    param_samples: dict[tuple[str, str, str], np.ndarray] = {}
    for p in uncertain_params:
        mode = p.nominal if p.distribution == "pert" else None
        param_samples[(p.sheet_path, p.reference, p.field_name)] = _pert_sample(rng, p.low, p.high, n_simulations, mode)

    # Nominal lambdas (no perturbation)
    nominal_comp_lambdas = {}
    if isinstance(sheet_data, dict):
        for path, data in sheet_data.items():
            if not isinstance(data, dict):
                continue
            comps = data.get("components")
            if comps is None or not isinstance(comps, list):
                continue
            for comp in comps:
                if not isinstance(comp, dict):
                    continue
                ref = comp.get("ref", "?")
                ovr = comp.get("override_lambda")
                if ovr is not None:
                    nominal_comp_lambdas[(path, ref)] = ovr * LAMBDA_PER_FIT
                else:
                    try:
                        nominal_comp_lambdas[(path, ref)] = float(comp.get("lambda", 0) or 0)
                    except (TypeError, ValueError):
                        nominal_comp_lambdas[(path, ref)] = 0.0

    nominal_sheet_lam = _sheet_lambdas_from_sheet_data(sheet_data)
    nominal_R = _system_r_from_sheet_lambdas(blocks, root_id, nominal_sheet_lam, mission_hours)  # noqa: N806
    nominal_lam_total = sum(nominal_sheet_lam.values())
    nominal_fit = nominal_lam_total * FIT_PER_LAMBDA

    # MC loop
    R_samples = np.empty(n_simulations)  # noqa: N806
    lam_samples = np.empty(n_simulations)

    for sim in range(n_simulations):
        comp_lambdas = {}
        if isinstance(sheet_data, dict):
            for path, data in sheet_data.items():
                if not isinstance(data, dict):
                    continue
                comps = data.get("components")
                if comps is None or not isinstance(comps, list):
                    continue
                for comp in comps:
                    if not isinstance(comp, dict):
                        continue
                    ref = comp.get("ref", "?")
                    ovr = comp.get("override_lambda")
                    if ovr is not None:
                        comp_lambdas[(path, ref)] = ovr * LAMBDA_PER_FIT
                        continue

                    params = dict(comp.get("params") or {})
                    ct = comp.get("class", "Resistor")
                    for sp, r, fn in param_index:
                        if sp == path and r == ref and fn in params:
                            with contextlib.suppress(TypeError, ValueError):
                                params[fn] = float(param_samples[(sp, r, fn)][sim])

                    try:
                        res = calculate_component_lambda(ct, params)
                        lam = float(res.get("lambda_total", 0) or 0)
                    except Exception:  # noqa: BLE001
                        try:
                            lam = float(comp.get("lambda", 0) or 0)
                        except (TypeError, ValueError):
                            lam = 0.0
                    comp_lambdas[(path, ref)] = max(0.0, lam)

        sheet_lam = _sheet_lambdas_from_components(sheet_data, comp_lambdas)
        R = _system_r_from_sheet_lambdas(blocks, root_id, sheet_lam, mission_hours)  # noqa: N806
        lam_total = sum(sheet_lam.values())

        R_samples[sim] = R
        lam_samples[sim] = lam_total

        if progress_callback and (sim + 1) % max(1, n_simulations // 50) == 0:
            progress_callback(sim + 1, n_simulations, "Monte Carlo...")

    alpha = (1 - confidence_level) / 2
    mean_R = float(np.mean(R_samples))  # noqa: N806
    median_R = float(np.median(R_samples))  # noqa: N806
    std_R = float(np.std(R_samples, ddof=1))  # noqa: N806
    ci_lo = float(np.percentile(R_samples, alpha * 100))
    ci_hi = float(np.percentile(R_samples, (1 - alpha) * 100))

    jensen = ""
    mean_lam = float(np.mean(lam_samples))
    r_at_mean_lambda = float(np.exp(-mean_lam * mission_hours))
    if mean_R + 1e-6 < r_at_mean_lambda:
        jensen = (
            f"Jensen diagnostic warning: E[R(t)] = {mean_R:.6f} is below "
            f"R(E[λ]) = {r_at_mean_lambda:.6f}. This suggests sampling noise, "
            "numerical error, or inconsistent baseline handling."
        )

    return UncertaintyResult(
        nominal_R=nominal_R,
        nominal_lambda_fit=nominal_fit,
        mean_R=mean_R,
        median_R=median_R,
        std_R=std_R,
        ci_low=ci_lo,
        ci_high=ci_hi,
        confidence_level=confidence_level,
        reliability_samples=R_samples,
        lambda_samples=lam_samples,
        n_simulations=n_simulations,
        runtime_seconds=time.time() - t0,
        jensen_note=jensen,
    )


def _sheet_lambdas_from_sheet_data(sheet_data: dict) -> dict[str, float]:
    result = {}
    if not isinstance(sheet_data, dict):
        return result
    for path, data in sheet_data.items():
        if not isinstance(data, dict):
            result[path] = 0.0
            continue
        try:
            result[path] = float(data.get("lambda", 0) or 0)
        except (TypeError, ValueError):
            result[path] = 0.0
    return result


# =============================================================================
# Sobol (Pick-Freeze)  # noqa: ERA001


def _sobol_pick_freeze(
    f: Callable[[np.ndarray], np.ndarray],
    X1: np.ndarray,  # noqa: N803
    X2: np.ndarray,  # noqa: N803
    names: list[str],  # noqa: ARG001
) -> tuple[np.ndarray, np.ndarray]:
    """Pick-Freeze estimator for first-order Sobol indices."""
    N, d = X1.shape  # noqa: N806
    if d == 0 or N < 10:
        return np.array([]), np.array([])

    Y1 = f(X1)  # noqa: N806
    f(X2)
    var_Y = np.var(Y1)  # noqa: N806

    if var_Y < 1e-30:
        return np.zeros(d), np.zeros(d)

    S = np.zeros(d)  # noqa: N806
    S_std = np.zeros(d)  # noqa: N806

    for i in range(d):
        X1_prime = X2.copy()  # noqa: N806
        X1_prime[:, i] = X1[:, i]
        Y1_prime = f(X1_prime)  # noqa: N806
        cov_est = np.mean(Y1 * Y1_prime) - np.mean(Y1) * np.mean(Y1_prime)
        S[i] = cov_est / var_Y
        S[i] = max(0.0, min(1.0, S[i]))
        # Bootstrap std (simplified)
        S_std[i] = np.std(Y1 * Y1_prime) / (np.sqrt(N) * var_Y) if var_Y > 0 else 0

    return S, S_std


def run_sobol(  # noqa: C901
    sheet_data: dict,
    blocks: dict,
    root_id: str | None,
    mission_hours: float,
    uncertain_params: list[UncertainParam],
    n_samples: int = 1000,
    seed: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,  # noqa: ARG001
) -> SobolResult:
    """
    Sobol first-order sensitivity via Pick-Freeze.
    Parameter-level: inputs = uncertain params, output = R(t).
    Component-level: inputs = component lambdas (sampled), output = R(t).
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    nominal_sheet_lam = _sheet_lambdas_from_sheet_data(sheet_data)
    nominal_R = _system_r_from_sheet_lambdas(blocks, root_id, nominal_sheet_lam, mission_hours)  # noqa: N806

    # Parameter-level Sobol
    param_names = [f"{p.reference}.{p.field_name}" for p in uncertain_params]
    n_params = len(uncertain_params)
    param_entries = []

    if n_params > 0:
        X1 = np.zeros((n_samples, n_params))  # noqa: N806
        X2 = np.zeros((n_samples, n_params))  # noqa: N806
        for j, p in enumerate(uncertain_params):
            mode = p.nominal if p.distribution == "pert" else None
            X1[:, j] = _pert_sample(rng, p.low, p.high, n_samples, mode)
            X2[:, j] = _pert_sample(rng, p.low, p.high, n_samples, mode)

        def f_param(X: np.ndarray) -> np.ndarray:  # noqa: N803
            R_out = np.zeros(X.shape[0])  # noqa: N806
            for i in range(X.shape[0]):
                comp_lambdas = _perturbed_comp_lambdas(sheet_data, uncertain_params, X[i, :])
                sheet_lam = _sheet_lambdas_from_components(sheet_data, comp_lambdas)
                R_out[i] = _system_r_from_sheet_lambdas(blocks, root_id, sheet_lam, mission_hours)
            return R_out

        S, S_std = _sobol_pick_freeze(f_param, X1, X2, param_names)  # noqa: N806
        for j, name in enumerate(param_names):
            param_entries.append(
                SobolEntry(
                    name=name,
                    sobol_index=float(S[j]),
                    sobol_std=float(S_std[j]),
                    rank=0,
                )
            )
        param_entries.sort(key=lambda e: -e.sobol_index)
        for rank, e in enumerate(param_entries, 1):
            e.rank = rank

    # Component-level Sobol: sample component lambdas via 2 MC runs, Pick-Freeze on R
    comp_entries = []
    comp_list = []
    if isinstance(sheet_data, dict):
        for path, data in sheet_data.items():
            if not isinstance(data, dict):
                continue
            comps = data.get("components")
            if comps is None or not isinstance(comps, list):
                continue
            comp_list.extend(
                (path, comp.get("ref", "?"))
                for comp in comps
                if isinstance(comp, dict) and comp.get("override_lambda") is None
            )
    n_comp = len(comp_list)

    if n_comp > 0 and len(uncertain_params) > 0:
        X1_lam = np.zeros((n_samples, n_comp))  # noqa: N806
        X2_lam = np.zeros((n_samples, n_comp))  # noqa: N806

        def sample_component_lambdas(rng, samples):
            param_samples = {}
            for p in uncertain_params:
                mode = p.nominal if p.distribution == "pert" else None
                param_samples[(p.sheet_path, p.reference, p.field_name)] = _pert_sample(
                    rng, p.low, p.high, samples, mode
                )

            out = np.zeros((samples, n_comp))
            for sim in range(samples):
                comp_lam = {}
                if isinstance(sheet_data, dict):
                    for path, data in sheet_data.items():
                        if not isinstance(data, dict):
                            continue
                        comps = data.get("components")
                        if comps is None or not isinstance(comps, list):
                            continue
                        for comp in comps:
                            if not isinstance(comp, dict):
                                continue
                            ref = comp.get("ref", "?")
                            if comp.get("override_lambda") is not None:
                                comp_lam[(path, ref)] = comp["override_lambda"] * LAMBDA_PER_FIT
                                continue
                            params = dict(comp.get("params") or {})
                        for p in uncertain_params:
                            if p.sheet_path == path and p.reference == ref and p.field_name in params:
                                params[p.field_name] = param_samples[(p.sheet_path, p.reference, p.field_name)][sim]
                        try:
                            res = calculate_component_lambda(comp.get("class", "Resistor"), params)
                            lam = float(res.get("lambda_total", 0) or 0)
                        except Exception:  # noqa: BLE001
                            lam = float(comp.get("lambda", 0) or 0)
                        comp_lam[(path, ref)] = max(0.0, lam)
                for j, (path, ref) in enumerate(comp_list):
                    out[sim, j] = comp_lam.get((path, ref), 0.0)
            return out

        X1_lam = sample_component_lambdas(rng, n_samples)  # noqa: N806
        X2_lam = sample_component_lambdas(rng, n_samples)  # noqa: N806

        def f_comp(X: np.ndarray) -> np.ndarray:  # noqa: N803
            R_out = np.zeros(X.shape[0])  # noqa: N806
            for i in range(X.shape[0]):
                comp_lam = {(comp_list[j][0], comp_list[j][1]): X[i, j] for j in range(n_comp)}
                sheet_lam = _sheet_lambdas_from_components(sheet_data, comp_lam)
                R_out[i] = _system_r_from_sheet_lambdas(blocks, root_id, sheet_lam, mission_hours)
            return R_out

        S_c, S_std_c = _sobol_pick_freeze(f_comp, X1_lam, X2_lam, [f"{r}" for _, r in comp_list])  # noqa: N806
        for j, (_path, ref) in enumerate(comp_list):
            comp_entries.append(
                SobolEntry(
                    name=f"{ref}",
                    sobol_index=float(S_c[j]),
                    sobol_std=float(S_std_c[j]),
                    rank=0,
                )
            )
        comp_entries.sort(key=lambda e: -e.sobol_index)
        for rank, e in enumerate(comp_entries, 1):
            e.rank = rank

    return SobolResult(
        parameter_level=param_entries,
        component_level=comp_entries,
        nominal_R=nominal_R,
        n_samples=n_samples,
        runtime_seconds=time.time() - t0,
    )


def _perturbed_comp_lambdas(
    sheet_data: dict,
    uncertain_params: list[UncertainParam],
    x: np.ndarray,
) -> dict[tuple[str, str], float]:
    comp_lambdas = {}
    param_by_idx = {(p.sheet_path, p.reference, p.field_name): (i, p) for i, p in enumerate(uncertain_params)}
    if not isinstance(sheet_data, dict):
        return comp_lambdas
    for path, data in sheet_data.items():
        if not isinstance(data, dict):
            continue
        comps = data.get("components")
        if comps is None or not isinstance(comps, list):
            continue
        for comp in comps:
            if not isinstance(comp, dict):
                continue
            ref = comp.get("ref", "?")
            ovr = comp.get("override_lambda")
            if ovr is not None:
                comp_lambdas[(path, ref)] = ovr * LAMBDA_PER_FIT
                continue
            params = dict(comp.get("params") or {})
            for (sp, r, fn), (idx, _p) in param_by_idx.items():
                if sp == path and r == ref and fn in params:
                    params[fn] = float(x[idx])
            try:
                res = calculate_component_lambda(comp.get("class", "Resistor"), params)
                lam = float(res.get("lambda_total", 0) or 0)
            except Exception:  # noqa: BLE001
                lam = float(comp.get("lambda", 0) or 0)
            comp_lambdas[(path, ref)] = max(0.0, lam)
    return comp_lambdas


# =============================================================================
# What-If (user-defined parameter shifts)
# =============================================================================


def run_whatif(
    sheet_data: dict,
    blocks: dict,
    root_id: str | None,
    mission_hours: float,
    shifts: list[WhatIfShift],
    scenario_name: str = "Custom",
) -> WhatIfResult:
    """Apply user-defined parameter shifts and compute resulting R."""
    sheet_data = sheet_data if isinstance(sheet_data, dict) else {}
    baseline_sheet_lam = _sheet_lambdas_from_sheet_data(sheet_data)
    baseline_R = _system_r_from_sheet_lambdas(blocks, root_id, baseline_sheet_lam, mission_hours)  # noqa: N806
    baseline_lam = sum(baseline_sheet_lam.values())
    baseline_fit = baseline_lam * FIT_PER_LAMBDA

    shift_map = {(s.sheet_path, s.reference, s.field_name): s.new_value for s in shifts}

    comp_lambdas = {}
    if isinstance(sheet_data, dict):
        for path, data in sheet_data.items():
            if not isinstance(data, dict):
                continue
            comps = data.get("components")
            if comps is None or not isinstance(comps, list):
                continue
            for comp in comps:
                if not isinstance(comp, dict):
                    continue
                ref = comp.get("ref", "?")
                ovr = comp.get("override_lambda")
                if ovr is not None:
                    comp_lambdas[(path, ref)] = ovr * LAMBDA_PER_FIT
                    continue
                params = dict(comp.get("params") or {})
                for (sp, r, fn), val in shift_map.items():
                    if sp == path and r == ref and fn in params:
                        params[fn] = val
                try:
                    res = calculate_component_lambda(comp.get("class", "Resistor"), params)
                    lam = float(res.get("lambda_total", 0) or 0)
                except Exception:  # noqa: BLE001
                    try:
                        lam = float(comp.get("lambda", 0) or 0)
                    except (TypeError, ValueError):
                        lam = 0.0
                comp_lambdas[(path, ref)] = max(0.0, lam)

    sheet_lam = _sheet_lambdas_from_components(sheet_data, comp_lambdas)
    shifted_R = _system_r_from_sheet_lambdas(blocks, root_id, sheet_lam, mission_hours)  # noqa: N806
    shifted_lam = sum(sheet_lam.values())
    shifted_fit = shifted_lam * FIT_PER_LAMBDA

    delta_R = shifted_R - baseline_R  # noqa: N806
    delta_lam_pct = ((shifted_fit - baseline_fit) / baseline_fit * 100) if baseline_fit > 0 else 0

    return WhatIfResult(
        name=scenario_name,
        baseline_R=baseline_R,
        baseline_lambda_fit=baseline_fit,
        shifted_R=shifted_R,
        shifted_lambda_fit=shifted_fit,
        delta_R=delta_R,
        delta_lambda_pct=delta_lam_pct,
    )
