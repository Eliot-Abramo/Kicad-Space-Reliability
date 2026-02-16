"""
Uncertainty Analysis Module (Monte Carlo)
==========================================
Parameter-level uncertainty propagation through IEC TR 62380 formulas
using PERT or Uniform distributions on user-specified three-point estimates.

Mathematical Foundation
-----------------------
1. Input Distributions:
   - PERT (Modified Beta):  Given (min, mode, max), gamma=4:
       alpha = 1 + gamma * (mode - min) / (max - min)
       beta  = 1 + gamma * (max - mode) / (max - min)
       X     = min + (max - min) * Beta(alpha, beta)
     Reference: Malcolm et al. (1959), NASA-STD-7009A
   - Uniform:  X ~ U(min, max)

2. Propagation:
   Each MC sample evaluates the EXACT IEC TR 62380 formula for every
   component with uncertain parameters.  No output-perturbation,
   no lognormal approximation, no ad-hoc CV.
       lambda_sys = SUM_i  lambda_i(theta_sampled)
       R(t) = exp(-lambda_sys * t)

3. Shared Parameters:
   Environmental parameters (T_ambient, delta_t, n_cycles) that affect
   the entire board receive ONE random draw per sample and that value
   is applied to every component.  This replaces artificial Cholesky
   correlation on outputs with physically correct input-level coupling.

4. Importance (SRRC):
   Standardised Rank Regression Coefficients.  After the MC run,
   rank-transform all inputs and the output, fit OLS on ranks,
   and the standardised coefficients approximate first-order
   sensitivity indices for monotonic models.
   Reference: NUREG/CR-6241, Saltelli et al. (2008) ch. 5

Key Properties
--------------
- E[R(t)] <= R(t; E[lambda])  by Jensen's inequality (exp is convex)
- Convergence rate: O(1/sqrt(N))  by the Central Limit Theorem
- SRRC captures monotonic sensitivity: valid for all IEC acceleration
  factors (Arrhenius, Coffin-Manson, power-law).

Limitations
-----------
- SRRC may understate importance of parameters with non-monotonic
  influence.  For the IEC model this case does not arise.
- PERT with gamma=4 assumes the mode is roughly 4x more likely than
  the tails.  For highly uncertain parameters, Uniform is safer.
- MC cost is O(N * K) where K = number of uncertain-parameter
  components.  For K > 200, N = 3000 may take 30-60 seconds.

Author:  Eliot Abramo
"""

import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class ParameterSpec:
    """Uncertainty specification for one parameter across components.

    If `shared` is True, a single random draw is applied to every
    component that uses this parameter (additive delta for shared,
    per-component PERT for independent).
    """
    name: str                           # IEC field name
    nominal_by_ref: Dict[str, float]    # {component_ref: nominal_value}
    delta_low: float = 0.0             # Shared mode: low delta (negative)
    delta_high: float = 0.0            # Shared mode: high delta (positive)
    rel_low: float = 0.0              # Independent mode: relative % low
    rel_high: float = 0.0             # Independent mode: relative % high
    distribution: str = "pert"          # "pert" or "uniform"
    shared: bool = True                 # One draw for all components?

    @property
    def is_uncertain(self) -> bool:
        """True if this parameter has any uncertainty range."""
        if self.shared:
            return abs(self.delta_low) > 1e-12 or abs(self.delta_high) > 1e-12
        return abs(self.rel_low) > 1e-12 or abs(self.rel_high) > 1e-12

    @property
    def n_components(self) -> int:
        return len(self.nominal_by_ref)


@dataclass
class ComponentInput:
    """Component ready for MC analysis."""
    reference: str
    component_type: str
    base_params: Dict[str, Any]        # Full parameter dict (nominal)
    nominal_lambda: float               # Pre-computed nominal lambda (/h)
    override_lambda: Optional[float]    # If set, lambda is fixed
    uncertain_field_names: List[str]    # Which fields are uncertain


@dataclass
class UncertaintyResult:
    """Complete uncertainty analysis results."""
    # Nominal
    nominal_lambda: float              # System lambda at nominal parameters
    nominal_reliability: float          # R(t) at nominal parameters
    nominal_mttf_hours: float

    # MC statistics on R(t)
    mean_reliability: float
    median_reliability: float
    std_reliability: float
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # MC statistics on lambda (FIT)
    mean_lambda_fit: float
    std_lambda_fit: float
    ci_lower_lambda_fit: float
    ci_upper_lambda_fit: float

    # Half-width CI on the mean (convergence diagnostic)
    mean_ci_halfwidth: float

    # Samples
    lambda_samples: np.ndarray          # (N,) system lambda per sample
    reliability_samples: np.ndarray     # (N,) system R(t) per sample

    # SRRC importance ranking
    parameter_importance: List[Dict]    # [{name, srrc, srrc_sq, rank}]

    # Convergence
    convergence_history: List[Tuple[int, float]]  # [(n_samples, running_mean)]

    # Meta
    n_simulations: int
    n_uncertain_params: int
    n_shared_params: int
    n_uncertain_components: int
    n_total_components: int
    runtime_seconds: float

    # Jensen's inequality note
    jensen_note: str

    def to_dict(self) -> Dict:
        """Serialise for report generation (no numpy arrays).

        Includes backward-compatible keys for existing report generator:
          mean, std, percentile_5/50/95, converged
        """
        samples_list = self.reliability_samples.tolist()
        p5 = float(np.percentile(self.reliability_samples, 5))
        p50 = float(np.percentile(self.reliability_samples, 50))
        p95 = float(np.percentile(self.reliability_samples, 95))
        return {
            # New v4 fields
            "nominal_lambda": self.nominal_lambda,
            "nominal_reliability": self.nominal_reliability,
            "nominal_mttf_hours": self.nominal_mttf_hours,
            "mean_reliability": self.mean_reliability,
            "median_reliability": self.median_reliability,
            "std_reliability": self.std_reliability,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "mean_lambda_fit": self.mean_lambda_fit,
            "std_lambda_fit": self.std_lambda_fit,
            "ci_lower_lambda_fit": self.ci_lower_lambda_fit,
            "ci_upper_lambda_fit": self.ci_upper_lambda_fit,
            "mean_ci_halfwidth": self.mean_ci_halfwidth,
            "parameter_importance": self.parameter_importance,
            "n_simulations": self.n_simulations,
            "n_uncertain_params": self.n_uncertain_params,
            "n_shared_params": self.n_shared_params,
            "n_uncertain_components": self.n_uncertain_components,
            "n_total_components": self.n_total_components,
            "runtime_seconds": self.runtime_seconds,
            "jensen_note": self.jensen_note,
            "samples": samples_list,
            "lambda_samples_fit": (self.lambda_samples * 1e9).tolist(),
            "convergence_history": self.convergence_history,
            # Backward-compatible keys for report generator
            "mean": self.mean_reliability,
            "std": self.std_reliability,
            "percentile_5": p5,
            "percentile_50": p50,
            "percentile_95": p95,
            "converged": self.mean_ci_halfwidth < 0.001,
        }


# =====================================================================
# Sampling Functions
# =====================================================================

def _pert_sample(rng, min_val, mode, max_val, size, gamma=4.0):
    """Sample from PERT (scaled Beta) distribution.

    Parameters
    ----------
    rng : numpy Generator
    min_val, mode, max_val : float
        Three-point estimate.  Must satisfy min <= mode <= max.
    size : int
        Number of samples.
    gamma : float
        Shape parameter (default 4.0).  Higher = more peaked at mode.

    Returns
    -------
    np.ndarray of shape (size,)

    Mathematical definition:
        X = min + (max - min) * Beta(alpha, beta)
        alpha = 1 + gamma * (mode - min) / (max - min)
        beta  = 1 + gamma * (max - mode) / (max - min)
    """
    range_val = max_val - min_val
    if range_val <= 1e-15:
        return np.full(size, mode)
    # Clamp mode to bounds
    m = max(min_val, min(max_val, mode))
    alpha = 1.0 + gamma * (m - min_val) / range_val
    beta_ = 1.0 + gamma * (max_val - m) / range_val
    # Clamp to avoid degenerate Beta
    alpha = max(alpha, 1.001)
    beta_ = max(beta_, 1.001)
    samples = rng.beta(alpha, beta_, size=size)
    return min_val + range_val * samples


def _uniform_sample(rng, min_val, max_val, size):
    """Sample from Uniform(min, max)."""
    if max_val <= min_val + 1e-15:
        return np.full(size, (min_val + max_val) / 2.0)
    return rng.uniform(min_val, max_val, size=size)


def _sample_parameter(rng, min_val, mode, max_val, distribution, size):
    """Draw samples from the specified distribution."""
    if distribution == "uniform":
        return _uniform_sample(rng, min_val, max_val, size)
    else:
        return _pert_sample(rng, min_val, mode, max_val, size)


# =====================================================================
# Rank Data (without scipy dependency)
# =====================================================================

def _rankdata(x):
    """Rank data using average method for ties.

    Equivalent to scipy.stats.rankdata(x, method='average').
    """
    n = len(x)
    sorter = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = np.arange(1, n + 1, dtype=np.float64)
    # Handle ties: assign average rank
    sorted_x = x[sorter]
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(sorted_x[j + 1] - sorted_x[j]) < 1e-15:
            j += 1
        if j > i:
            avg_rank = np.mean(ranks[sorter[i:j + 1]])
            ranks[sorter[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _compute_srrc(input_matrix, output_vector):
    """Compute Standardised Rank Regression Coefficients.

    Parameters
    ----------
    input_matrix : np.ndarray, shape (N, k)
        Columns are different uncertain parameter samples.
    output_vector : np.ndarray, shape (N,)
        System lambda or R(t) samples.

    Returns
    -------
    np.ndarray of shape (k,) -- the SRRC values.

    Method:
        1. Rank-transform all variables.
        2. Standardise (zero mean, unit variance).
        3. OLS regression: Y_ranked = X_ranked @ beta
        4. beta values are the SRRCs.
    """
    N, k = input_matrix.shape
    if N < k + 2 or k == 0:
        return np.zeros(k)

    # Rank-transform
    ranked_X = np.empty_like(input_matrix, dtype=np.float64)
    for j in range(k):
        ranked_X[:, j] = _rankdata(input_matrix[:, j])
    ranked_Y = _rankdata(output_vector)

    # Standardise
    X_mean = ranked_X.mean(axis=0)
    X_std = ranked_X.std(axis=0)
    X_std[X_std < 1e-15] = 1.0  # Prevent division by zero (constant col)
    Y_mean = ranked_Y.mean()
    Y_std = ranked_Y.std()
    if Y_std < 1e-15:
        return np.zeros(k)

    X_s = (ranked_X - X_mean) / X_std
    Y_s = (ranked_Y - Y_mean) / Y_std

    # OLS via normal equations: beta = (X^T X)^{-1} X^T Y
    try:
        beta, _, _, _ = np.linalg.lstsq(X_s, Y_s, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(k)

    return beta


# =====================================================================
# Core Monte Carlo Engine
# =====================================================================

def _import_reliability_math():
    try:
        from .reliability_math import (
            calculate_component_lambda, reliability_from_lambda
        )
    except ImportError:
        from reliability_math import (
            calculate_component_lambda, reliability_from_lambda
        )
    return calculate_component_lambda, reliability_from_lambda


def run_uncertainty_analysis(
    components: List[ComponentInput],
    param_specs: List[ParameterSpec],
    mission_hours: float,
    n_simulations: int = 3000,
    confidence_level: float = 0.90,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> UncertaintyResult:
    """Run parameter-level Monte Carlo uncertainty analysis.

    For each simulation sample:
      1. Draw shared parameters once (one value per shared parameter).
      2. Draw independent parameters per-component.
      3. Evaluate calculate_component_lambda() for each component
         that has at least one perturbed parameter.
      4. Sum to get lambda_sys, compute R(t) = exp(-lambda_sys * t).
      5. Record parameter samples for SRRC computation.

    Parameters
    ----------
    components : list of ComponentInput
        All system components with nominal parameters.
    param_specs : list of ParameterSpec
        Uncertainty specifications.  Only specs with is_uncertain=True
        participate in the MC.
    mission_hours : float
        Mission duration for R(t) computation.
    n_simulations : int
        Number of MC samples (default 3000).
    confidence_level : float
        For CI computation (default 0.90 => 5th-95th percentile).
    seed : int, optional
        RNG seed for reproducibility.
    progress_callback : callable, optional
        fn(current, total, message) for GUI progress.

    Returns
    -------
    UncertaintyResult with full statistics, samples, and SRRC rankings.
    """
    calc_lambda, rel_from_lambda = _import_reliability_math()
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # ---- Identify uncertain parameters ----
    uncertain_specs = [s for s in param_specs if s.is_uncertain]
    shared_specs = [s for s in uncertain_specs if s.shared]
    indep_specs = [s for s in uncertain_specs if not s.shared]
    n_uncertain = len(uncertain_specs)

    # ---- Build ref -> component index ----
    ref_to_idx = {c.reference: i for i, c in enumerate(components)}
    n_comp = len(components)

    # ---- Pre-compute nominal system lambda ----
    nominal_lambdas = np.array([c.nominal_lambda for c in components])
    nominal_sys_lambda = nominal_lambdas.sum()
    nominal_R = rel_from_lambda(nominal_sys_lambda, mission_hours)
    nominal_mttf = 1.0 / nominal_sys_lambda if nominal_sys_lambda > 0 else float('inf')

    # ---- Identify which components need re-evaluation per sample ----
    # A component needs re-evaluation if ANY of its fields is uncertain.
    uncertain_fields_by_ref = {}
    for spec in uncertain_specs:
        for ref in spec.nominal_by_ref:
            uncertain_fields_by_ref.setdefault(ref, set()).add(spec.name)

    uncertain_comp_indices = []
    for ref, fields in uncertain_fields_by_ref.items():
        if ref in ref_to_idx:
            idx = ref_to_idx[ref]
            components[idx].uncertain_field_names = list(fields)
            uncertain_comp_indices.append(idx)
    uncertain_comp_indices = sorted(set(uncertain_comp_indices))
    n_uncertain_comps = len(uncertain_comp_indices)

    # ---- Fixed component lambda sum (never changes) ----
    fixed_mask = np.ones(n_comp, dtype=bool)
    for idx in uncertain_comp_indices:
        fixed_mask[idx] = False
    for i, c in enumerate(components):
        if c.override_lambda is not None:
            fixed_mask[i] = True
    fixed_lambda_sum = nominal_lambdas[fixed_mask].sum()

    # ---- Pre-generate all parameter samples ----
    # Shared: one sample vector per parameter (N,)
    # Independent: one sample per parameter per component (N, n_uses)
    shared_samples = {}    # {param_name: np.ndarray(N,)}  deltas
    indep_samples = {}     # {param_name: {ref: np.ndarray(N,)}}  absolute values

    for spec in shared_specs:
        # Shared: draw a delta from PERT(delta_low, 0, delta_high) or Uniform
        d_lo = spec.delta_low   # typically negative
        d_hi = spec.delta_high  # typically positive
        mode = 0.0  # The mode of the delta is 0 (= nominal)
        shared_samples[spec.name] = _sample_parameter(
            rng, d_lo, mode, d_hi, spec.distribution, n_simulations
        )

    for spec in indep_specs:
        indep_samples[spec.name] = {}
        for ref, nom in spec.nominal_by_ref.items():
            if ref not in ref_to_idx:
                continue
            lo = nom * (1.0 - abs(spec.rel_low) / 100.0)
            hi = nom * (1.0 + abs(spec.rel_high) / 100.0)
            # Ensure lo <= nom <= hi
            lo = min(lo, nom)
            hi = max(hi, nom)
            indep_samples[spec.name][ref] = _sample_parameter(
                rng, lo, nom, hi, spec.distribution, n_simulations
            )

    # ---- SRRC tracking: build input sample matrix ----
    # Columns: one per uncertain spec.
    # For shared specs: the delta value.
    # For independent specs: averaged across components (for ranking).
    srrc_input = np.zeros((n_simulations, n_uncertain))
    srrc_names = []
    for col_idx, spec in enumerate(uncertain_specs):
        srrc_names.append(spec.name)
        if spec.shared:
            srrc_input[:, col_idx] = shared_samples[spec.name]
        else:
            # Average the independent samples across components
            refs = list(indep_samples.get(spec.name, {}).keys())
            if refs:
                stacked = np.column_stack([
                    indep_samples[spec.name][r] for r in refs
                ])
                srrc_input[:, col_idx] = stacked.mean(axis=1)

    # ---- Run MC loop ----
    sys_lambda_samples = np.empty(n_simulations)
    convergence_history = []

    report_interval = max(1, n_simulations // 50)

    for sim in range(n_simulations):
        # Start from fixed lambda sum
        sys_lam = fixed_lambda_sum

        # Evaluate each uncertain component
        for ci in uncertain_comp_indices:
            comp = components[ci]
            if comp.override_lambda is not None:
                sys_lam += comp.override_lambda * 1e-9  # override is in FIT, convert to /h
                continue

            # Build perturbed params
            p = dict(comp.base_params)
            for spec in shared_specs:
                if comp.reference in spec.nominal_by_ref:
                    p[spec.name] = spec.nominal_by_ref[comp.reference] + \
                                   shared_samples[spec.name][sim]
            for spec in indep_specs:
                ref_samps = indep_samples.get(spec.name, {})
                if comp.reference in ref_samps:
                    p[spec.name] = ref_samps[comp.reference][sim]

            # Evaluate IEC formula
            try:
                result = calc_lambda(comp.component_type, p)
                lam_i = result.get("lambda_total", 0.0)
                # Safety: negative lambda is physically impossible
                lam_i = max(0.0, lam_i)
            except Exception:
                lam_i = comp.nominal_lambda
            sys_lam += lam_i

        sys_lambda_samples[sim] = sys_lam

        # Convergence tracking
        if (sim + 1) % report_interval == 0 or sim == n_simulations - 1:
            running_mean = np.mean(
                np.exp(-sys_lambda_samples[:sim + 1] * mission_hours)
            )
            convergence_history.append((sim + 1, float(running_mean)))
            if progress_callback:
                progress_callback(sim + 1, n_simulations, "Sampling...")

    # ---- Compute R(t) samples ----
    reliability_samples = np.exp(-sys_lambda_samples * mission_hours)

    # ---- Statistics ----
    alpha = (1.0 - confidence_level) / 2.0
    mean_r = float(np.mean(reliability_samples))
    median_r = float(np.median(reliability_samples))
    std_r = float(np.std(reliability_samples, ddof=1))
    ci_lo_r = float(np.percentile(reliability_samples, alpha * 100))
    ci_hi_r = float(np.percentile(reliability_samples, (1 - alpha) * 100))

    lambda_fit = sys_lambda_samples * 1e9
    mean_lfit = float(np.mean(lambda_fit))
    std_lfit = float(np.std(lambda_fit, ddof=1))
    ci_lo_lfit = float(np.percentile(lambda_fit, alpha * 100))
    ci_hi_lfit = float(np.percentile(lambda_fit, (1 - alpha) * 100))

    # Half-width CI on the mean (frequentist)
    # hw = z_{1-alpha/2} * s / sqrt(N)
    z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z_val = z.get(confidence_level, 1.645)
    mean_hw = z_val * std_r / math.sqrt(n_simulations) if n_simulations > 1 else 0.0

    # ---- SRRC ----
    importance = []
    if n_uncertain > 0 and n_simulations > n_uncertain + 2:
        srrc_values = _compute_srrc(srrc_input, sys_lambda_samples)
        for i, name in enumerate(srrc_names):
            importance.append({
                "name": name,
                "srrc": float(srrc_values[i]),
                "srrc_sq": float(srrc_values[i] ** 2),
                "n_components": uncertain_specs[i].n_components,
                "shared": uncertain_specs[i].shared,
            })
        # Sort by |SRRC| descending
        importance.sort(key=lambda x: -abs(x["srrc"]))
        # Assign ranks
        for rank, entry in enumerate(importance, 1):
            entry["rank"] = rank
        # Normalise srrc_sq to sum to <= 1 (R^2 decomposition)
        total_sq = sum(x["srrc_sq"] for x in importance)
        if total_sq > 0:
            for entry in importance:
                entry["variance_fraction"] = entry["srrc_sq"] / total_sq
        else:
            for entry in importance:
                entry["variance_fraction"] = 0.0

    # ---- Jensen's note ----
    if mean_r < nominal_R - 1e-6:
        jensen_note = (
            f"Mean R(t) = {mean_r:.6f} is lower than the nominal point estimate "
            f"R(t) = {nominal_R:.6f}. This is a mathematical property of Jensen's "
            f"inequality: since R(t) = exp(-lambda*t) is a strictly convex function "
            f"of lambda, E[R(t)] < R(t; E[lambda]) whenever lambda has positive "
            f"variance. This is not an error -- it quantifies the reliability cost "
            f"of parameter uncertainty."
        )
    else:
        jensen_note = (
            "Mean R(t) is consistent with the nominal estimate. Parameter "
            "uncertainty has negligible impact on expected reliability."
        )

    runtime = time.time() - t0

    return UncertaintyResult(
        nominal_lambda=nominal_sys_lambda,
        nominal_reliability=nominal_R,
        nominal_mttf_hours=nominal_mttf,
        mean_reliability=mean_r,
        median_reliability=median_r,
        std_reliability=std_r,
        ci_lower=ci_lo_r,
        ci_upper=ci_hi_r,
        confidence_level=confidence_level,
        mean_lambda_fit=mean_lfit,
        std_lambda_fit=std_lfit,
        ci_lower_lambda_fit=ci_lo_lfit,
        ci_upper_lambda_fit=ci_hi_lfit,
        mean_ci_halfwidth=mean_hw,
        lambda_samples=sys_lambda_samples,
        reliability_samples=reliability_samples,
        parameter_importance=importance,
        convergence_history=convergence_history,
        n_simulations=n_simulations,
        n_uncertain_params=n_uncertain,
        n_shared_params=len(shared_specs),
        n_uncertain_components=n_uncertain_comps,
        n_total_components=n_comp,
        runtime_seconds=runtime,
        jensen_note=jensen_note,
    )


# =====================================================================
# Helper: Build ComponentInput list from sheet_data
# =====================================================================

def build_component_inputs(
    sheet_data: Dict[str, Dict],
    active_sheets: Optional[List[str]] = None,
    excluded_types: Optional[set] = None,
) -> List[ComponentInput]:
    """Extract ComponentInput list from the standard sheet_data dict.

    Parameters
    ----------
    sheet_data : dict
        {sheet_path: {lambda, r, components: [...]}}
    active_sheets : list, optional
        Only include these sheets.
    excluded_types : set, optional
        Component types to exclude.

    Returns
    -------
    List of ComponentInput, one per component.
    """
    calc_lambda, _ = _import_reliability_math()
    excluded = excluded_types or set()

    if active_sheets:
        filtered = {k: v for k, v in sheet_data.items() if k in active_sheets}
    else:
        filtered = sheet_data

    inputs = []
    for path, data in filtered.items():
        for comp in data.get("components", []):
            ctype = comp.get("class", "Unknown")
            if ctype in excluded:
                continue
            ref = comp.get("ref", "?")
            params = comp.get("params", {})
            override = comp.get("override_lambda")

            if override is not None:
                nom_lam = override * 1e-9  # override is in FIT, convert to /h
            else:
                nom_lam = float(comp.get("lambda", 0) or 0)

            inputs.append(ComponentInput(
                reference=ref,
                component_type=ctype,
                base_params=dict(params),
                nominal_lambda=nom_lam,
                override_lambda=override,
                uncertain_field_names=[],
            ))

    return inputs


# =====================================================================
# Helper: Build default ParameterSpec list from components
# =====================================================================

def build_default_param_specs(
    components: List[ComponentInput],
    global_uncertainty_pct: float = 10.0,
    distribution: str = "pert",
    shared_params: Optional[set] = None,
) -> List[ParameterSpec]:
    """Build ParameterSpec list by scanning components for numeric fields.

    Default shared parameters (environmental, board-level):
        t_ambient, n_cycles, delta_t

    Default independent parameters (component-specific):
        Everything else (t_junction, tau_on, operating_power, ...)

    Parameters
    ----------
    global_uncertainty_pct : float
        Applied as relative +/- percent for independent params,
        and as absolute delta scaled to typical ranges for shared params.
    distribution : str
        "pert" or "uniform"
    shared_params : set, optional
        Parameter names to treat as shared.  Defaults to
        {'t_ambient', 'n_cycles', 'delta_t'}.
    """
    if shared_params is None:
        shared_params = {"t_ambient", "n_cycles", "delta_t"}

    # Scan all components for numeric parameters
    # param_name -> {ref: nominal_value}
    param_nominals = {}
    for comp in components:
        if comp.override_lambda is not None:
            continue
        for pname, pval in comp.base_params.items():
            if pname.startswith("_"):
                continue
            try:
                v = float(pval)
            except (TypeError, ValueError):
                continue
            if v == 0:
                continue
            param_nominals.setdefault(pname, {})[comp.reference] = v

    specs = []
    pct = abs(global_uncertainty_pct)

    for pname, ref_vals in param_nominals.items():
        if not ref_vals:
            continue
        is_shared = pname in shared_params

        if is_shared:
            # Compute typical nominal across components
            vals = list(ref_vals.values())
            typical = np.median(vals)
            # Delta is pct% of the typical value
            delta = abs(typical * pct / 100.0)
            spec = ParameterSpec(
                name=pname,
                nominal_by_ref=dict(ref_vals),
                delta_low=-delta,
                delta_high=delta,
                distribution=distribution,
                shared=True,
            )
        else:
            spec = ParameterSpec(
                name=pname,
                nominal_by_ref=dict(ref_vals),
                rel_low=pct,
                rel_high=pct,
                distribution=distribution,
                shared=False,
            )
        specs.append(spec)

    return specs