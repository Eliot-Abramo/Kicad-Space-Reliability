"""
Correlated Monte Carlo Module
===============================
Extends the Monte Carlo engine with correlation support using
Cholesky decomposition for correlated normal/lognormal sampling.

In reality, parameters are correlated:
  - T_junction and T_ambient share thermal environment
  - Power and temperature have physical coupling
  - Components on the same board share environmental conditions

Ignoring correlations can under- or over-estimate system variance.

Mathematical approach:
  1. User defines correlation groups (e.g., "same board" or "same thermal zone")
  2. Build correlation matrix for grouped parameters
  3. Use Cholesky decomposition: L = chol(Sigma)
  4. Generate independent normal samples Z
  5. Correlated samples X = mu + L @ Z
  6. Transform to lognormal domain for failure rates

Author:  Eliot Abramo
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import time


@dataclass
class CorrelationGroup:
    """A group of components with correlated parameters.

    Components in the same group share environmental conditions
    (e.g., same board, same thermal zone, same power supply).
    """
    name: str
    component_refs: List[str]    # References of components in this group
    correlation: float = 0.80    # Intra-group correlation coefficient (0-1)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component_refs": self.component_refs,
            "correlation": self.correlation,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CorrelationGroup":
        return cls(
            name=d.get("name", ""),
            component_refs=d.get("component_refs", []),
            correlation=float(d.get("correlation", 0.8)),
            description=d.get("description", ""),
        )


@dataclass
class CorrelatedMCResult:
    """Results from correlated Monte Carlo simulation."""
    # Independent MC results (for comparison)
    independent_mean: float
    independent_std: float
    independent_ci: Tuple[float, float]

    # Correlated MC results
    correlated_mean: float
    correlated_std: float
    correlated_ci: Tuple[float, float]

    # Comparison
    std_ratio: float          # correlated_std / independent_std
    ci_width_ratio: float     # correlated CI width / independent CI width
    variance_impact: str      # "wider", "narrower", or "similar"

    # Samples
    correlated_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    independent_samples: np.ndarray = field(default_factory=lambda: np.array([]))

    n_simulations: int = 0
    n_groups: int = 0
    runtime_seconds: float = 0.0
    confidence_level: float = 0.90

    def to_dict(self) -> Dict[str, Any]:
        return {
            "independent_mean": self.independent_mean,
            "independent_std": self.independent_std,
            "independent_ci": list(self.independent_ci),
            "correlated_mean": self.correlated_mean,
            "correlated_std": self.correlated_std,
            "correlated_ci": list(self.correlated_ci),
            "std_ratio": self.std_ratio,
            "ci_width_ratio": self.ci_width_ratio,
            "variance_impact": self.variance_impact,
            "n_simulations": self.n_simulations,
            "n_groups": self.n_groups,
            "runtime_seconds": self.runtime_seconds,
            "confidence_level": self.confidence_level,
        }


def _build_correlation_matrix(
    n_components: int,
    component_refs: List[str],
    groups: List[CorrelationGroup],
) -> np.ndarray:
    """Build the n x n correlation matrix from group definitions.

    Components within the same group have intra-group correlation.
    Components in different groups or ungrouped are independent.
    """
    # Start with identity (all independent)
    corr = np.eye(n_components)

    # Build ref -> index mapping
    ref_to_idx = {ref: i for i, ref in enumerate(component_refs)}

    # Apply group correlations
    for group in groups:
        rho = max(0.0, min(1.0, group.correlation))
        group_indices = [ref_to_idx[ref] for ref in group.component_refs
                         if ref in ref_to_idx]

        for i in range(len(group_indices)):
            for j in range(i + 1, len(group_indices)):
                idx_i = group_indices[i]
                idx_j = group_indices[j]
                corr[idx_i, idx_j] = rho
                corr[idx_j, idx_i] = rho

    return corr


def _cholesky_safe(corr_matrix: np.ndarray) -> np.ndarray:
    """Compute Cholesky decomposition with numerical safety.

    If the correlation matrix is not positive definite (can happen with
    conflicting group assignments), applies nearest PD correction.
    """
    try:
        return np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Nearest positive definite matrix correction
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-10)  # Clamp negative eigenvalues
        corrected = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalize diagonal to 1
        d = np.sqrt(np.diag(corrected))
        corrected = corrected / np.outer(d, d)
        return np.linalg.cholesky(corrected)


def correlated_monte_carlo(
    component_lambdas: np.ndarray,
    component_refs: List[str],
    groups: List[CorrelationGroup],
    mission_hours: float,
    n_simulations: int = 5000,
    uncertainty_percent: float = 20.0,
    confidence_level: float = 0.90,
    seed: Optional[int] = None,
    is_fixed: Optional[np.ndarray] = None,
) -> CorrelatedMCResult:
    """
    Run Monte Carlo with correlated failure rate sampling.

    Args:
        component_lambdas: Array of nominal lambda values (per hour)
        component_refs:    Component reference designators
        groups:            Correlation groups
        mission_hours:     Mission duration
        n_simulations:     Number of MC samples
        uncertainty_percent: CV as percentage
        confidence_level:  CI level (e.g. 0.90)
        seed:              RNG seed
        is_fixed:          Boolean array; True = fixed lambda (no perturbation)

    Returns:
        CorrelatedMCResult with both independent and correlated results
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    n_comp = len(component_lambdas)
    cv = uncertainty_percent / 100.0

    if is_fixed is None:
        is_fixed = np.zeros(n_comp, dtype=bool)

    variable = ~is_fixed & (component_lambdas > 0)

    # ---- Independent sampling (baseline) ----
    lambda_matrix_indep = np.tile(component_lambdas, (n_simulations, 1))
    if cv > 0 and variable.any():
        var_lambdas = component_lambdas[variable]
        mu = np.log(var_lambdas) - 0.5 * np.log(1 + cv**2)
        sigma = np.sqrt(np.log(1 + cv**2))
        lambda_matrix_indep[:, variable] = rng.lognormal(
            mu[np.newaxis, :], sigma, size=(n_simulations, variable.sum())
        )

    sys_lambda_indep = lambda_matrix_indep.sum(axis=1)
    sys_r_indep = np.exp(-sys_lambda_indep * mission_hours)

    # ---- Correlated sampling ----
    if not groups or n_comp < 2:
        # No correlations: correlated = independent
        sys_r_corr = sys_r_indep.copy()
    else:
        # Build correlation matrix
        corr = _build_correlation_matrix(n_comp, component_refs, groups)

        # Generate correlated standard normal samples via Cholesky
        L = _cholesky_safe(corr)

        # Generate independent standard normals
        Z = rng.standard_normal((n_simulations, n_comp))

        # Apply correlation: X = L @ Z^T => each row is a correlated sample
        X_corr = (L @ Z.T).T  # (n_simulations, n_comp)

        # Transform to lognormal domain
        lambda_matrix_corr = np.tile(component_lambdas, (n_simulations, 1))
        if cv > 0 and variable.any():
            var_lambdas = component_lambdas[variable]
            mu = np.log(var_lambdas) - 0.5 * np.log(1 + cv**2)
            sigma_val = np.sqrt(np.log(1 + cv**2))

            # Use correlated normals for variable components
            # Map variable indices to full index
            var_indices = np.where(variable)[0]
            for k, full_idx in enumerate(var_indices):
                # Transform correlated normal to lognormal
                lambda_matrix_corr[:, full_idx] = np.exp(
                    mu[k] + sigma_val * X_corr[:, full_idx]
                )

        sys_lambda_corr = lambda_matrix_corr.sum(axis=1)
        sys_r_corr = np.exp(-sys_lambda_corr * mission_hours)

    # ---- Compute statistics ----
    ci_alpha = (1 - confidence_level) / 2

    indep_mean = float(np.mean(sys_r_indep))
    indep_std = float(np.std(sys_r_indep))
    indep_ci = (
        float(np.percentile(sys_r_indep, ci_alpha * 100)),
        float(np.percentile(sys_r_indep, (1 - ci_alpha) * 100)),
    )

    corr_mean = float(np.mean(sys_r_corr))
    corr_std = float(np.std(sys_r_corr))
    corr_ci = (
        float(np.percentile(sys_r_corr, ci_alpha * 100)),
        float(np.percentile(sys_r_corr, (1 - ci_alpha) * 100)),
    )

    std_ratio = corr_std / max(indep_std, 1e-15)
    ci_width_indep = indep_ci[1] - indep_ci[0]
    ci_width_corr = corr_ci[1] - corr_ci[0]
    ci_ratio = ci_width_corr / max(ci_width_indep, 1e-15)

    if ci_ratio > 1.05:
        impact = "wider"
    elif ci_ratio < 0.95:
        impact = "narrower"
    else:
        impact = "similar"

    runtime = time.time() - start_time

    return CorrelatedMCResult(
        independent_mean=indep_mean,
        independent_std=indep_std,
        independent_ci=indep_ci,
        correlated_mean=corr_mean,
        correlated_std=corr_std,
        correlated_ci=corr_ci,
        std_ratio=std_ratio,
        ci_width_ratio=ci_ratio,
        variance_impact=impact,
        correlated_samples=sys_r_corr,
        independent_samples=sys_r_indep,
        n_simulations=n_simulations,
        n_groups=len(groups),
        runtime_seconds=runtime,
        confidence_level=confidence_level,
    )


# =========================================================================
# Auto-grouping helpers
# =========================================================================

def auto_group_by_sheet(
    sheet_data: Dict[str, Dict],
    correlation: float = 0.80,
) -> List[CorrelationGroup]:
    """Automatically create correlation groups from sheet structure.

    All components on the same schematic sheet share environmental conditions.
    """
    groups = []
    for path, data in sheet_data.items():
        refs = [c.get("ref", "?") for c in data.get("components", [])
                if c.get("override_lambda") is None]
        if len(refs) >= 2:
            name = path.rstrip("/").split("/")[-1] or "Root"
            groups.append(CorrelationGroup(
                name=f"Sheet: {name}",
                component_refs=refs,
                correlation=correlation,
                description=f"Components on {name} share thermal environment",
            ))
    return groups


def auto_group_by_type(
    sheet_data: Dict[str, Dict],
    correlation: float = 0.60,
) -> List[CorrelationGroup]:
    """Create correlation groups by component type.

    Components of the same type on the same board tend to have
    correlated failure modes (same manufacturing lot, similar stress).
    """
    type_refs = {}
    for data in sheet_data.values():
        for comp in data.get("components", []):
            if comp.get("override_lambda") is not None:
                continue
            ct = comp.get("class", "Unknown")
            type_refs.setdefault(ct, []).append(comp.get("ref", "?"))

    groups = []
    for ct, refs in type_refs.items():
        if len(refs) >= 2:
            groups.append(CorrelationGroup(
                name=f"Type: {ct}",
                component_refs=refs,
                correlation=correlation,
                description=f"All {ct} components (same type correlation)",
            ))
    return groups


def auto_group_all_on_board(
    sheet_data: Dict[str, Dict],
    correlation: float = 0.50,
) -> List[CorrelationGroup]:
    """Single group: all components share T_ambient (same board)."""
    all_refs = []
    for data in sheet_data.values():
        for comp in data.get("components", []):
            if comp.get("override_lambda") is None:
                all_refs.append(comp.get("ref", "?"))

    if len(all_refs) >= 2:
        return [CorrelationGroup(
            name="All components (shared board)",
            component_refs=all_refs,
            correlation=correlation,
            description="All components share ambient temperature",
        )]
    return []
