"""
Mission Profile Phasing Module
===============================
Multi-phase mission profile support per IEC TR 62380:2004.

The standard's mathematical model sums over Y phases, each with its own:
  - Temperature (T_ambient or T_junction)
  - Thermal cycling parameters (n_cycles, delta_t)
  - Duty cycle (tau_on)
  - Duration fraction (pi_t_i = fraction of total mission in this phase)

The general formula for die failure rate with phasing:

    lambda_die = (lambda_1 * N * e^{-0.35*a} + lambda_2)
                 * SUM_i(pi_thermal_i * tau_i * pi_t_i)
                 / SUM_i(tau_i * pi_t_i)

    lambda_pkg = 2.75e-3 * pi_alpha * lambda_3
                 * SUM_i(pi_n_i * delta_t_i^0.68 * pi_t_i)

where pi_t_i is the fractional time in phase i (sum = 1.0).

Author:  Eliot Abramo
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy


@dataclass
class MissionPhase:
    """Single mission phase definition.

    Attributes:
        name:           Human-readable phase name (e.g. "Sunlit", "Eclipse")
        duration_frac:  Fraction of total mission time in this phase (0-1).
                        All phases must sum to 1.0.
        t_ambient:      Ambient temperature for this phase (deg C)
        t_junction:     Junction temperature for this phase (deg C).
                        If None, uses t_ambient + component-specific delta.
        n_cycles:       Annual thermal cycles during this phase
        delta_t:        Temperature excursion per cycle (deg C)
        tau_on:         Working time ratio during this phase (0-1)
    """
    name: str = "Nominal"
    duration_frac: float = 1.0
    t_ambient: float = 25.0
    t_junction: Optional[float] = None
    n_cycles: int = 5256
    delta_t: float = 3.0
    tau_on: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_frac": self.duration_frac,
            "t_ambient": self.t_ambient,
            "t_junction": self.t_junction,
            "n_cycles": self.n_cycles,
            "delta_t": self.delta_t,
            "tau_on": self.tau_on,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MissionPhase":
        return cls(
            name=d.get("name", "Nominal"),
            duration_frac=float(d.get("duration_frac", 1.0)),
            t_ambient=float(d.get("t_ambient", 25.0)),
            t_junction=d.get("t_junction"),
            n_cycles=int(d.get("n_cycles", 5256)),
            delta_t=float(d.get("delta_t", 3.0)),
            tau_on=float(d.get("tau_on", 1.0)),
        )


@dataclass
class MissionProfile:
    """Complete multi-phase mission profile.

    A mission profile defines the environmental conditions across the entire
    mission lifetime. Each phase represents a distinct operating regime
    (e.g., launch vibration, orbit sunlit, orbit eclipse, safe mode).

    The duration_frac of all phases must sum to 1.0 (normalized).
    """
    phases: List[MissionPhase] = field(default_factory=lambda: [MissionPhase()])
    mission_years: float = 5.0

    @property
    def mission_hours(self) -> float:
        return self.mission_years * 365.0 * 24.0

    @property
    def is_single_phase(self) -> bool:
        return len(self.phases) <= 1

    def validate(self) -> Tuple[bool, str]:
        """Validate the mission profile. Returns (valid, message)."""
        if not self.phases:
            return False, "Mission profile must have at least one phase."

        total_frac = sum(p.duration_frac for p in self.phases)
        if abs(total_frac - 1.0) > 0.01:
            return False, (
                f"Phase duration fractions sum to {total_frac:.3f}, "
                f"must sum to 1.0 (within 1%)."
            )

        for i, p in enumerate(self.phases):
            if p.duration_frac < 0 or p.duration_frac > 1:
                return False, f"Phase '{p.name}' has invalid duration fraction: {p.duration_frac}"
            if p.tau_on < 0 or p.tau_on > 1:
                return False, f"Phase '{p.name}' has invalid tau_on: {p.tau_on}"
            if p.n_cycles < 0:
                return False, f"Phase '{p.name}' has negative n_cycles: {p.n_cycles}"
            if p.delta_t < 0:
                return False, f"Phase '{p.name}' has negative delta_t: {p.delta_t}"

        return True, "Valid"

    def normalize(self):
        """Normalize duration fractions to sum to 1.0."""
        total = sum(p.duration_frac for p in self.phases)
        if total > 0:
            for p in self.phases:
                p.duration_frac /= total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_years": self.mission_years,
            "phases": [p.to_dict() for p in self.phases],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MissionProfile":
        phases_data = d.get("phases", [])
        phases = [MissionPhase.from_dict(pd) for pd in phases_data] if phases_data else [MissionPhase()]
        return cls(
            phases=phases,
            mission_years=float(d.get("mission_years", 5.0)),
        )

    @classmethod
    def single_phase(cls, years: float = 5.0, n_cycles: int = 5256,
                     delta_t: float = 3.0, tau_on: float = 1.0,
                     t_ambient: float = 25.0) -> "MissionProfile":
        """Create a single-phase mission profile (backward compatible)."""
        return cls(
            phases=[MissionPhase(
                name="Nominal",
                duration_frac=1.0,
                t_ambient=t_ambient,
                n_cycles=n_cycles,
                delta_t=delta_t,
                tau_on=tau_on,
            )],
            mission_years=years,
        )


# =========================================================================
# Predefined mission profile templates
# =========================================================================

MISSION_TEMPLATES = {
    "Ground (Continuous)": MissionProfile(
        phases=[MissionPhase(
            name="Continuous Operation",
            duration_frac=1.0,
            t_ambient=25.0, n_cycles=365, delta_t=5.0, tau_on=1.0,
        )],
        mission_years=10.0,
    ),
    "Ground (Office 8h/day)": MissionProfile(
        phases=[
            MissionPhase(name="Operating", duration_frac=0.33,
                         t_ambient=30.0, n_cycles=365, delta_t=10.0, tau_on=1.0),
            MissionPhase(name="Standby", duration_frac=0.67,
                         t_ambient=22.0, n_cycles=365, delta_t=3.0, tau_on=0.0),
        ],
        mission_years=10.0,
    ),
    "Automotive (Daily commute)": MissionProfile(
        phases=[
            MissionPhase(name="Engine On", duration_frac=0.08,
                         t_ambient=85.0, n_cycles=730, delta_t=40.0, tau_on=1.0),
            MissionPhase(name="Parked (day)", duration_frac=0.42,
                         t_ambient=45.0, n_cycles=365, delta_t=20.0, tau_on=0.0),
            MissionPhase(name="Parked (night)", duration_frac=0.50,
                         t_ambient=10.0, n_cycles=365, delta_t=10.0, tau_on=0.0),
        ],
        mission_years=15.0,
    ),
    "LEO Satellite": MissionProfile(
        phases=[
            MissionPhase(name="Sunlit", duration_frac=0.60,
                         t_ambient=60.0, n_cycles=5256, delta_t=30.0, tau_on=1.0),
            MissionPhase(name="Eclipse", duration_frac=0.37,
                         t_ambient=-20.0, n_cycles=5256, delta_t=30.0, tau_on=0.8),
            MissionPhase(name="Safe Mode", duration_frac=0.03,
                         t_ambient=0.0, n_cycles=100, delta_t=5.0, tau_on=0.1),
        ],
        mission_years=7.0,
    ),
    "GEO Satellite": MissionProfile(
        phases=[
            MissionPhase(name="Sunlit", duration_frac=0.94,
                         t_ambient=50.0, n_cycles=730, delta_t=15.0, tau_on=1.0),
            MissionPhase(name="Eclipse Season", duration_frac=0.05,
                         t_ambient=-10.0, n_cycles=90, delta_t=40.0, tau_on=1.0),
            MissionPhase(name="Station Keeping", duration_frac=0.01,
                         t_ambient=40.0, n_cycles=24, delta_t=10.0, tau_on=0.5),
        ],
        mission_years=15.0,
    ),
    "Avionics (Flight cycle)": MissionProfile(
        phases=[
            MissionPhase(name="Ground Idle", duration_frac=0.15,
                         t_ambient=30.0, n_cycles=1460, delta_t=15.0, tau_on=0.5),
            MissionPhase(name="Takeoff/Climb", duration_frac=0.10,
                         t_ambient=45.0, n_cycles=1460, delta_t=25.0, tau_on=1.0),
            MissionPhase(name="Cruise", duration_frac=0.60,
                         t_ambient=40.0, n_cycles=730, delta_t=5.0, tau_on=1.0),
            MissionPhase(name="Descent/Landing", duration_frac=0.15,
                         t_ambient=40.0, n_cycles=1460, delta_t=20.0, tau_on=1.0),
        ],
        mission_years=20.0,
    ),
    "Industrial (24/7)": MissionProfile(
        phases=[MissionPhase(
            name="Continuous",
            duration_frac=1.0,
            t_ambient=40.0, n_cycles=1000, delta_t=10.0, tau_on=1.0,
        )],
        mission_years=20.0,
    ),
}


# =========================================================================
# Phase-aware calculation helpers
# =========================================================================

def compute_phased_die_factor(
    phases: List[MissionPhase],
    ea: float,
    t_ref: float,
    t_junction_override: Optional[float] = None,
) -> float:
    """Compute the phased die acceleration factor.

    Returns the weighted sum: SUM_i(pi_thermal_i * tau_on_i * frac_i)

    For single-phase, this reduces to: pi_thermal * tau_on
    (identical to the non-phased formula).

    Args:
        phases:              List of mission phases
        ea:                  Activation energy Ea/k_B in Kelvin
        t_ref:               Reference temperature in Kelvin
        t_junction_override: If set, use this T_junction for all phases
                             (component-level override from editor)
    """
    from .reliability_math import pi_temperature

    weighted_sum = 0.0
    for phase in phases:
        if t_junction_override is not None:
            t_op = t_junction_override
        elif phase.t_junction is not None:
            t_op = phase.t_junction
        else:
            t_op = phase.t_ambient

        pi_t = pi_temperature(t_op, ea, t_ref)
        weighted_sum += pi_t * phase.tau_on * phase.duration_frac

    return weighted_sum


def compute_phased_pkg_factor(phases: List[MissionPhase]) -> float:
    """Compute the phased package stress factor.

    Returns: SUM_i(pi_n_i * delta_t_i^0.68 * frac_i)

    For single-phase, this reduces to: pi_n * delta_t^0.68
    (identical to the non-phased formula).
    """
    from .reliability_math import pi_thermal_cycles

    weighted_sum = 0.0
    for phase in phases:
        pi_n = pi_thermal_cycles(phase.n_cycles)
        dt_factor = max(0.0, phase.delta_t) ** 0.68
        weighted_sum += pi_n * dt_factor * phase.duration_frac

    return weighted_sum


def compute_phased_tau_weighted(phases: List[MissionPhase]) -> float:
    """Compute the effective weighted tau_on across phases.

    Returns: SUM_i(tau_on_i * frac_i)
    """
    return sum(p.tau_on * p.duration_frac for p in phases)


def override_params_for_phase(
    base_params: Dict[str, Any],
    phase: MissionPhase,
) -> Dict[str, Any]:
    """Create a parameter dict with phase-specific environmental values.

    Used for per-phase recalculation in sensitivity/what-if analysis.
    Preserves component-specific parameters while overriding environmental ones.
    """
    params = dict(base_params)
    params["t_ambient"] = phase.t_ambient
    if phase.t_junction is not None:
        params["t_junction"] = phase.t_junction
    params["n_cycles"] = phase.n_cycles
    params["delta_t"] = phase.delta_t
    params["tau_on"] = phase.tau_on
    return params


def compute_phased_lambda(
    component_type: str,
    base_params: Dict[str, Any],
    phases: List[MissionPhase],
) -> Dict[str, Any]:
    """Compute failure rate for a component across multiple mission phases.

    This is the primary entry point for phased calculations. For each phase,
    it computes the component's failure rate under that phase's conditions,
    then returns the weighted average.

    For single-phase profiles, this gives identical results to the
    non-phased calculate_component_lambda.

    Returns:
        Dict with lambda_total, fit_total, and per-phase breakdown.
    """
    try:
        from .reliability_math import calculate_component_lambda
    except ImportError:
        from reliability_math import calculate_component_lambda

    if not phases or len(phases) == 1:
        # Single-phase: use original calculation directly
        phase = phases[0] if phases else MissionPhase()
        params = override_params_for_phase(base_params, phase)
        return calculate_component_lambda(component_type, params)

    # Multi-phase: weighted sum
    total_lambda = 0.0
    phase_results = []

    for phase in phases:
        params = override_params_for_phase(base_params, phase)
        try:
            result = calculate_component_lambda(component_type, params)
            phase_lambda = result.get("lambda_total", 0.0)
        except Exception:
            phase_lambda = 0.0
            result = {"lambda_total": 0.0, "fit_total": 0.0}

        weighted_lambda = phase_lambda * phase.duration_frac
        total_lambda += weighted_lambda

        phase_results.append({
            "phase_name": phase.name,
            "duration_frac": phase.duration_frac,
            "lambda_phase": phase_lambda,
            "lambda_weighted": weighted_lambda,
            "fit_phase": phase_lambda * 1e9,
            "fit_weighted": weighted_lambda * 1e9,
            "details": result,
        })

    return {
        "lambda_total": total_lambda,
        "fit_total": total_lambda * 1e9,
        "phase_breakdown": phase_results,
        "n_phases": len(phases),
    }


def estimate_phasing_impact(
    component_type: str,
    base_params: Dict[str, Any],
    phases: List[MissionPhase],
) -> Dict[str, Any]:
    """Compare phased vs. averaged single-phase calculation.

    Quantifies how much the multi-phase model differs from the simplified
    single-phase average. This helps engineers understand the importance
    of proper mission profiling.
    """
    phased = compute_phased_lambda(component_type, base_params, phases)
    phased_lambda = phased.get("lambda_total", 0.0)

    # Compute single-phase with averaged conditions
    avg_t_amb = sum(p.t_ambient * p.duration_frac for p in phases)
    avg_t_junc = None
    t_junc_vals = [p.t_junction for p in phases if p.t_junction is not None]
    if t_junc_vals:
        avg_t_junc = sum(
            p.t_junction * p.duration_frac
            for p in phases if p.t_junction is not None
        ) / sum(p.duration_frac for p in phases if p.t_junction is not None)
    avg_cycles = sum(p.n_cycles * p.duration_frac for p in phases)
    avg_dt = sum(p.delta_t * p.duration_frac for p in phases)
    avg_tau = sum(p.tau_on * p.duration_frac for p in phases)

    avg_phase = MissionPhase(
        name="Averaged",
        duration_frac=1.0,
        t_ambient=avg_t_amb,
        t_junction=avg_t_junc,
        n_cycles=int(avg_cycles),
        delta_t=avg_dt,
        tau_on=avg_tau,
    )
    avg_result = compute_phased_lambda(component_type, base_params, [avg_phase])
    avg_lambda = avg_result.get("lambda_total", 0.0)

    ratio = phased_lambda / avg_lambda if avg_lambda > 0 else 1.0
    delta_pct = (ratio - 1.0) * 100.0

    return {
        "phased_lambda": phased_lambda,
        "phased_fit": phased_lambda * 1e9,
        "averaged_lambda": avg_lambda,
        "averaged_fit": avg_lambda * 1e9,
        "ratio": ratio,
        "delta_percent": delta_pct,
        "conclusion": (
            f"Phased model gives {abs(delta_pct):.1f}% "
            f"{'higher' if delta_pct > 0 else 'lower'} failure rate "
            f"than single-phase average."
        ),
    }
