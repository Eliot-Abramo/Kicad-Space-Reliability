"""
Reliability Growth Tracking Module
====================================
Tracks system reliability across design revisions, attributing improvements
to specific component changes and showing progress toward targets.

Features:
  - Snapshot storage with timestamps and metadata
  - Before/after comparison with component-level attribution
  - Growth trend analysis
  - Target progress tracking

Snapshots are stored as JSON in the project's Reliability/ folder.

Author:  Eliot Abramo
"""

import json
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class ReliabilitySnapshot:
    """A point-in-time snapshot of system reliability state."""
    timestamp: str
    version_label: str
    notes: str

    system_lambda: float          # /h
    system_fit: float             # FIT
    system_reliability: float
    mission_hours: float

    n_components: int
    n_sheets: int

    # Per-sheet summary
    sheet_summary: Dict[str, Dict] = field(default_factory=dict)
    # {sheet_path: {lambda, fit, r, n_components}}

    # Per-component lambda (for diff attribution)
    component_lambdas: Dict[str, float] = field(default_factory=dict)
    # {reference: lambda_per_hour}

    # Component details for diff
    component_details: Dict[str, Dict] = field(default_factory=dict)
    # {reference: {class, params, fit, ...}}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "version_label": self.version_label,
            "notes": self.notes,
            "system_lambda": self.system_lambda,
            "system_fit": self.system_fit,
            "system_reliability": self.system_reliability,
            "mission_hours": self.mission_hours,
            "n_components": self.n_components,
            "n_sheets": self.n_sheets,
            "sheet_summary": self.sheet_summary,
            "component_lambdas": self.component_lambdas,
            "component_details": self.component_details,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReliabilitySnapshot":
        return cls(
            timestamp=d.get("timestamp", ""),
            version_label=d.get("version_label", ""),
            notes=d.get("notes", ""),
            system_lambda=float(d.get("system_lambda", 0)),
            system_fit=float(d.get("system_fit", 0)),
            system_reliability=float(d.get("system_reliability", 1)),
            mission_hours=float(d.get("mission_hours", 0)),
            n_components=int(d.get("n_components", 0)),
            n_sheets=int(d.get("n_sheets", 0)),
            sheet_summary=d.get("sheet_summary", {}),
            component_lambdas=d.get("component_lambdas", {}),
            component_details=d.get("component_details", {}),
        )


@dataclass
class ComponentChange:
    """A single component-level change between revisions."""
    reference: str
    component_type: str
    change_type: str      # "modified", "added", "removed"
    fit_before: float
    fit_after: float
    delta_fit: float
    delta_percent: float
    parameter_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    # {param_name: (old_value, new_value)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "component_type": self.component_type,
            "change_type": self.change_type,
            "fit_before": self.fit_before,
            "fit_after": self.fit_after,
            "delta_fit": self.delta_fit,
            "delta_percent": self.delta_percent,
            "parameter_changes": {
                k: {"old": v[0], "new": v[1]}
                for k, v in self.parameter_changes.items()
            },
        }


@dataclass
class RevisionComparison:
    """Comparison between two design revisions."""
    from_version: str
    to_version: str
    from_timestamp: str
    to_timestamp: str

    system_fit_before: float
    system_fit_after: float
    system_delta_fit: float
    system_delta_percent: float

    reliability_before: float
    reliability_after: float
    reliability_improvement: float

    component_changes: List[ComponentChange] = field(default_factory=list)
    components_improved: int = 0
    components_degraded: int = 0
    components_added: int = 0
    components_removed: int = 0
    components_unchanged: int = 0

    # Top attributions
    top_improvements: List[ComponentChange] = field(default_factory=list)
    top_degradations: List[ComponentChange] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "from_timestamp": self.from_timestamp,
            "to_timestamp": self.to_timestamp,
            "system_fit_before": self.system_fit_before,
            "system_fit_after": self.system_fit_after,
            "system_delta_fit": self.system_delta_fit,
            "system_delta_percent": self.system_delta_percent,
            "reliability_before": self.reliability_before,
            "reliability_after": self.reliability_after,
            "reliability_improvement": self.reliability_improvement,
            "components_improved": self.components_improved,
            "components_degraded": self.components_degraded,
            "components_added": self.components_added,
            "components_removed": self.components_removed,
            "components_unchanged": self.components_unchanged,
            "top_improvements": [c.to_dict() for c in self.top_improvements],
            "top_degradations": [c.to_dict() for c in self.top_degradations],
            "all_changes": [c.to_dict() for c in self.component_changes],
        }


@dataclass
class GrowthTimeline:
    """Timeline of reliability growth across all snapshots."""
    snapshots: List[ReliabilitySnapshot] = field(default_factory=list)
    target_reliability: Optional[float] = None
    target_fit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_reliability": self.target_reliability,
            "target_fit": self.target_fit,
            "points": [
                {
                    "timestamp": s.timestamp,
                    "version": s.version_label,
                    "fit": s.system_fit,
                    "reliability": s.system_reliability,
                    "n_components": s.n_components,
                }
                for s in self.snapshots
            ],
        }


# =========================================================================
# Snapshot creation from current design state
# =========================================================================

def create_snapshot(
    sheet_data: Dict[str, Dict],
    system_lambda: float,
    mission_hours: float,
    version_label: str = "",
    notes: str = "",
) -> ReliabilitySnapshot:
    """Create a snapshot from current design state."""
    try:
        from .reliability_math import reliability_from_lambda
    except ImportError:
        from reliability_math import reliability_from_lambda

    system_fit = system_lambda * 1e9
    system_r = reliability_from_lambda(system_lambda, mission_hours)

    sheet_summary = {}
    component_lambdas = {}
    component_details = {}
    n_total = 0

    for path, data in sheet_data.items():
        comps = data.get("components", [])
        sheet_summary[path] = {
            "lambda": float(data.get("lambda", 0)),
            "fit": float(data.get("lambda", 0)) * 1e9,
            "r": float(data.get("r", 1)),
            "n_components": len(comps),
        }

        for comp in comps:
            ref = comp.get("ref", "?")
            lam = float(comp.get("lambda", 0) or 0)
            component_lambdas[ref] = lam
            component_details[ref] = {
                "class": comp.get("class", "Unknown"),
                "value": comp.get("value", ""),
                "fit": lam * 1e9,
                "params": comp.get("params", {}),
                "sheet": path,
                "override_lambda": comp.get("override_lambda"),
            }
            n_total += 1

    if not version_label:
        version_label = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return ReliabilitySnapshot(
        timestamp=datetime.now().isoformat(),
        version_label=version_label,
        notes=notes,
        system_lambda=system_lambda,
        system_fit=system_fit,
        system_reliability=system_r,
        mission_hours=mission_hours,
        n_components=n_total,
        n_sheets=len(sheet_data),
        sheet_summary=sheet_summary,
        component_lambdas=component_lambdas,
        component_details=component_details,
    )


# =========================================================================
# Revision comparison
# =========================================================================

def compare_revisions(
    before: ReliabilitySnapshot,
    after: ReliabilitySnapshot,
    threshold_pct: float = 1.0,
) -> RevisionComparison:
    """Compare two snapshots and attribute changes to components.

    Args:
        before:        Earlier revision snapshot
        after:         Later revision snapshot
        threshold_pct: Minimum % change to count as modified (vs unchanged)
    """
    changes = []
    refs_before = set(before.component_lambdas.keys())
    refs_after = set(after.component_lambdas.keys())

    common = refs_before & refs_after
    added = refs_after - refs_before
    removed = refs_before - refs_after

    n_improved = n_degraded = n_unchanged = 0

    # Modified components
    for ref in common:
        lam_b = before.component_lambdas.get(ref, 0)
        lam_a = after.component_lambdas.get(ref, 0)
        fit_b = lam_b * 1e9
        fit_a = lam_a * 1e9
        delta = fit_a - fit_b
        delta_pct = (delta / fit_b * 100) if fit_b > 0 else 0

        if abs(delta_pct) < threshold_pct:
            n_unchanged += 1
            continue

        # Identify parameter changes
        param_changes = {}
        details_b = before.component_details.get(ref, {})
        details_a = after.component_details.get(ref, {})
        params_b = details_b.get("params", {})
        params_a = details_a.get("params", {})

        all_params = set(list(params_b.keys()) + list(params_a.keys()))
        for pname in all_params:
            vb = params_b.get(pname)
            va = params_a.get(pname)
            if vb != va:
                param_changes[pname] = (vb, va)

        comp_type = details_a.get("class", details_b.get("class", "Unknown"))
        change = ComponentChange(
            reference=ref,
            component_type=comp_type,
            change_type="modified",
            fit_before=fit_b,
            fit_after=fit_a,
            delta_fit=delta,
            delta_percent=delta_pct,
            parameter_changes=param_changes,
        )
        changes.append(change)

        if delta < 0:
            n_improved += 1
        else:
            n_degraded += 1

    # Added components
    for ref in added:
        lam = after.component_lambdas.get(ref, 0)
        fit = lam * 1e9
        details = after.component_details.get(ref, {})
        changes.append(ComponentChange(
            reference=ref,
            component_type=details.get("class", "Unknown"),
            change_type="added",
            fit_before=0,
            fit_after=fit,
            delta_fit=fit,
            delta_percent=100.0,
        ))

    # Removed components
    for ref in removed:
        lam = before.component_lambdas.get(ref, 0)
        fit = lam * 1e9
        details = before.component_details.get(ref, {})
        changes.append(ComponentChange(
            reference=ref,
            component_type=details.get("class", "Unknown"),
            change_type="removed",
            fit_before=fit,
            fit_after=0,
            delta_fit=-fit,
            delta_percent=-100.0,
        ))

    # Sort all changes by delta (improvements first)
    changes.sort(key=lambda c: c.delta_fit)

    top_improvements = [c for c in changes if c.delta_fit < 0][:5]
    top_degradations = [c for c in changes if c.delta_fit > 0][-5:]
    top_degradations.reverse()

    sys_delta_fit = after.system_fit - before.system_fit
    sys_delta_pct = (sys_delta_fit / before.system_fit * 100) if before.system_fit > 0 else 0

    return RevisionComparison(
        from_version=before.version_label,
        to_version=after.version_label,
        from_timestamp=before.timestamp,
        to_timestamp=after.timestamp,
        system_fit_before=before.system_fit,
        system_fit_after=after.system_fit,
        system_delta_fit=sys_delta_fit,
        system_delta_percent=sys_delta_pct,
        reliability_before=before.system_reliability,
        reliability_after=after.system_reliability,
        reliability_improvement=after.system_reliability - before.system_reliability,
        component_changes=changes,
        components_improved=n_improved,
        components_degraded=n_degraded,
        components_added=len(added),
        components_removed=len(removed),
        components_unchanged=n_unchanged,
        top_improvements=top_improvements,
        top_degradations=top_degradations,
    )


# =========================================================================
# Storage: save/load snapshots to project folder
# =========================================================================

SNAPSHOTS_FILENAME = "reliability_snapshots.json"


def save_snapshot(snapshot: ReliabilitySnapshot, project_path: str) -> str:
    """Save a snapshot to the project's Reliability folder.

    Returns the path to the snapshots file.
    """
    rel_dir = Path(project_path) / "Reliability"
    rel_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = rel_dir / SNAPSHOTS_FILENAME

    # Load existing snapshots
    existing = []
    if snapshots_path.exists():
        try:
            with open(snapshots_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing = data.get("snapshots", [])
        except Exception:
            existing = []

    # Append new snapshot
    existing.append(snapshot.to_dict())

    # Save
    with open(snapshots_path, "w", encoding="utf-8") as f:
        json.dump({"snapshots": existing}, f, indent=2)

    return str(snapshots_path)


def load_snapshots(project_path: str) -> List[ReliabilitySnapshot]:
    """Load all snapshots from the project's Reliability folder."""
    snapshots_path = Path(project_path) / "Reliability" / SNAPSHOTS_FILENAME
    if not snapshots_path.exists():
        return []

    try:
        with open(snapshots_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [
                ReliabilitySnapshot.from_dict(d)
                for d in data.get("snapshots", [])
            ]
    except Exception:
        return []


def build_growth_timeline(
    project_path: str,
    target_reliability: Optional[float] = None,
    target_fit: Optional[float] = None,
) -> GrowthTimeline:
    """Build a growth timeline from all stored snapshots."""
    snapshots = load_snapshots(project_path)
    snapshots.sort(key=lambda s: s.timestamp)
    return GrowthTimeline(
        snapshots=snapshots,
        target_reliability=target_reliability,
        target_fit=target_fit,
    )


def delete_snapshot(project_path: str, version_label: str) -> bool:
    """Delete a specific snapshot by version label."""
    snapshots_path = Path(project_path) / "Reliability" / SNAPSHOTS_FILENAME
    if not snapshots_path.exists():
        return False

    try:
        with open(snapshots_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        snapshots = data.get("snapshots", [])
        filtered = [s for s in snapshots if s.get("version_label") != version_label]
        if len(filtered) == len(snapshots):
            return False  # Not found

        with open(snapshots_path, "w", encoding="utf-8") as f:
            json.dump({"snapshots": filtered}, f, indent=2)
        return True
    except Exception:
        return False
