# Architecture Overview

## Why The Architecture Matters

This plugin is doing more than decorating a schematic with handbook data.
It combines four layers that are usually split across separate tools:

- component-level IEC TR 62380 stress modeling
- mission and usage modeling
- explicit system-topology composition
- design-closure and reporting workflows

That separation of concerns is what lets the project stay ambitious without
becoming unreadable.

## System Map

At a high level, the architecture looks like this:

```text
KiCad project / schematic
-> schematic_parser.py
-> classification + component_editor
-> reliability_math.py
-> sheet_data
-> block_editor.py + topology helpers
-> analysis modules
-> report_generator.py
-> HTML / PDF / Markdown / CSV / JSON artifacts
```

The key design decision is that most downstream analyses operate on the same
shared design state rather than re-inventing their own private data models.

## Primary Modules

### User-Surface And Workflow

- `plugins/plugin.py`
  Entry point for the KiCad action plugin.
- `plugins/reliability_dialog.py`
  Main workflow dialog for project import, editing, and baseline analysis.
- `plugins/analysis_dialog.py`
  Advanced analysis surface for Monte Carlo, tornado, criticality, budget,
  derating, swap, history, and report-generation workflows.
- `plugins/block_editor.py`
  Visual system-topology editor for series, parallel, and k-of-n structures.

### Core Computation

- `plugins/reliability_math.py`
  The numerical heart of the project. Implements component-family equations,
  shared acceleration factors, unit conversion helpers, and topology utilities.
- `plugins/mission_profile.py`
  Multi-phase mission modeling and per-phase parameter override logic.
- `plugins/analysis/engine.py`
  Topology-aware uncertainty and what-if execution utilities.

### Analysis And Closure Tooling

- `plugins/monte_carlo.py`
  Production Monte Carlo uncertainty propagation, including shared and
  independent parameter sampling plus SRRC ranking.
- `plugins/sensitivity_analysis.py`
  Tornado sensitivity, design scenarios, smart-action heuristics, and
  elasticity-based criticality analysis.
- `plugins/budget_allocation.py`
  Converts system targets into sheet-level and component-level FIT budgets.
- `plugins/derating_engine.py`
  Inverse-search tooling for finding parameter values that can close a gap.
- `plugins/component_swap.py`
  Clone-and-modify swap ranking for packages, technologies, and subtype changes.
- `plugins/growth_tracking.py`
  Revision snapshotting, comparison, and growth-timeline construction.

### Reporting And Persistence

- `plugins/report_generator.py`
  Consolidates analysis outputs into HTML, Markdown, CSV, JSON, and optional
  PDF reports, including inline SVG visualizations.
- `plugins/project_manager.py`
  Project-path and persistence helpers.
- `plugins/schematic_parser.py`
  Extracts component data from KiCad project structures.

## Data Model In Practice

The analysis pipeline revolves around a few durable ideas:

- **Component records**
  Each component carries a type, a parameter set, and a computed lambda or
  override.
- **Sheet data**
  Components are grouped by sheet so the tool can report both local and global
  structure without flattening everything into one bag of parts.
- **Block graph**
  Sheets become leaves inside a hierarchical reliability topology composed of
  series, parallel, and k-of-n groups.
- **Report payload**
  The report layer consumes a structured `ReportData` object so every export
  format is built from the same analysis state.

This architecture keeps the math modules relatively pure while allowing the UI
and report layers to evolve without duplicating core calculations.

## Design Rational

Several key choices shape this tool:
- **One shared numerical backbone**
  Monte Carlo, mission phasing, what-if scenarios, derating, and swap analysis
  all route back to the same component-level math instead of drifting into
  inconsistent approximations.
- **Reliability-space topology evaluation**
  The design correctly switches from lambda aggregation to reliability
  composition once redundancy appears, which is essential for meaningful
  architecture-level conclusions.
- **Action-oriented downstream tooling**
  The codebase does not stop at estimation. It includes explicit mechanisms for
  target closure, revision comparison, and report-ready communication.
- **Public-surface discipline**
  Tests guard against advertising features that are not truly shipped, which is
  a subtle but important architectural value in an engineering tool.

## If You Want To Trace A Result

For most questions, these are the fastest entry points:

- Start in `plugins/reliability_dialog.py` or `plugins/analysis_dialog.py` to
  see how the user initiates the calculation.
- Follow into `plugins/reliability_math.py` for component-level formulas.
- Use `plugins/mission_profile.py` for phased operation logic.
- Use `plugins/monte_carlo.py` and `plugins/sensitivity_analysis.py` for
  uncertainty and leverage analyses.
- Use `plugins/report_generator.py` to see how results become review artifacts.

The payoff of the architecture is that a number visible in the UI can usually
be traced back to a compact, named function rather than a pile of hidden state.
