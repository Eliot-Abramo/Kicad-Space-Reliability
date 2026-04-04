# KiCad Reliability Calculator

<p align="center">
  <img src="plugins/icon.png" alt="KiCad Reliability Calculator logo" width="120" />
</p>

<p align="center">
  <strong>Architected, implemented, and maintained by Eliot Abramo</strong>
</p>

<p align="center">
  IEC TR 62380 reliability engineering for KiCad projects, from component stress modeling to system-level design closure.
</p>

<p align="center">
  <img alt="Version 3.3.0" src="https://img.shields.io/badge/version-3.3.0-0f766e" />
  <img alt="Author Eliot Abramo" src="https://img.shields.io/badge/author-Eliot%20Abramo-111827" />
  <img alt="IEC TR 62380 2004" src="https://img.shields.io/badge/IEC%20TR%2062380-2004-1d4ed8" />
  <img alt="KiCad plugin" src="https://img.shields.io/badge/KiCad-plugin-f97316" />
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-16a34a" />
</p>

Reliability Calculator is a KiCad plugin that turns a schematic and block diagram into a reliability model. It estimates failure rate at the component level with IEC TR 62380, rolls those results up to the system level, and then gives you a practical way to inspect uncertainty, sensitivity, dominant contributors, and likely improvement paths.

The point is not to produce a mysterious single number. The point is to help you answer, with one tool, what the current design is expected to do, how much that answer depends on your assumptions, and what you should change first if the margin is not good enough.

From the first run, the plugin is meant to answer four practical questions:

- What is the predicted system FIT and mission reliability for this design?
- How uncertain is that estimate if key inputs vary within plausible bounds?
- Which sheets, components, and parameters are driving the result?
- What should the team change first if the design needs a better reliability margin?

The exported report is built around those same questions. It explains the current reliability estimate, shows the uncertainty and sensitivity context around that estimate, and highlights the most actionable contributors and target-closure work. For the detailed math, interpretation limits, and validation notes, see [Methodology and Validation](./docs/METHODOLOGY.md).

The project is deliberately opinionated about engineering communication. It tries to make the math auditable, the assumptions visible, and the improvement path actionable.

## Features

- **It models systems, not just parts.** Component lambdas are lifted into sheet totals and then composed through a user-defined block diagram, so redundancy and architecture are first-class inputs rather than afterthoughts.
- **It distinguishes nominal results from uncertainty.** The plugin does Monte Carlo propagation with shared and independent parameter sampling, then reports distributions, confidence intervals, and ranking aids instead of pretending one deterministic run is the whole story.
- **It supports design closure, not just diagnosis.** Tornado sensitivity, elasticity-based criticality, budget allocation, inverse derating guidance, component swap analysis, and growth tracking all exist to help teams move from "interesting number" to "next design action."
- **It produces review-grade artifacts.** HTML reports include structured sections, tables, and inline SVG charts, with Markdown, CSV, JSON, and optional PDF export available for downstream workflows.
- **IEC TR 62380 component modeling:** 12 component families with temperature, voltage, thermal cycling, package, and usage stress terms implemented in code.
- **Mission-profile support:** single-phase and multi-phase mission definitions with per-phase temperature, duty-cycle, thermal cycling, and duration weighting.
- **Topology-aware system reliability:** block modeling for series, parallel, and k-of-n structures evaluated in reliability space.
- **Monte Carlo uncertainty propagation:** user-bounded sampling through the full component formulas, with shared-parameter support and SRRC-based ranking.
- **Deterministic leverage analysis:** tornado sensitivity and component-level parameter criticality via elasticity.
- **Target-closure tooling:** design-margin scenarios, reliability budget allocation, derating guidance, and component swap ranking.
- **Reliability growth tracking:** saved snapshots, revision comparison, attribution of FIT deltas, and trend visualization across design history.
- **Report export:** HTML, Markdown, CSV, JSON, and optional PDF when `weasyprint` or `reportlab` is installed.

## What The Plugin Is Really Doing

At a high level, the workflow looks like this:

```text
KiCad schematic data
-> component classification + IEC parameters
-> component lambda_i
-> sheet-level aggregation
-> block-diagram composition
-> mission reliability metrics
-> uncertainty / sensitivity / closure analyses
-> exportable engineering report
```

That matters because the tool is not just decorating a BOM with handbook values.
It is coupling component stress modeling, mission assumptions, and explicit system topology in one place.

## Typical Engineering Loop

1. Import the schematic-derived component set into the plugin.
2. Classify parts and fill in the IEC parameters that actually drive the physics.
3. Define mission duration, duty cycle, thermal cycling, and temperature assumptions.
4. Build the block diagram so the reliability model matches the intended architecture.
5. Run the overview to locate dominant sheets and components.
6. Run Monte Carlo to understand distribution, spread, and confidence interval.
7. Run tornado and criticality analyses to rank the highest-leverage assumptions.
8. Use scenario, budget, derating, and swap tools to plan how to close the gap.
9. Save snapshots and compare revisions to track whether the design is actually improving.
10. Export HTML, Markdown, JSON, CSV, or PDF artifacts for review.

## Documentation Map

- [Methodology and Validation](./docs/METHODOLOGY.md): what is shipped, how to interpret it, and where the limits are.
- [Mathematics Reference](./docs/MATHEMATICS.md): equation-level notes tied directly to the implementation.
- [Architecture Overview](./docs/ARCHITECTURE.md): module map, data flow, and why the design is structured the way it is.

## Installation

1. Open KiCad Plugin and Content Manager and click `Install from File...`.
2. Choose the ZIP from the [`Releases`](./Releases) folder or a packaged release artifact.
3. Open a KiCad project and launch **Reliability Calculator** from the action plugins menu.
4. If you want PDF export, install one of the optional backends:
   - `weasyprint`
   - `reportlab`

### How to use after installing

The UI is quite explicit. It is important to note to select multiple blocs for k-of-n or series, etc. configurations you need to CTRL+CLICK on the blocs then right click and the options will present themselves.

## Report Outputs

- **HTML:** the best default output, with styled tables, charts, and full review narrative.
- **PDF:** available when `weasyprint` or `reportlab` is installed.
- **Markdown:** useful for design notebooks, issue trackers, and repositories.
- **CSV:** tabular export of component-level results.
- **JSON:** structured machine-readable output for automation and custom pipelines.

## Assumptions And Limits

- The reported mission reliability is based on IEC TR 62380-style constant hazard-rate modeling over the stated interval.
- Monte Carlo results are only as defensible as the parameter bounds, distributions, and component metadata entered by the user.
- Tornado and criticality analyses are local leverage tools around the current design point.
- SRRC is used as a monotonic ranking aid, not as a claim of full variance decomposition.
- High-consequence release decisions still require independent engineering review.

## Why This Exists

I first started building this tool at Spacelocker after seeing how reliability work on electronics in the space industry often turned into a documentation chore: read a huge standard, fill a spreadsheet, and learn almost nothing about what to change. The plugin is an attempt to turn that process into an actual design workflow.

Special thanks to Louise Grangette for the first pass on some of the mathematics used in this tool.

## License

MIT. See [LICENSE](./LICENSE).
