# KiCad Reliability Calculator Plugin

**Version:** 3.3.0
**Author:** Eliot Abramo
**Standard:** IEC TR 62380:2004

Reliability Calculator is a KiCad plugin that turns a schematic and block diagram into a reliability model. It estimates failure rate at the component level with IEC TR 62380, rolls those results up to the system level, and then gives you a practical way to inspect uncertainty, sensitivity, dominant contributors, and likely improvement paths.

The point is not to produce a mysterious single number. The point is to help you answer, with one tool, what the current design is expected to do, how much that answer depends on your assumptions, and what you should change first if the margin is not good enough.

From the first run, the plugin is meant to answer four practical questions:

- What is the predicted system FIT and mission reliability for this design?
- How uncertain is that estimate if key inputs vary within plausible bounds?
- Which sheets, components, and parameters are driving the result?
- What should the team change first if the design needs a better reliability margin?

The exported report is built around those same questions. It explains the current reliability estimate, shows the uncertainty and sensitivity context around that estimate, and highlights the most actionable contributors and target-closure work. For the detailed math, interpretation limits, and validation notes, see [Methodology and Validation](./docs/METHODOLOGY.md).

## What Ships Today

- IEC TR 62380-based component failure-rate calculations
- Block-diagram system modeling with series, parallel, and k-of-n redundancy
- Monte Carlo uncertainty analysis with shared and independent parameter sampling
- Tornado sensitivity analysis for local design-point screening
- Component-level criticality based on elasticity
- Design-margin and what-if scenario analysis
- Reliability budget allocation, derating guidance, component swap analysis, and design history tracking
- HTML, Markdown, CSV, and JSON report export
- PDF export when an optional PDF backend is installed

## What This Tool Is For

Use the plugin when you want to answer questions like:

- What is the predicted system FIT and mission reliability for this design?
- Which sheet, component, or parameter is driving the estimate?
- How sensitive is the current design point to thermal, electrical, or usage assumptions?
- How wide is the uncertainty band if key inputs vary within plausible bounds?
- What design actions appear most promising before making a board spin?

## What This Tool Does Not Claim

- It is not a substitute for qualification testing, field return analysis, or independent safety review.
- It does not claim certification by itself.
- The exposed sensitivity workflow is not a production Sobol workflow. The shipped analysis surfaces local OAT screening, elasticity, and Monte Carlo-based SRRC ranking.
- PDF export is not guaranteed unless an optional dependency is installed.

## Installation

1. Install the plugin into KiCad using your normal plugin workflow or a local checkout.
2. Open a KiCad project and launch **Reliability Calculator** from the PCB editor action plugins menu.
3. If you need PDF export, install one of the optional backends:
   - `weasyprint`
   - `reportlab`

## Typical Workflow

1. Open the project and import the schematic-derived component set.
2. Classify parts and fill in the IEC parameters that matter for your design.
3. Set mission duration, thermal cycling, delta-T, and duty-cycle assumptions.
4. Build the system block diagram so the analysis matches the intended architecture.
5. Run the overview to identify dominant FIT contributors.
6. Run Monte Carlo to quantify uncertainty and inspect the confidence interval.
7. Run tornado and criticality analysis to identify leverage points.
8. Explore what-if scenarios, budgets, derating, and swap recommendations.
9. Export a report for design review or downstream documentation.

## Report Outputs

- **HTML:** best default output, includes styled tables and charts
- **PDF:** available when `weasyprint` or `reportlab` is installed
- **Markdown:** lightweight text export for engineering notes and repos
- **CSV:** component table export
- **JSON:** structured data export for pipelines or custom tooling

## Assumptions And Limits

- The plugin assumes IEC TR 62380-style constant failure-rate modeling for the reported mission interval.
- Monte Carlo results are only as credible as the parameter bounds and distributions you enter.
- Tornado and criticality views are local sensitivity tools around the current design point.
- SRRC is used as a monotonic importance ranking aid; it is not presented as a validated Sobol replacement.
- Independent verification is still required for high-consequence release decisions.

## Validation And Methodology

The user-facing README is intentionally concise. The mathematical notes, interpretation limits, and validation guidance live in:

- [Methodology and Validation](./docs/METHODOLOGY.md)

## Repository Quality Gates

This repository includes lightweight publish-readiness checks:

- automated unit and smoke tests for the core analysis modules
- regression checks for public claims and version consistency
- CI to run compile checks, tests, and linting on every push

## Release Notes

Version 3.3.0 focuses on publish-readiness hardening:

- public claims aligned with currently shipped features
- corrected Jensen-diagnostic language
- clearer separation between OAT, elasticity, SRRC, and internal Sobol code
- explicit PDF dependency messaging
- added tests and CI for core analysis and report export paths

## License

MIT. See [LICENSE](./LICENSE).
