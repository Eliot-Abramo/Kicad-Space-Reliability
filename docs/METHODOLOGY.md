# Methodology And Validation Notes

## Purpose

This document explains the analysis methods that are surfaced as production
features in the plugin, how those methods should be interpreted, and where the
tool deliberately stops short of making stronger claims.

The design goal is not mathematical theatrics.
The design goal is disciplined, auditable engineering communication.

For equation-level notes tied directly to the implementation, see
[Mathematics Reference](./MATHEMATICS.md).
For the module map and data-flow story, see
[Architecture Overview](./ARCHITECTURE.md).

## Public-Surface Philosophy

This repository contains both production analysis paths and a small amount of
exploratory or internal helper code. The documentation intentionally describes
what is actually shipped to users, not every experiment that has ever existed
in the codebase.

## End-To-End Analytical Pipeline

The production workflow is best understood as a sequence of transformations:

1. **Schematic-derived design state**  
   Components are imported from the KiCad project and organized by sheet.

2. **Component classification and parameterization**  
   Each part is mapped to an IEC TR 62380 component family and given the stress
   parameters that drive its failure-rate model.

3. **Component-level lambda evaluation**  
   The relevant IEC-style temperature, voltage, thermal-cycling, package, and
   usage terms are evaluated to obtain a component failure rate.

4. **Sheet-level aggregation**  
   Component lambdas are summed into sheet totals so the design can be analyzed
   both structurally and hierarchically.

5. **System-topology composition**  
   Sheet-level results are mapped into a user-defined block diagram and
   combined according to series, parallel, or k-of-n logic.

6. **Derived mission metrics**  
   The tool computes mission reliability, MTTF, system FIT, and related
   rollups from the resulting system hazard model.

7. **Uncertainty and leverage analyses**  
   Monte Carlo, SRRC ranking, tornado sensitivity, and component criticality
   are used to understand spread and local design leverage.

8. **Target-closure analyses**  
   Design-margin scenarios, budget allocation, derating guidance, component
   swap analysis, and growth tracking help turn results into action.

That last step is one of the unusual aspects of the project: the plugin is not
only trying to estimate reliability, it is trying to help improve it.

## Shipped Production Methods

### Core Reliability Model

The core numerical model uses IEC TR 62380 component stress equations to
compute part-level lambdas and then aggregates them upward.

- For a plain series architecture, lambdas sum.
- For explicit block diagrams, leaf reliabilities are composed in reliability
  space according to series, parallel, and k-of-n structure.
- Mission reliability is then evaluated over the user-specified interval.

Interpretation:

- This is a constant-hazard mission model over the stated interval.
- The quality of the output depends heavily on correct component
  classification, realistic mission assumptions, and correct block topology.

### Mission Phasing

The plugin supports both simple single-phase missions and multi-phase mission
profiles.

Each phase can override:

- ambient or junction temperature
- duty cycle
- annual thermal cycles
- thermal excursion
- duration fraction within the total mission

In the production path, the tool recomputes the full component model for each
phase and then time-weights those phase results.

Interpretation:

- This is a practical phased-use model, not an attempt to infer missing field
  data from mission descriptions.
- Phase definitions should be physically plausible and should sum to a
  meaningful total mission profile.

### Monte Carlo Uncertainty Propagation

Monte Carlo analysis propagates user-entered uncertainty through the full
component formulas and then through the system model.

The implementation supports:

- bounded uniform sampling
- bounded PERT-style sampling
- shared-parameter sampling for assumptions that should move coherently
- independent per-component sampling where that is more appropriate

The tool reports empirical distributions for reliability and failure rate,
along with confidence intervals and summary statistics.

Interpretation:

- This is uncertainty propagation, not proof that the model is valid.
- The outputs are only as credible as the bounds, distributions, and parameter
  ownership rules supplied by the user.

### SRRC Importance Ranking

After Monte Carlo sampling, the plugin computes Standardized Rank Regression
Coefficients to rank monotonic importance.

Interpretation:

- SRRC is used as a screening and ranking aid.
- Normalized `SRRC^2` is reported as a relative importance share.
- This should not be read as a claim of full variance decomposition.

### Tornado Sensitivity

The tornado chart is a deterministic One-At-a-Time local sensitivity tool.

- One parameter is perturbed around the current design point.
- The affected formulas are recomputed.
- The resulting system FIT swing is recorded and ranked.

Interpretation:

- Good for asking which assumptions move the present design most.
- Good for design-review prioritization.
- Not a global sensitivity analysis.

### Component Criticality

The criticality view is based on elasticity, the local log-log derivative of
component failure rate with respect to a parameter.

Interpretation:

- The sign indicates direction of influence.
- The magnitude indicates local leverage near the current design point.
- This is more granular than a tornado chart because it operates at the
  component-parameter level rather than system-wide one-factor perturbations.

### Design-Margin Scenarios

Scenario analysis evaluates predefined or user-defined shifts away from the
nominal operating point to estimate how much margin the current design may have
against plausible stress changes.

Interpretation:

- Scenario outputs are comparative planning tools.
- They are useful for communicating margin and fragility, but they do not turn
  a handbook-based model into a field-validated prediction.

### Reliability Budget Allocation

Budget allocation converts a target system reliability into an allowable FIT
budget and then apportions that budget across the design.

The shipped implementation includes:

- equal apportionment
- proportional allocation
- complexity-weighted allocation
- criticality-weighted allocation

Interpretation:

- Budgeting is a closure-planning tool, not a theorem about the "correct"
  share each component deserves.
- It is most useful for identifying over-budget contributors and quantifying
  how much FIT must be recovered.

### Derating Guidance

The derating engine works backwards from a target component failure rate and
searches for a parameter value that would close the gap.

Interpretation:

- Recommendations come from repeated evaluation of the nonlinear model, not
  from a simple first-order approximation.
- Output should be treated as model-conditioned guidance for engineering review.

### Component Swap Analysis

Swap analysis clones a component configuration, changes one design choice
(package, subtype, technology, and similar alternatives), and ranks the
resulting FIT delta.

Interpretation:

- This is a targeted what-if engine.
- It is particularly useful for identifying high-leverage substitutions before
  a board spin or procurement decision.

### Reliability Growth Tracking

The plugin can save point-in-time snapshots, compare revisions, and attribute
improvements or degradations to specific component changes.

Interpretation:

- Growth tracking is a design-management aid.
- It does not replace change review, but it gives teams a quantitative memory
  of whether the system is actually moving toward its target.

## Jensen Diagnostic

For the reliability function

`R(t) = exp(-lambda * t)`

the second derivative with respect to `lambda` is positive for `t > 0`, so the
function is convex in `lambda`. Jensen's inequality therefore gives:

`E[R(t)] >= R(E[lambda], t)`

The plugin uses this as a consistency diagnostic on Monte Carlo output:

- it compares the sample mean of reliability to reliability evaluated at the
  mean sampled lambda
- it treats violations beyond numerical tolerance as a warning to inspect
  sampling noise, implementation consistency, or baseline handling

This diagnostic is intentionally modest.
It is not used to claim that the nominal design point is a formal bound.

## Interpretation Rules That Matter In Practice

- **Topology matters.** Once redundancy enters the picture, reliability must be
  composed in reliability space, not by blindly summing FIT values.
- **Temperature is not linear.** Many IEC stress factors are exponential or
  strong power laws, so small assumption changes can create large output swings.
- **Uncertainty is input-dependent.** Wide confidence bands usually say as much
  about parameter ambiguity as they do about the design itself.
- **Tornado and criticality are local tools.** They rank leverage around the
  current design point; they do not map the entire response surface.
- **Budget, derating, and swap outputs are conditional recommendations.** They
  are only as trustworthy as the underlying model and parameterization.

## Recommended Validation Practice

Before relying on a result in a serious review:

1. Confirm component classification and key stress parameters manually.
2. Check mission-profile assumptions for physical plausibility.
3. Verify that the block diagram actually matches the intended architecture.
4. Review uncertainty bounds and shared-parameter choices for defensibility.
5. Treat HTML export as the primary review artifact; use PDF only when the
   optional backend is installed and verified in your environment.
6. Independently review high-consequence conclusions before using them in
   customer-facing, safety-facing, or certification-facing material.

## Automated Validation In This Repository

This repository includes automated checks for:

- analysis-core behavior
- tornado and criticality regression coverage
- report export smoke paths
- public-claim regression checks
- version consistency across release-facing files

These checks do not prove the model is universally correct, but they do help
keep the implementation, the release surface, and the documentation aligned.
