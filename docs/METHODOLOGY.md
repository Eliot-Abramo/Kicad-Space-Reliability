# Methodology And Validation Notes

## Purpose

This document explains the analysis methods that are currently surfaced as production features in the plugin and, just as importantly, how they should be interpreted.

The goal is defensible engineering communication, not exaggerated mathematical claims.

## Shipped Analysis Methods

### Core Reliability Model

The plugin computes component failure rate using IEC TR 62380 stress models and aggregates component lambdas into sheet and system-level results. For series structures, system lambda is the sum of component lambdas. Block modeling extends this with parallel and k-of-n reliability composition.

### Tornado Sensitivity

The tornado chart is a local deterministic sensitivity tool.

- One parameter is perturbed at a time around the current design point.
- The affected IEC formulas are re-evaluated.
- The output is the resulting swing in system FIT.

Interpretation:

- Good for engineering prioritisation.
- Good for asking "what moves the current estimate the most?"
- Not a global variance decomposition.
- Not a Sobol index.

### Component Criticality

Criticality uses elasticity, the local log-log derivative:

`d(ln lambda) / d(ln theta)`

Interpretation:

- Positive elasticity means increasing the parameter tends to increase lambda.
- Negative elasticity means increasing the parameter tends to decrease lambda.
- The magnitude indicates local leverage around the current operating point.

### Monte Carlo Uncertainty

Monte Carlo propagates user-specified parameter uncertainty through the full IEC formulas.

- Shared parameters are sampled once per simulation and applied consistently across affected components.
- Independent parameters are sampled per component.
- The plugin reports empirical distributions for lambda and reliability, plus confidence intervals.

Interpretation:

- This is uncertainty propagation, not model validation.
- Output credibility depends on input bounds, distribution choices, and component parameter quality.

### SRRC Importance Ranking

The plugin computes Standardized Rank Regression Coefficients after Monte Carlo sampling.

Interpretation:

- SRRC is used as a monotonic importance ranking aid.
- `SRRC^2` is normalized into a relative importance share for reporting.
- The result is useful for screening and ranking, but it is not presented here as a validated Sobol substitute.

## Jensen Diagnostic

For the reliability function:

`R(t) = exp(-lambda * t)`

the second derivative with respect to lambda is positive for `t > 0`, so `R` is convex in lambda. Jensen's inequality therefore gives:

`E[R(t)] >= R(E[lambda], t)`

The plugin uses this relationship as a consistency check:

- It compares the Monte Carlo sample mean of reliability to the reliability evaluated at the mean sampled lambda.
- It does not interpret the nominal design-point reliability as a theorem-backed lower or upper bound.

If the reported mean reliability falls below `R(E[lambda], t)` by more than numerical tolerance, that should be treated as a warning to review sampling noise, implementation consistency, or baseline handling.

## What Is Not Shipped As A Production Claim

The repository still contains internal and exploratory Sobol-related code paths. Those are not part of the current production-facing plugin workflow and are not advertised as shipped features in the release-facing documentation.

Until a Sobol workflow is fully integrated, validated, and regression-tested end to end, it should be treated as internal or experimental.

## Recommended Validation Practice

Before relying on a release result:

1. Confirm component classification and key IEC inputs manually.
2. Review overrides and verify that their provenance is documented.
3. Check whether the chosen uncertainty bounds are physically defensible.
4. Re-run the analyses after major mission-profile or architecture changes.
5. Export HTML as the primary review artifact and use PDF only when the optional backend is installed and verified in your environment.
6. Independently review high-consequence conclusions before using them in customer, safety, or certification-facing material.

## Automated Checks In This Repository

The repository includes automated checks for:

- core Monte Carlo behavior
- tornado and criticality regression coverage
- report export smoke paths
- public-claim regression checks
- version consistency across release-facing files

These checks improve release confidence, but they do not replace external engineering verification.
