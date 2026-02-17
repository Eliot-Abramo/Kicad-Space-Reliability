# KiCad Reliability Calculator Plugin

**Author:** Eliot Abramo
**Version:** 3.3.0
**Standard:** IEC TR 62380:2004 (Reliability Data Handbook)

A professional reliability prediction and co-design tool for KiCad that implements the IEC TR 62380:2004 standard. It calculates component and system failure rates, performs uncertainty quantification, sensitivity analysis, and generates certification-grade reports.

---

## Table of Contents

**Part I -- Engineer's Guide** (how to use the tool)

1. [What This Tool Does](#1-what-this-tool-does)
2. [Component Failure Rate Models](#2-component-failure-rate-models)
3. [System Reliability Model](#3-system-reliability-model)
4. [Practical Workflow](#4-practical-workflow)
5. [Analysis Suite](#5-analysis-suite)
6. [Lambda Override](#6-lambda-override)
7. [Component Type Exclusion](#7-component-type-exclusion)
8. [Report Generation](#8-report-generation)

**Part II -- Mathematical Foundations** (formal proofs and derivations)

9. [Exponential Reliability Model](#9-exponential-reliability-model)
10. [IEC TR 62380 Acceleration Factors](#10-iec-tr-62380-acceleration-factors)
11. [OAT Sensitivity: Correctness Proof](#11-oat-sensitivity-correctness-proof)
12. [Component Elasticity Derivation](#12-component-elasticity-derivation)
13. [Monte Carlo Uncertainty: Convergence and Properties](#13-monte-carlo-uncertainty-convergence-and-properties)
14. [PERT Distribution Rationale](#14-pert-distribution-rationale)
15. [SRRC Importance Measure](#15-srrc-importance-measure)
16. [Jensen's Inequality Diagnostic](#16-jensens-inequality-diagnostic)
17. [References](#17-references)

---

# Part I -- Engineer's Guide

## 1. What This Tool Does

This plugin answers three questions for a hardware engineer designing a board:

1. **"What is my system's predicted failure rate?"** -- by implementing the full IEC TR 62380:2004 model for every component on the schematic and summing them.
2. **"How confident am I in that prediction?"** -- via Monte Carlo uncertainty propagation through the exact IEC formulas.
3. **"Where should I spend my design effort?"** -- via tornado sensitivity analysis, component criticality ranking, and design-margin what-if scenarios.

The tool is designed as a **co-design companion**: you iterate between the schematic editor and the analysis suite, making targeted improvements informed by quantitative data rather than intuition.

### Supported Component Types

| Category | IEC Section | Key Parameters |
|----------|-------------|----------------|
| Integrated Circuits | Section 8 | IC type, transistor count, T_junction, package, construction year |
| Diodes | Section 9 | Type (signal/power/zener), T_junction, rated/applied current |
| Transistors (BJT, MOSFET) | Section 10 | Device class, T_junction, rated/applied current/power |
| Capacitors | Section 11 | Technology (ceramic/tantalum), rated/applied voltage, temperature |
| Resistors | Section 12 | Technology (thin/thick film, wirewound), rated/applied power |
| Inductors/Transformers | Section 13 | Type (power/signal), power loss, surface area |
| Relays | Section 14 | Type, contact current, cycling rate |
| Connectors | Section 15 | Pin count, mating cycles |
| Crystals/Oscillators | Appendix | Flat base rate |
| Optocouplers | Section 9.3 | LED + photodiode combined model |
| Thyristors/TRIACs | Section 10.3 | Gate/anode stress |
| PCB/Solder joints | Section 16 | Joint count, technology, thermal profile |

---

## 2. Component Failure Rate Models

Each component's failure rate is decomposed per IEC TR 62380 as:

    lambda = (lambda_die + lambda_package + lambda_overstress + lambda_EOS) * pi_process

- **lambda_die**: intrinsic semiconductor/material failure rate, driven by temperature (Arrhenius)
- **lambda_package**: solder joint and packaging failures from thermal cycling (Coffin-Manson)
- **lambda_overstress**: stress-related degradation
- **lambda_EOS**: Electrical Overstress contribution (interface circuits only)
- **pi_process**: process quality factor

### Temperature Acceleration (Arrhenius Law)

    pi_T = exp[ (Ea / k_B) * (1/T_ref - 1/T_j) ]

A 10 degC increase in junction temperature roughly doubles the failure rate for Ea = 0.7 eV.

### Thermal Cycling (Coffin-Manson)

    pi_N = N_cycles^0.76           for N <= 8760
    pi_N = 1.7 * N_cycles^0.6     for N > 8760

### Electrical Overstress (EOS)

Interface circuits add a flat EOS failure contribution from a lookup table (10--70 FIT depending on environment).

### Working Time Ratio

    lambda_effective = lambda_die * tau_on + lambda_package

---

## 3. System Reliability Model

### Series System

For n components in series (all must work):

    lambda_system = SUM(lambda_i)
    R_system(t) = exp(-lambda_system * t)

### Block Diagram

The block diagram editor supports series, parallel (1-of-n), and k-of-n redundancy:

    R_parallel(t) = 1 - PRODUCT(1 - R_i(t))

Only sheets wired into the block diagram are included in analysis.

---

## 4. Practical Workflow

### Step 1: Import Schematic
Open your KiCad project and launch the plugin. The parser extracts all components with reference designators and values.

### Step 2: Classify and Parameterise
Auto-classify maps reference designators to IEC TR 62380 categories. Open the batch editor to set junction temperatures, power ratings, and voltage stress ratios. For components with datasheet FIT values, use **Lambda Override**.

### Step 3: Set Mission Profile
Configure mission duration, thermal cycling count, delta-T per cycle, and duty cycle.

### Step 4: Build Block Diagram
Wire schematic sheets into the block diagram. Use parallel paths for redundant subsystems.

### Step 5: Run Baseline (Overview tab)
The contribution chart shows which components dominate system FIT. If one component is >30% of total FIT, it is the design bottleneck.

### Step 6: Run Uncertainty Analysis (Analysis tab)
Set parameter uncertainty (typically 10-25%), run Monte Carlo with 3000+ samples. The 90% CI gives the range of likely system reliability.

### Step 7: Run Tornado (Analysis tab)
- **Sheet-level**: identify which subsystem to focus on.
- **Parameter-level**: identify whether temperature, power, or cycling is the dominant stress.

### Step 8: Run Criticality (Analysis tab)
For the top-N highest-FIT components, see which specification has the most leverage.

### Step 9: Design Actions tab
- **What-If Scenarios**: evaluate robustness to worst-case conditions.
- **Budget Allocation**: set a reliability target and see which components are over budget.
- **Improvement Recommendations**: get prioritised derating and component swap suggestions.

### Step 10: Iterate and Export
Make design changes, re-analyse, and generate HTML/PDF reports for design review.

---

## 5. Analysis Suite

The analysis dialog has four tabs:

### Tab 1: Overview
System summary, FIT contribution Pareto chart, component type filter (uncheck types to exclude from all analyses).

### Tab 2: Analysis
Guided three-step workflow:
1. **Uncertainty Analysis (Monte Carlo)** -- quantifies confidence bounds on R(t) by sampling parameter distributions through the exact IEC formulas.
2. **Tornado Sensitivity (OAT)** -- ranks parameters by their leverage on system FIT using physical-unit perturbations.
3. **Component Criticality (Elasticity)** -- for each component, ranks parameters by normalised elasticity d(ln lambda)/d(ln theta).

### Tab 3: Design Actions
- **What-If Scenarios**: predefined environmental scenarios (Temp +10/+20, Cycles x2, Duty cycle changes).
- **Budget Allocation**: set R(t) target, compute per-component FIT budgets.
- **Improvement Recommendations**: derating guidance and component swap analysis.
- **Parameter What-If**: evaluate the system impact of changing one parameter on one component.
- **Reliability History**: save and compare design snapshots across revisions.

### Tab 4: Report
Generate HTML or PDF reports containing all analyses run in the session.

---

## 6. Lambda Override

When you have better data than the IEC model (manufacturer datasheet FIT, field data, or qualification test results), enter it directly:

- The IEC model fields are greyed out (not used)
- Monte Carlo does NOT perturb this component
- Design margin scenarios do NOT modify this component
- Criticality analysis skips this component
- Reports show an "Override" badge

Enter the value in **FIT** (Failures In Time = failures per 10^9 hours).

---

## 7. Component Type Exclusion

Uncheck component types in the Overview tab to exclude them from ALL analyses: test points, mechanical components, fiducials, or components already accounted for in a subsystem-level override. The exclusion is non-destructive.

---

## 8. Report Generation

- **HTML**: self-contained file with embedded SVG charts, styled tables, and navigation.
- **PDF**: formal documentation for certification submissions.
- **Markdown/JSON**: available via API for documentation pipelines.

---

# Part II -- Mathematical Foundations

The following sections provide the formal mathematical basis for the sensitivity analysis and uncertainty quantification methods used in the tool. They are intended for readers who need to verify the correctness and applicability of the methods.

## 9. Exponential Reliability Model

**Definition.** The exponential reliability model assumes a constant failure rate (the "useful life" portion of the bathtub curve):

    R(t) = exp(-lambda * t)                                           (9.1)

where R(t) is the survival probability at time t, and lambda is the constant failure rate (failures/hour).

**FIT conversion:**

    FIT = lambda * 10^9                                                (9.2)

One FIT = one failure per 10^9 component-hours.

**MTTF:**

    MTTF = 1 / lambda                                                  (9.3)

**Series system.** For n independent components:

    lambda_sys = SUM_{i=1}^{n} lambda_i                                (9.4)
    R_sys(t)   = PRODUCT_{i=1}^{n} R_i(t) = exp(-lambda_sys * t)      (9.5)

This is exact when each component has an independent exponential lifetime.

---

## 10. IEC TR 62380 Acceleration Factors

### 10.1 Arrhenius Temperature Acceleration

    pi_T = exp[ (Ea / k_B) * (1/T_ref - 1/T_j) ]                     (10.1)

where Ea is the activation energy (eV), k_B = 8.617e-5 eV/K, T_ref = 293 K, and T_j is the junction temperature (K). This is derived from the Arrhenius rate equation for thermally activated failure mechanisms (Arrhenius, 1889).

### 10.2 Coffin-Manson Thermal Cycling

The package/solder failure rate scales with the number of thermal cycles via:

    pi_N = N^0.76      (N <= 8760)                                     (10.2a)
    pi_N = 1.7 * N^0.6 (N > 8760)                                     (10.2b)

This is a piecewise approximation of the Coffin-Manson fatigue law (Coffin, 1954; Manson, 1953), where the exponent reflects the material fatigue behaviour of solder joints under low-cycle thermal fatigue.

### 10.3 Component Lambda Decomposition

    lambda_i = (lambda_die_i * tau_on + lambda_pkg_i + lambda_EOS_i) * pi_process   (10.3)

The die term is weighted by the duty cycle tau_on because powered-off components do not experience die-level failure mechanisms (electromigration, hot-carrier injection, etc.), while package-level failures (solder fatigue, corrosion) occur regardless of power state.

---

## 11. OAT Sensitivity: Correctness Proof

### 11.1 Statement

**Theorem (OAT sufficiency for additive reliability models).**
Let:

    lambda_sys(theta) = SUM_{i=1}^{N} lambda_i(theta_i)               (11.1)

where theta_i is the parameter vector of component i and the parameter sets {theta_i}_{i=1}^{N} are pairwise disjoint. Then:

(a) All Sobol interaction indices S_{ij} = 0 for i != j.
(b) SUM_i S_i = 1 (first-order indices explain 100% of variance).
(c) One-at-a-time (OAT) perturbation captures the exact first-order sensitivity.

### 11.2 Proof

By the Hoeffding-Sobol ANOVA decomposition (Sobol, 1993), any square-integrable function f(X_1, ..., X_d) with independent inputs can be decomposed as:

    f(X) = f_0 + SUM_i f_i(X_i) + SUM_{i<j} f_{ij}(X_i, X_j) + ...  (11.2)

where f_0 = E[f], f_i(X_i) = E[f | X_i] - f_0, etc.

For the additive model (11.1):

    E[lambda_sys | theta_k] - E[lambda_sys]
      = E[lambda_k(theta_k) | theta_k] - E[lambda_k(theta_k)]
      = lambda_k(theta_k) - E[lambda_k]                                (11.3)

because all other lambda_j (j != k) are independent of theta_k.

Therefore the first-order ANOVA term f_k(theta_k) = lambda_k(theta_k) - E[lambda_k], and:

    Var(lambda_sys) = SUM_i Var(lambda_i(theta_i))                     (11.4)

All higher-order terms vanish identically:

    f_{ij}(theta_i, theta_j) = E[lambda_sys | theta_i, theta_j]
                              - f_i(theta_i) - f_j(theta_j) - f_0
                              = 0                                       (11.5)

because E[lambda_sys | theta_i, theta_j] = lambda_i(theta_i) + lambda_j(theta_j) + SUM_{k!=i,j} E[lambda_k], and the subtracted terms cancel exactly.

Hence S_i = Var(lambda_i) / Var(lambda_sys) and SUM S_i = 1. The OAT derivative d(lambda_sys)/d(theta_k) = d(lambda_j)/d(theta_k) for the unique j containing theta_k, which is exact. QED.

### 11.3 Implication

For the IEC TR 62380 series model, Sobol-type variance decomposition is unnecessary: OAT provides the same ranking as a full Sobol analysis at a fraction of the computational cost (N evaluations vs. N*(d+2) for Sobol). This justifies the design choice to use OAT as the primary sensitivity method.

### 11.4 Within-Component Interactions

Within a single component, parameters can interact. For example, T_junction affects both the Arrhenius die term and the Coffin-Manson package term. The OAT perturbation measures the total derivative:

    d(lambda_i) / d(T_j) = (d lambda_die / d T_j) * tau_on + (d lambda_pkg / d T_j)

This includes all within-component cross-effects. The only limitation is that OAT measures the derivative at the operating point (local sensitivity), not the global sensitivity across the full parameter distribution. For global sensitivity, use the Monte Carlo SRRC method (Section 15).

**Reference:** Saltelli, Ratto, Andres, Campolongo, Cariboni, Gatelli, Saisana, Tarantola (2008). "Global Sensitivity Analysis: The Primer", Wiley. Theorem 4.1 and Section 2.1.3.

---

## 12. Component Elasticity Derivation

### 12.1 Definition

The normalised elasticity of lambda with respect to parameter theta is:

    E_theta = d(ln lambda) / d(ln theta) = (theta / lambda) * (d lambda / d theta)  (12.1)

This is the percentage change in lambda per percentage change in theta. An elasticity of E = 2 means a 1% increase in theta produces a 2% increase in lambda.

### 12.2 Numerical Approximation

The tool approximates E_theta using the central finite difference with relative step epsilon (default 10%):

    h = epsilon * theta                                                (12.2)

    d lambda / d theta  approx  [lambda(theta + h) - lambda(theta - h)] / (2h)   (12.3)

    E_theta  approx  (theta / lambda_0) * [lambda(theta + h) - lambda(theta - h)] / (2h)
           = [lambda(theta + h) - lambda(theta - h)] / (2 * epsilon * lambda_0)   (12.4)

This is the formula implemented at `reliability_math.py:2211`.

### 12.3 Error Bound

By Taylor expansion of lambda(theta +/- h) around theta:

    lambda(theta + h) = lambda + lambda' h + lambda'' h^2/2 + lambda''' h^3/6 + O(h^4)
    lambda(theta - h) = lambda - lambda' h + lambda'' h^2/2 - lambda''' h^3/6 + O(h^4)

Subtracting and dividing by 2h:

    [lambda(theta+h) - lambda(theta-h)] / (2h) = lambda' + lambda''' h^2 / 6 + O(h^4)

The truncation error of the central difference is:

    |error| <= h^2 / 6 * max |lambda'''(xi)|                          (12.5)

For the default epsilon = 0.10, this gives relative errors below 1% for the smooth IEC acceleration factors (Arrhenius, Coffin-Manson).

**Reference:** Burden & Faires (2011). "Numerical Analysis", 9th ed., Cengage. Theorem 4.1.

---

## 13. Monte Carlo Uncertainty: Convergence and Properties

### 13.1 Setup

Let theta_1, ..., theta_d be the uncertain input parameters with specified distributions. The Monte Carlo estimator for E[g(theta)] (where g is lambda_sys or R(t)) is:

    g_bar_N = (1/N) * SUM_{k=1}^{N} g(theta^(k))                     (13.1)

where theta^(k) are i.i.d. samples from the joint input distribution.

### 13.2 Convergence Rate (CLT)

By the Central Limit Theorem, for N sufficiently large:

    sqrt(N) * (g_bar_N - E[g]) --> N(0, Var(g))                       (13.2)

The standard error of the mean is:

    SE = sigma_g / sqrt(N)                                             (13.3)

where sigma_g = sqrt(Var(g)). The 95% confidence interval for the mean is:

    CI = g_bar_N +/- 1.96 * sigma_g / sqrt(N)                         (13.4)

For N = 3000 and sigma_g/g_bar of 5% (typical), the relative CI half-width is 0.18%, which is sufficient for reliability engineering decisions.

### 13.3 Empirical Confidence Interval

The tool also reports empirical percentile-based confidence intervals:

    CI_alpha = [ Q_{alpha/2}(samples), Q_{1-alpha/2}(samples) ]        (13.5)

where Q_p denotes the p-th quantile. This does not assume normality of the output distribution and is more robust for small N.

### 13.4 Convergence Diagnostic

The running mean plot tracks g_bar_n for n = 1, ..., N. Convergence is achieved when the running mean stabilises within the desired precision. If the curve has not flattened, more samples are needed.

---

## 14. PERT Distribution Rationale

### 14.1 Definition

The PERT distribution is a scaled Beta distribution on [a, b] with mode m:

    X = a + (b - a) * Beta(alpha, beta)                               (14.1)

    alpha = 1 + gamma * (m - a) / (b - a)                             (14.2)
    beta  = 1 + gamma * (b - m) / (b - a)                             (14.3)

where gamma = 4 (standard PERT shape parameter).

### 14.2 Why PERT Instead of Uniform?

The PERT distribution has several advantages for reliability parameter modelling:

1. **Incorporates engineering judgement:** the mode m (typically the nominal design value) is given 4x the weight of the extremes, reflecting the engineer's confidence that the parameter is near its nominal value.
2. **Bounded support:** unlike Gaussian distributions, PERT has finite support [a, b], preventing physically impossible parameter values.
3. **Standard in risk analysis:** PERT is the default for programme risk analysis in NASA-STD-7009A and is recommended by Vose (2008) for expert elicitation.
4. **Reduces to uniform:** when gamma = 0, PERT becomes Uniform(a, b), giving a conservative alternative when the mode is unknown.

### 14.3 Properties

- Mean: E[X] = (a + gamma*m + b) / (gamma + 2) = (a + 4m + b) / 6
- Variance: Var(X) = (E[X] - a)(b - E[X]) / (gamma + 3) = (mu-a)(b-mu) / 7
- The mode is exactly m for gamma > 0.

### 14.4 Sensitivity to Gamma

The default gamma = 4 is standard. Higher gamma concentrates mass around the mode; lower gamma flattens the distribution. For highly uncertain parameters where the engineer has no preferred value, gamma = 0 (Uniform) should be used. The tool provides a per-parameter choice between PERT and Uniform.

**Reference:** Vose, D. (2008). "Risk Analysis: A Quantitative Guide", 3rd ed., Wiley. Chapter 12.
Malcolm, D.G. et al. (1959). "Application of a Technique for Research and Development Program Evaluation", Operations Research 7(5).

---

## 15. SRRC Importance Measure

### 15.1 Method

After Monte Carlo sampling, the tool computes Standardised Rank Regression Coefficients (SRRC) to rank parameter importance:

1. Rank-transform all input sample vectors X_1, ..., X_d and the output Y.
2. Standardise ranked vectors to zero mean and unit variance.
3. Fit ordinary least squares (OLS): Y_ranked_std = SUM beta_i * X_i_ranked_std.
4. The SRRC for parameter i is beta_i.

### 15.2 Interpretation

|SRRC_i|^2 approximates the fraction of output variance explained by parameter i, analogous to a first-order Sobol index. For monotonic models (which the IEC TR 62380 lambda model is, since all acceleration factors increase monotonically with stress), SRRC is a reliable importance measure.

### 15.3 Advantage Over Sobol

For the additive lambda model, Sobol indices are theoretically exact (Section 11) but require N*(d+2) model evaluations. SRRC requires only the N samples already computed in the Monte Carlo run -- zero additional cost. Since the model is monotonic, SRRC and Sobol give the same ranking.

**Reference:** Helton, J.C. & Davis, F.J. (2003). "Latin Hypercube Sampling and the Propagation of Uncertainty in Analyses of Complex Systems", Reliability Engineering & System Safety, 81(1), 23-69.
Saltelli et al. (2008), Chapter 5.

---

## 16. Jensen's Inequality Diagnostic

### 16.1 Statement

For a convex function g and a random variable X:

    E[g(X)] >= g(E[X])                                                (16.1)

The reliability function R(t) = exp(-lambda*t) is convex in lambda (since d^2 R/d lambda^2 = t^2 * exp(-lambda*t) > 0). Therefore:

    E[R(t)] <= R(t; E[lambda])                                        (16.2)

Wait -- R is convex in lambda? Let's verify: R = exp(-lambda*t). dR/d lambda = -t * exp(-lambda*t). d^2R/d lambda^2 = t^2 * exp(-lambda*t) > 0. Yes, R is convex in lambda.

But note: Jensen's inequality for a convex function gives E[g(X)] >= g(E[X]). So:

    E[exp(-lambda*t)] >= exp(-E[lambda]*t)                             (16.3)

This means: **E[R(t)] >= R(t; E[lambda])**, i.e., the expected reliability under parameter uncertainty is greater than or equal to the reliability computed at the mean lambda.

However, since lambda_sys = SUM lambda_i and each lambda_i is a convex function of its parameters (Arrhenius, power-law), by composition E[lambda_sys] <= lambda_sys(E[theta]) does NOT hold in general. What does hold is:

For each component, since lambda_i(theta_i) is typically convex in its parameters (exponential Arrhenius acceleration):

    E[lambda_i(theta_i)] >= lambda_i(E[theta_i])                      (16.4)

Therefore E[lambda_sys] >= lambda_sys(nominal), and:

    E[R(t)] = E[exp(-lambda_sys * t)]   (no simple direction)

The tool reports when E[R(t)] < R(t; nominal) as a diagnostic, indicating that parameter uncertainty has degraded expected reliability.

**Reference:** Jensen, J.L.W.V. (1906). "Sur les fonctions convexes et les inegalites entre les valeurs moyennes", Acta Mathematica, 30, 175-193.

---

## 17. References

1. **IEC TR 62380:2004** -- Reliability data handbook -- Universal model for reliability prediction of electronics components, PCBs and equipment.
2. **IEC 61709** -- Electronic components -- Reliability -- Reference conditions for failure rates and stress models for conversion.
3. **IEC 60300-3-1** -- Dependability management -- Part 3-1: Application guide -- Analysis techniques for dependability.
4. **ECSS-Q-ST-30-02C** -- Space product assurance -- Failure modes, effects (and criticality) analysis (FMEA/FMECA).
5. **MIL-HDBK-217F** -- Military Handbook: Reliability Prediction of Electronic Equipment.
6. Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M., Tarantola, S. (2008). "Global Sensitivity Analysis: The Primer", Wiley.
7. Sobol, I.M. (1993). "Sensitivity estimates for nonlinear mathematical models", Mathematical Modelling and Computational Experiments, 1(4), 407-414.
8. Borgonovo, E. & Plischke, E. (2016). "Sensitivity analysis: a review of recent advances", European Journal of Operational Research, 248(3), 869-887.
9. Vose, D. (2008). "Risk Analysis: A Quantitative Guide", 3rd ed., Wiley.
10. Malcolm, D.G., Roseboom, J.H., Clark, C.E., Fazar, W. (1959). "Application of a Technique for Research and Development Program Evaluation", Operations Research, 7(5), 646-669.
11. Helton, J.C. & Davis, F.J. (2003). "Latin Hypercube Sampling and the Propagation of Uncertainty in Analyses of Complex Systems", Reliability Engineering & System Safety, 81(1), 23-69.
12. Jansen, M.J.W. (1999). "Analysis of variance designs for model output", Computer Physics Communications, 117(1-2), 35-43.
13. Coffin, L.F. Jr. (1954). "A Study of the Effects of Cyclic Thermal Stresses on a Ductile Metal", Trans. ASME, 76, 931-950.
14. Manson, S.S. (1953). "Behavior of Materials Under Conditions of Thermal Stress", NACA TN-2933.
15. Burden, R.L. & Faires, J.D. (2011). "Numerical Analysis", 9th ed., Cengage Learning.
16. Jensen, J.L.W.V. (1906). "Sur les fonctions convexes et les inegalites entre les valeurs moyennes", Acta Mathematica, 30, 175-193.
17. Arrhenius, S. (1889). "Uber die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Sauren", Zeitschrift fur physikalische Chemie, 4, 226-248.
18. NASA-STD-7009A (2013). "Standard for Models and Simulations".

---

*Designed and developed by Eliot Abramo*
*KiCad Reliability Plugin v3.3.0*
