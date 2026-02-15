# KiCad Reliability Calculator Plugin

**Author:** Eliot Abramo
**Version:** 3.1.0
**Standard:** IEC TR 62380:2004 (Reliability Data Handbook)

A professional reliability prediction tool for KiCad that implements the IEC TR 62380:2004 standard. It calculates component and system failure rates, performs uncertainty quantification, sensitivity analysis, and generates certification-grade reports.

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Component Failure Rate Models](#2-component-failure-rate-models)
3. [System Reliability Model](#3-system-reliability-model)
4. [Monte Carlo Uncertainty Analysis](#4-monte-carlo-uncertainty-analysis)
5. [Tornado Sensitivity Analysis](#5-tornado-sensitivity-analysis)
6. [Design Margin Analysis](#6-design-margin-analysis)
7. [Component Criticality Analysis](#7-component-criticality-analysis)
8. [Lambda Override](#8-lambda-override)
9. [Component Type Exclusion](#9-component-type-exclusion)
10. [Report Generation](#10-report-generation)
11. [Practical Workflow](#11-practical-workflow-designing-a-reliable-board)
12. [References](#12-references)

---

## 1. Mathematical Foundation

### 1.1 Failure Rate and Reliability

The plugin is built on the exponential reliability model, which assumes a constant failure rate (the "flat" part of the bathtub curve). This is the standard assumption for electronic components during their useful life:

**Reliability function:**

    R(t) = exp(-lambda * t)

where:
- `R(t)` is the probability of survival at time `t` (dimensionless, 0 to 1)
- `lambda` is the constant failure rate (failures per hour)
- `t` is the mission time (hours)

**FIT (Failures In Time):**

    FIT = lambda * 10^9

One FIT equals one failure per 10^9 component-hours. Typical electronic components range from 1 to 100 FIT.

**MTTF (Mean Time To Failure):**

    MTTF = 1 / lambda

### 1.2 Series System Model

For a system of `n` components in series (all must work for the system to function):

    lambda_system = sum(lambda_i)  for i = 1..n

    R_system(t) = product(R_i(t)) = exp(-lambda_system * t)

This is the fundamental assumption: every component is a single point of failure. The block diagram editor allows modelling redundancy (parallel paths), but the default is pure series.

### 1.3 IEC TR 62380 General Component Model

Each component's failure rate is decomposed as:

    lambda = (lambda_die + lambda_package + lambda_overstress + lambda_EOS) * pi_process

where:
- `lambda_die`: intrinsic semiconductor/material failure rate, temperature-dependent
- `lambda_package`: solder joint and packaging failures from thermal cycling
- `lambda_overstress`: stress-related degradation
- `lambda_EOS`: Electrical Overstress contribution (interface circuits only)
- `pi_process`: process quality factor

---

## 2. Component Failure Rate Models

### 2.1 Temperature Acceleration (Arrhenius Law)

The die failure rate is strongly temperature-dependent per the Arrhenius equation:

    pi_T = exp[ (Ea / k_B) * (1/T_ref - 1/T_j) ]

where:
- `Ea` is the activation energy (eV), typically 0.7 eV for CMOS
- `k_B = 8.617e-5 eV/K` (Boltzmann constant)
- `T_ref = 293 K` (20 deg C reference)
- `T_j` is the junction temperature in Kelvin

**Impact:** A 10 deg C increase in junction temperature roughly doubles the failure rate for Ea = 0.7 eV.

### 2.2 Thermal Cycling (Coffin-Manson)

Package and solder joint failures are driven by thermal cycling:

    pi_N = N_cycles^0.76           for N <= 8760
    pi_N = 1.7 * N_cycles^0.6     for N > 8760

    lambda_package = lambda_base_pkg * pi_N * f(delta_T, CTE_mismatch)

where:
- `N_cycles` is the number of thermal cycles per year
- `delta_T` is the temperature swing per cycle (deg C)
- CTE mismatch between component and substrate drives the stress amplitude

The CTE mismatch factor uses material-specific thermal expansion coefficients (ppm/deg C) for both the package (plastic: 21.5, ceramic: 6.5) and the substrate (FR4: 15.0, alumina: 6.5, etc.).

### 2.3 Electrical Overstress (EOS)

Interface circuits (I/O pins connected to the outside world) add an EOS failure contribution:

    lambda_EOS = L_EOS_value    (from lookup table, in FIT)

Values depend on the environment type: Computer (10 FIT), Telecom (15-70 FIT), Avionics (20 FIT), Power Supply (40 FIT), Space (25-35 FIT).

### 2.4 Working Time Ratio (tau_on)

The die failure rate scales with the fraction of time the component is powered:

    lambda_effective = lambda_die * tau_on + lambda_package

where `tau_on` ranges from 0 (always off) to 1 (continuous operation).

### 2.5 Supported Component Types

The plugin implements IEC TR 62380 models for:

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

## 3. System Reliability Model

### 3.1 Block Diagram

The block diagram editor defines the system topology:
- **Series blocks:** All must function. Lambda values add.
- **Parallel (redundant) blocks:** System survives if at least one path works.

For `k`-of-`n` redundancy:

    R_parallel(t) = 1 - product(1 - R_i(t))   for 1-of-n

### 3.2 Active Sheet Filtering

Only schematic sheets included in the block diagram are considered in the analysis. Sheets not wired into the diagram (test fixtures, unused variants) are automatically excluded. This ensures the analysis matches the actual deployed hardware.

---

## 4. Monte Carlo Uncertainty Analysis

### 4.1 Mathematical Basis

Component parameters (temperature, power, voltage) have inherent uncertainty. The Monte Carlo method propagates this uncertainty through the nonlinear IEC TR 62380 models to produce a probability distribution of system reliability.

**Algorithm:**
1. For each simulation `i = 1..N`:
   a. For each component `j`, sample each parameter from a distribution centred on its nominal value:
      `p_j ~ Uniform(p_nominal * (1 - u), p_nominal * (1 + u))`
      where `u` is the uncertainty fraction (default 25%).
   b. Recompute `lambda_j` using the IEC TR 62380 model with perturbed parameters.
   c. Sum to get `lambda_system_i`.
   d. Compute `R_i = exp(-lambda_system_i * t_mission)`.
2. The resulting `{R_1, ..., R_N}` samples characterise the reliability distribution.

### 4.2 Confidence Intervals

From the empirical distribution, the plugin computes:
- **Mean** and **standard deviation** of R(t)
- **Configurable confidence interval** at 80%, 90%, 95%, or 99%:

      CI = [R_{alpha/2}, R_{1-alpha/2}]

  where alpha = 1 - confidence_level and percentiles are taken from the sorted samples.

### 4.3 Convergence

The running mean plot shows whether N simulations is sufficient. The mean should stabilise to within the desired precision. Typical convergence requires 3000-10000 simulations.

### 4.4 Override Handling

Components with a lambda override (see Section 8) use a fixed failure rate and are NOT perturbed during Monte Carlo. This reflects the higher certainty of measured/datasheet values.

---

## 5. Tornado Sensitivity Analysis

### 5.1 Method: One-At-a-Time (OAT) Deterministic Sensitivity

The tornado chart uses the standard **one-at-a-time (OAT)** sensitivity method, recommended by IEC 60300-3-1 for reliability models where the output is a monotonic function of inputs (which is the case for the IEC TR 62380 additive lambda model).

**Why OAT instead of Sobol?**

The IEC TR 62380 system failure rate is:

    lambda_system = sum(lambda_i(params_i))

This is a **linear sum** of component failure rates. Each component's lambda depends only on its own parameters. In a linear model, Sobol first-order indices `S_i` are all approximately equal (each sheet contributes proportionally to its share of total lambda), and total-order indices `S_T,i` approximately equal `S_i` (no interactions). This makes Sobol uninformative -- it tells you what you already know from the contributions tab.

OAT perturbation is the correct tool here because:
1. It directly answers "if this factor changes by X%, how much does system FIT change?"
2. For additive models, OAT captures 100% of the variance (no missed interactions).
3. It is deterministic (no sampling noise) and computationally cheap.
4. The results are directly actionable for design decisions.

### 5.2 Sheet-Level Tornado

For each schematic sheet `s`:
1. Compute `lambda_system` at baseline.
2. Perturb sheet's lambda: `lambda_s_low = lambda_s * (1 - p)`, `lambda_s_high = lambda_s * (1 + p)`.
3. Compute system FIT with the perturbed value.
4. Record: `swing = FIT_high - FIT_low`.

The chart ranks sheets by swing. **The sheet with the largest swing is the highest-priority target for redesign.**

### 5.3 Parameter-Level Tornado

For each design parameter (e.g. `t_ambient`, `operating_power`):
1. Identify all components that use this parameter.
2. Perturb the parameter by +/- p% across ALL affected components simultaneously.
3. Recompute each component's lambda using the IEC TR 62380 model.
4. Sum the deltas to get the system-level impact.

This answers: "If the ambient temperature increases 20%, how much worse does system reliability get?" -- a critical question for thermal design.

### 5.4 How to Use

- **Start with sheet-level** to identify which subsystem dominates system FIT.
- **Switch to parameter-level** to identify which design parameter has the most leverage.
- If `t_junction` or `t_ambient` has the largest swing, invest in thermal management.
- If `n_cycles` or `delta_t` dominates, the board is thermomechanically limited -- consider conformal coating or vibration isolation.

---

## 6. Design Margin Analysis

### 6.1 Purpose

Design margin analysis answers: **"How robust is my design to environmental and operational changes?"**

Unlike sensitivity analysis (which varies parameters by a small percentage), design margin evaluates specific, physically meaningful scenarios that represent real-world conditions your board might face.

### 6.2 Built-In Scenarios

| Scenario | What It Models | Design Action if Delta > 10% |
|----------|---------------|------------------------------|
| Temp +10 C | Hotter enclosure, reduced airflow | Add heatsinks, improve ventilation |
| Temp +20 C | Worst-case thermal environment | Derate components, redesign thermal path |
| Power derate 70% | Running components at 70% rated power | Already a standard derating strategy |
| Power derate 50% | Aggressive derating (ECSS/MIL best practice) | Quantifies the reliability gain |
| Thermal cycles x2 | Harsher vibration/thermal profile | Use flexible solder, conformal coating |
| Delta-T x2 | Larger temperature swings per cycle | Improve thermal mass, reduce gradients |
| 50% duty cycle | Intermittent operation / sleep modes | Shows benefit of power management |

### 6.3 How Each Scenario Works

For each scenario, the plugin:
1. Takes the baseline component parameters.
2. Applies the modification (e.g., adds 10 deg C to all temperature fields).
3. Recomputes every component's lambda from scratch using the full IEC TR 62380 model.
4. Sums to get the new system lambda and reliability.
5. Reports the delta as both absolute FIT change and percentage change.

Components with a lambda override are NOT modified by scenarios (their failure rate is fixed).

### 6.4 How to Use

- Run design margin analysis early in the design cycle.
- If "Temp +10 C" causes > 5% increase in FIT, your thermal margin is thin.
- Compare "Power derate 70%" vs baseline to quantify the benefit of derating.
- If "Thermal cycles x2" causes a large increase, your package/solder choices are critical.

---

## 7. Component Criticality Analysis

### 7.1 Method

Criticality analysis identifies which input parameters most influence each component's failure rate. For each of the top-N highest-FIT components:

1. For each parameter `p` with nominal value `v`:
   a. Compute `lambda(v * (1+epsilon))` and `lambda(v * (1-epsilon))`.
   b. Compute the **elasticity** (normalised sensitivity):

          E_p = (delta_lambda / lambda_base) / (delta_p / p_nominal)

   c. Compute **impact** as the percentage change in lambda for the given perturbation.

### 7.2 Field Selection

The field picker lets you choose which parameters to include per component category. This is useful when you want to focus validation on specific parameters (e.g., only temperatures, or only voltage stress ratios).

Categories are listed in logical order with ICs (Digital, Analog, FPGA) appearing first, followed by passives, discretes, and interconnects.

---

## 8. Lambda Override

### 8.1 When to Use

The IEC TR 62380 model is a prediction based on generic component categories. In some cases, you have better data:

- **Manufacturer datasheet** provides a tested FIT value (e.g., "< 5 FIT at 55 deg C junction")
- **FIDES** or other prediction methodology gives a different result
- **Field data** from deployed systems provides measured failure rates
- **Qualification test results** (HTOL, TC, etc.) give a known failure rate

### 8.2 How It Works

In the Component Editor, check "Override calculated lambda with fixed value" and enter the FIT value directly. When overridden:

- The IEC TR 62380 model fields are greyed out (not used)
- The component uses the fixed FIT value in all analyses
- Monte Carlo does NOT perturb this component (reflecting higher certainty)
- Design margin scenarios do NOT modify this component
- Criticality analysis skips this component
- Reports show an "Override" badge

### 8.3 Units

Enter the value in **FIT** (Failures In Time = failures per 10^9 hours).

---

## 9. Component Type Exclusion

### 9.1 Purpose

Not all components on a schematic contribute to the reliability model. You may want to exclude:

- **Connectors** on test points (not in the deployed product)
- **Mechanical components** (mounting hardware, heatsinks)
- **Fiducials, logos, or reference markers**
- **Components already accounted for in a subsystem-level override**

### 9.2 How It Works

In the Contributions tab, uncheck any component type to exclude it from ALL analyses:
- System FIT is recomputed without those components
- Monte Carlo, Tornado, Design Margin, and Criticality all respect the exclusion
- Reports only include the checked types
- The exclusion is non-destructive: re-check to include them again

---

## 10. Report Generation

### 10.1 HTML Report

The primary output: a self-contained HTML file with embedded SVG charts, styled tables, and interactive navigation. Suitable for web viewing, printing, or archiving.

### 10.2 PDF Report

Export to PDF for formal documentation and certification submissions. The PDF is generated from the HTML report using reportlab, preserving all tables and metrics.

### 10.3 Markdown and JSON

Available via the ReportGenerator API for integration with documentation pipelines or automated analysis systems.

---

## 11. Practical Workflow: Designing a Reliable Board

### Step 1: Import Schematic

Open your KiCad project and launch the Reliability Plugin. The schematic parser extracts all components with their reference designators and values.

### Step 2: Classify and Parameterise Components

- Auto-classify maps reference designators to IEC TR 62380 categories.
- Open the batch editor to review and adjust: set junction temperatures, power ratings, voltage stress ratios.
- For components with datasheet FIT values, use the **Lambda Override**.

### Step 3: Set Mission Profile

- Mission duration (hours or years)
- Thermal cycling: cycles per year and delta-T per cycle
- These define the environmental stress that drives package/solder failures.

### Step 4: Set Up Block Diagram

Wire schematic sheets into the block diagram. Only sheets in the diagram are analysed. Use parallel paths for redundant subsystems.

### Step 5: Run Baseline Analysis

Check the Contributions tab. The Pareto chart shows which sheets/components dominate system FIT. If one component contributes > 30% of total FIT, it's the design bottleneck.

### Step 6: Run Monte Carlo

Set uncertainty to 25% (typical for early design) and run 5000+ simulations. The 90% CI gives the range of likely system reliability. If the CI is too wide, reduce uncertainty in critical parameters through better characterisation.

### Step 7: Run Tornado Analysis

- **Sheet-level** first: identify the subsystem to focus on.
- **Parameter-level** next: identify whether temperature, power, or cycling is the dominant stress.

### Step 8: Run Design Margin Analysis

Evaluate robustness to worst-case conditions. If "Temp +10 C" causes > 10% FIT increase, your thermal design needs attention.

### Step 9: Run Criticality Analysis

For the top-10 highest-FIT components, see which parameter has the most influence. This tells you exactly which specifications to tighten.

### Step 10: Iterate

Based on the analysis:
- Add heatsinks or improve airflow (reduces T_junction)
- Derate components (use 50V capacitor on 12V rail instead of 16V)
- Replace high-FIT components with more reliable alternatives
- Add redundancy for critical functions

Re-run the analysis to verify improvements.

### Step 11: Export Report

Generate HTML or PDF report for design review, certification, or archival.

---

## 12. References

1. **IEC TR 62380:2004** -- Reliability data handbook -- Universal model for reliability prediction of electronics components, PCBs and equipment.
2. **IEC 61709** -- Electronic components -- Reliability -- Reference conditions for failure rates and stress models for conversion.
3. **IEC 60300-3-1** -- Dependability management -- Part 3-1: Application guide -- Analysis techniques for dependability.
4. **ECSS-Q-ST-30-02C** -- Space product assurance -- Failure modes, effects (and criticality) analysis (FMEA/FMECA).
5. **MIL-HDBK-217F** -- Military Handbook: Reliability Prediction of Electronic Equipment (legacy, superseded by IEC TR 62380 for modern electronics).
6. Coffin, L.F. Jr. (1954). "A Study of the Effects of Cyclic Thermal Stresses on a Ductile Metal." *Trans. ASME*, 76, 931-950.
7. Manson, S.S. (1953). "Behavior of Materials Under Conditions of Thermal Stress." NACA TN-2933.

---

*Designed and developed by Eliot Abramo*
*KiCad Reliability Plugin v3.1.0*
