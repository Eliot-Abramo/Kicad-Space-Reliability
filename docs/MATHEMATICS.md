# Reliability Methodology And Math Notes

This document explains the math implemented by the plugin, with the goal of
making the numerical outputs easy to audit against the code.

The main implementation lives in:

- `reliability_math.py`
- `mission_profile.py`
- `monte_carlo.py`
- `sensitivity_analysis.py`
- `budget_allocation.py`
- `derating_engine.py`
- `analysis/engine.py`

## 1. Units, conventions, and overall flow

The plugin works with two closely related failure-rate units:

- `FIT`: failures per `10^9` device-hours
- `lambda`: failures per hour

The conversion is:

```text
lambda = FIT * 1e-9
FIT = lambda * 1e9
```

Most component models in `reliability_math.py` build up contributions in FIT,
then convert the returned `lambda_total` to failures per hour at the output
boundary.

At a high level the tool does this:

```text
component parameters
-> component lambda_i
-> sheet lambda
-> system reliability model
-> R(t), MTTF, uncertainty, sensitivity, and budget views
```

For a constant hazard rate, the core reliability identities are:

```text
R(t) = exp(-lambda * t)
lambda = -ln(R) / t
MTTF = 1 / lambda
```

These are implemented by:

- `reliability_from_lambda()`
- `lambda_from_reliability()`
- `mttf_from_lambda()`

## 2. Shared acceleration factors

Several component families reuse the same stress factors.

### 2.1 Arrhenius temperature acceleration

Implemented by `pi_temperature()`:

```text
pi_T = exp(ea * (1 / T_ref - 1 / T_op))
T_op = 273 + T_C
```

Where:

- `ea` is stored as `Ea / k_B` in Kelvin
- `T_ref` is the reference temperature in Kelvin from the standard tables
- `T_C` is the user-entered operating temperature in degC

This makes temperature influence exponential. A modest rise in junction or
ambient temperature can therefore produce a large increase in lambda.

### 2.2 Thermal cycling acceleration

Implemented by `pi_thermal_cycles()`:

```text
pi_n = n_cycles^0.76                  if n_cycles <= 8760
pi_n = 1.7 * n_cycles^0.6             if n_cycles > 8760
```

Package and solder-like terms then multiply this by a power law in
temperature excursion:

```text
package_term ~ pi_n * DeltaT^0.68
```

This is the Coffin-Manson style fatigue contribution used throughout the code.

### 2.3 CTE mismatch factor

Implemented by `pi_alpha()`:

```text
pi_alpha = 0.06 * |alpha_s - alpha_p|^1.68
```

Where:

- `alpha_s` is substrate CTE
- `alpha_p` is package CTE

This appears in the integrated-circuit package model.

### 2.4 Voltage stress factor

Implemented by `pi_voltage_stress()`:

```text
pi_V = (V_applied / V_rated)^m
```

The exponent `m` depends on the component family. For example, capacitor
models use family-specific exponents such as `2.5`, `3.0`, `4.0`, or `5.0`.

### 2.5 EOS term for interface circuits

Implemented by `lambda_eos()`:

```text
lambda_EOS = pi_I * lambda_EOS_table
```

The actual value comes from `INTERFACE_EOS_VALUES`, keyed by context such as
computer, telecom, automotive, or space.

## 3. Component-level math

All component models return a dictionary with a total failure rate:

```text
lambda_total   # failures / hour
fit_total      # failures / 1e9 hours
```

The most important pattern is additive composition:

```text
lambda_total_fit = lambda_die_fit + lambda_package_fit + lambda_other_fit
```

The exact terms vary by family.

### 3.1 Integrated circuits

Implemented by `lambda_integrated_circuit()`.

The die term is:

```text
effective_n = transistor_count * n_per
a = max(construction_year - 1998, 0)

lambda_die_fit =
    (l1 * effective_n * exp(-0.35 * a) + l2) * pi_T * tau_on
```

Where:

- `l1`, `l2`, `ea`, `T_ref`, and `n_per` come from `IC_DIE_TABLE`
- `tau_on` is the duty ratio
- `exp(-0.35 * a)` models construction-year improvement

The package term is:

```text
lambda_pkg_fit =
    2.75e-3 * pi_alpha * pi_n * DeltaT^0.68 * lambda_3
```

Where `lambda_3` comes from `calculate_ic_lambda3()` and depends on package
style, pin count, or package diagonal depending on the chosen package table.

The total is:

```text
lambda_total_fit = lambda_die_fit + lambda_pkg_fit + lambda_EOS_fit
```

### 3.2 Diodes

Implemented by `lambda_diode()`.

The plugin uses:

```text
lambda_die_fit = pi_U * lambda_0 * pi_T * tau_on
lambda_pkg_fit = 2.75e-3 * pi_n * DeltaT^0.68 * lambda_B
lambda_total_fit = lambda_die_fit + lambda_pkg_fit + lambda_EOS_fit
```

In the current implementation `pi_U = 1` for diodes, and `lambda_B` comes from
the discrete package table.

### 3.3 Transistors

Implemented by `lambda_transistor()`.

The stress term depends on the transistor technology:

```text
pi_S = 0.22 * exp(1.7 * Vce_ratio)                              # bipolar
pi_S = 0.22 * exp(1.7 * Vds_ratio) * exp(3.0 * Vgs_ratio)      # mos/gan/sic
```

Then:

```text
lambda_die_fit = pi_S * lambda_0 * pi_T * tau_on
lambda_pkg_fit = 2.75e-3 * pi_n * DeltaT^0.68 * lambda_B
lambda_total_fit = lambda_die_fit + lambda_pkg_fit + lambda_EOS_fit
```

### 3.4 Optocouplers

Implemented by `lambda_optocoupler()`.

Forward-current stress is:

```text
pi_IF = (I_F_applied / I_F_rated)^2
```

Then:

```text
lambda_die_fit = lambda_0 * pi_T * pi_IF * tau_on
lambda_pkg_fit = 2.75e-3 * pi_n * DeltaT^0.68 * lambda_3
lambda_total_fit = lambda_die_fit + lambda_pkg_fit
```

### 3.5 Thyristors and TRIACs

Implemented by `lambda_thyristor()`.

The voltage stress factor is:

```text
pi_V = (V_applied / V_rated)^2.5
```

Then:

```text
lambda_die_fit = lambda_0 * pi_T * pi_V * tau_on
lambda_pkg_fit = 2.75e-3 * pi_n * DeltaT^0.68 * lambda_B
lambda_total_fit = lambda_die_fit + lambda_pkg_fit
```

### 3.6 Capacitors

Implemented by `lambda_capacitor()`.

The code first computes operating temperature:

```text
T_op = T_ambient + 20 * ripple_ratio^2    # aluminum families only
T_op = T_ambient                          # all other capacitor families
```

Then:

```text
lambda_base_fit = lambda_0 * pi_T * pi_V * tau_on
lambda_pkg_fit = lambda_0 * pkg_coef * pi_n * DeltaT^0.68
lambda_total_fit = lambda_base_fit + lambda_pkg_fit
```

This means capacitor voltage stress enters multiplicatively through `pi_V`,
while the package stress scales with the family's `pkg_coef`.

### 3.7 Resistors and potentiometers

Implemented by `lambda_resistor()`.

The model estimates resistor body temperature from ambient and power loading:

```text
power_ratio = min(P_operating / P_rated, 1.0)
T_resistor = T_ambient + temp_coef * power_ratio
```

Then:

```text
l0_eff = lambda_0 * n_resistors

lambda_base_fit = l0_eff * pi_T * tau_on
lambda_pkg_fit = l0_eff * pkg_coef * pi_n * DeltaT^0.68
lambda_total_fit = lambda_base_fit + lambda_pkg_fit
```

### 3.8 Inductors and transformers

Implemented by `lambda_inductor()`.

The component temperature estimate is:

```text
surface_area_dm2 = surface_area_mm2 / 10000
T_component = T_ambient + 8.2 * (power_loss / surface_area_dm2)
```

Then:

```text
lambda_base_fit = lambda_0 * pi_T * tau_on
lambda_pkg_fit = lambda_0 * 7e-3 * pi_n * DeltaT^0.68
lambda_total_fit = lambda_base_fit + lambda_pkg_fit
```

### 3.9 Relays

Implemented by `lambda_relay()`.

This model splits electrical, mechanical, and package stress:

```text
pi_contact = 1 + 2 * contact_current_ratio^2

lambda_electrical_fit = lambda_0 * pi_T * tau_on
lambda_mechanical_fit = mech_coef * cycles_per_hour * pi_contact
lambda_package_fit = 0.5 * 2.75e-3 * pi_n * DeltaT^0.68

lambda_total_fit =
    lambda_electrical_fit + lambda_mechanical_fit + lambda_package_fit
```

### 3.10 Connectors

Implemented by `lambda_connector()`.

The connector model is additive across contacts, housing, thermal fatigue, and
mating wear:

```text
lambda_contacts_fit = lambda_0_pin * n_contacts * tau_on
lambda_housing_fit = lambda_housing
lambda_thermal_fit = pkg_coef * n_contacts * pi_n * DeltaT^0.68
lambda_mating_fit = 0.01 * n_contacts * mating_cycles_per_year

lambda_total_fit =
    lambda_contacts_fit + lambda_housing_fit +
    lambda_thermal_fit + lambda_mating_fit
```

### 3.11 PCB and solder joints

Implemented by `lambda_pcb_solder()`.

```text
lambda_base_fit = lambda_0 * n_joints
lambda_thermal_fit = lambda_0 * 0.5e-3 * n_joints * pi_n * DeltaT^0.68
lambda_total_fit = lambda_base_fit + lambda_thermal_fit
```

### 3.12 Miscellaneous parts

Implemented by `lambda_misc_component()`.

```text
lambda_total_fit = base_fit * (tau_on + 3e-3 * pi_n * DeltaT^0.68)
```

If the subtype name contains `Connector`, the code multiplies the base rate by
the number of contacts before applying that expression.

## 4. Mission phasing

Mission-phased calculations live in `mission_profile.py`.

There are helper functions for analytically weighted die and package factors,
but the main production path is `compute_phased_lambda()`, which uses a simpler
and more robust rule:

```text
lambda_phased =
    sum over phases i of [ duration_frac_i * lambda_i(phase_i_params) ]
```

For each phase the tool overrides:

- ambient temperature
- junction temperature, if provided
- thermal cycle count
- thermal excursion
- duty ratio

Then it recomputes the full component lambda using the same
`calculate_component_lambda()` function used elsewhere.

This means phase weighting is applied to the full component model, not just to
one isolated stress factor.

For a single phase:

```text
lambda_phased = lambda_single_phase
```

## 5. Sheet and system aggregation

### 5.1 Sheet lambda

Sheets are treated as simple sums of their component lambdas:

```text
lambda_sheet = sum over components j of lambda_j
```

### 5.2 Simple series model

For a plain series system:

```text
lambda_sys = sum_i lambda_i
R_sys(t) = exp(-lambda_sys * t)
```

This is exposed directly by `lambda_series()` and `reliability_from_lambda()`.

### 5.3 Block-diagram reliability

For hierarchical architectures the code works in reliability space, not
lambda-space. `analysis/engine.py` computes leaf reliabilities from the sheet
lambdas and mission time, then combines them by topology:

Series:

```text
R_series = product_i R_i
```

Parallel:

```text
R_parallel = 1 - product_i (1 - R_i)
```

K-of-N:

```text
R_kofn = sum from i=k to n of C(n, i) * R^i * (1 - R)^(n - i)    # identical branches
```

For non-identical branches, the implementation uses the standard recursion:

```text
R_kofn(branches, k) =
    R_last * R_kofn(rest, k - 1) + (1 - R_last) * R_kofn(rest, k)
```

## 6. Monte Carlo uncertainty propagation

The production uncertainty engine lives in `monte_carlo.py`.

### 6.1 Input distributions

Two distribution families are supported.

Uniform:

```text
X ~ U(min, max)
```

PERT, implemented as a scaled Beta distribution with `gamma = 4`:

```text
alpha = 1 + gamma * (mode - min) / (max - min)
beta  = 1 + gamma * (max - mode) / (max - min)
X     = min + (max - min) * Beta(alpha, beta)
```

Shared parameters are modeled as one random delta per sample and are applied to
every component using that field. Independent parameters are sampled separately
for each component that owns the field.

### 6.2 Propagation

For simulation sample `s`:

```text
lambda_sys^(s) = sum_i lambda_i(theta^(s))
R^(s)(t) = exp(-lambda_sys^(s) * t)
```

The important detail is that uncertain components are re-evaluated through the
full IEC model for each sample. This is input-level propagation, not a shortcut
that perturbs outputs directly.

### 6.3 Statistics

The plugin reports:

- sample mean
- sample median
- sample standard deviation
- empirical quantile confidence interval

If the configured confidence level is `CL`, then:

```text
alpha = (1 - CL) / 2
CI = [quantile(alpha), quantile(1 - alpha)]
```

It also reports a frequentist half-width for the mean:

```text
halfwidth = z_(1 - alpha) * s / sqrt(N)
```

Where `s` is the sample standard deviation and `N` is the number of
simulations.

### 6.4 Jensen diagnostic

The code explicitly checks Jensen's inequality for:

```text
f(lambda) = exp(-lambda * t)
```

Since:

```text
f''(lambda) = t^2 * exp(-lambda * t) > 0
```

`f` is convex, so:

```text
E[f(lambda)] >= f(E[lambda])
```

The plugin compares:

- mean sampled reliability `E[R(t)]`
- reliability at the mean sampled lambda `R(E[lambda], t)`

This is used as a consistency diagnostic on the Monte Carlo run.

## 7. Parameter importance and deterministic sensitivity

### 7.1 SRRC ranking

`monte_carlo.py` computes Standardized Rank Regression Coefficients.

The algorithm is:

1. Rank-transform each input column and the output vector.
2. Standardize the ranked variables.
3. Solve the least-squares regression:

```text
Y_ranked = X_ranked * beta
```

4. Interpret `beta_j` as the SRRC for parameter `j`.

The report also shows `SRRC^2` and a normalized `variance_fraction`:

```text
variance_fraction_j = SRRC_j^2 / sum_k SRRC_k^2
```

This is a ranking aid for monotonic relationships. It is not a Sobol
decomposition.

### 7.2 Tornado sensitivity

`sensitivity_analysis.py` implements One-At-a-Time sensitivity.

Each perturbation changes one parameter, recomputes the affected lambdas, and
measures the system FIT movement.

For a low and high perturbation:

```text
FIT_swing = |FIT_high - FIT_low|
```

The docstring also describes the underlying central-difference idea:

```text
f'(x) ~= (f(x + h) - f(x - h)) / (2h)
```

This makes tornado charts a local leverage measure around the current design
point.

### 7.3 Component elasticity

The criticality view uses a normalized log-log sensitivity:

```text
E_theta = d(ln lambda) / d(ln theta)
        ~= (Delta lambda / lambda) / (Delta theta / theta)
```

An elasticity of `1` means a `1%` change in the parameter produces roughly a
`1%` change in component lambda near the current operating point.

## 8. Reliability budgeting

Budget allocation is implemented in `budget_allocation.py`.

The starting point is the system target:

```text
lambda_target = -ln(R_target) / mission_hours
FIT_target = lambda_target * 1e9
```

The code then applies a design margin:

```text
FIT_available = FIT_target * (1 - margin_percent / 100)
lambda_available = FIT_available * 1e-9
```

That available budget is apportioned by strategy.

Equal:

```text
budget_i = available / N
```

Proportional:

```text
scale = lambda_available / sum_j lambda_actual_j
budget_i = lambda_actual_i * scale
```

Complexity-weighted:

```text
sheet_fraction = n_components_in_sheet / total_components
sheet_budget = lambda_available * sheet_fraction
component_budget = sheet_budget / n_components_in_sheet
```

Criticality-weighted:

```text
budget_i = lambda_available * (1 / lambda_i) / sum_j (1 / lambda_j)
```

The component review fields are then:

```text
margin_fit = budget_fit - actual_fit
utilization = actual_fit / budget_fit
required_savings_fit = max(0, actual_fit - budget_fit)
```

## 9. Derating guidance

The derating engine in `derating_engine.py` takes a target component lambda and
searches for a parameter value that achieves it.

The current implementation uses binary search:

```text
find x such that lambda_component(x) ~= lambda_target
```

For each midpoint:

1. copy the component parameters
2. replace the chosen field with the trial value
3. recompute `lambda_total`
4. keep the half-interval that contains the target

So the derating recommendations are not based on a linear approximation. They
come from repeated evaluation of the full nonlinear model.

## 10. Practical interpretation notes

The plugin is mathematically consistent with a constant-hazard reliability
model plus IEC-style stress acceleration terms, but the outputs are still only
as good as the inputs and model scope.

Important implications:

- Temperature matters exponentially because of `pi_T`.
- Thermal cycling matters through a strong power law in both cycle count and
  excursion.
- Block topology must be modeled correctly, because parallel and K-of-N results
  do not reduce to simply summing FIT values.
- Monte Carlo captures uncertainty in the chosen inputs, not unknown physics
  outside the chosen model.
- Tornado and elasticity are local sensitivity tools; they rank leverage near
  the current operating point.

## 11. Code map

If you want to trace a number in the UI back to the implementation, these are
the main entry points:

- `calculate_component_lambda()` dispatches component-family formulas
- `compute_phased_lambda()` applies mission-phase overrides
- `reliability_from_lambda()` converts lambda to mission reliability
- `r_series()`, `r_parallel()`, and `r_k_of_n()` combine block reliabilities
- `run_uncertainty_analysis()` performs Monte Carlo propagation
- `tornado_analysis()` computes deterministic sensitivity swings
- `allocate_budget()` turns a target reliability into FIT budgets
- `_find_required_value()` in `derating_engine.py` solves inverse derating

