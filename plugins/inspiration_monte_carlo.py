"""
Task 1: Monte Carlo Simulation for Reliability Analysis
=========================================================

RANDOM PARAMETERS SPECIFICATIONS:
----------------------------------
Refer to the comments in the original code for the probability distributions
of each parameter. Key parameters with uncertainty:

1. LamB (Lambda_B values from IEC page 37):
   - Q5, Q6, D3, D12, D2:                                                       uniform(5.7, 6.9)
   - D10, D8:                                                                   uniform(1, 6.9)
   - Q10, Q14, Q17, Q19, Q20, Q22, Q12, Q13, Q16, Q23, Q26, Q32, Q34, Q36,      uniform(1, 6.9)
   - Q9, Q1, Q11, Q15, Q18, Q2, Q21, Q28, Q3, Q4, Q8, D4, D5, D6, D7:           uniform(1, 6.9)
   - Q24, Q25, Q27, Q33, Q35, Q37, Q7:                                          uniform(5.7, 6.9)

2. Lam3 (Lambda_3 values from IEC page 34):
   - U22:                                                                       50% probability 6.479, 50% probability 1.3
   - U17, U19:                                                                  uniform(0.315, 0.627)
   - U11, U21, U3, U7:                                                          uniform(0.202, 0.371)
   - U42:                                                                       uniform(0.084, 0.118)
   - U10, U2, U6:                                                               50% probability 4.1, 50% probability 1.3
   - U12, U4, U8:                                                               uniform(1.3, 4.1)
   - U35:                                                                       uniform(1.3, 2.94)
   - U23, U32, U14, U20, U25, U27, U29, U31, U36, U39, U40, U41:                50% probability 1.164, 50% probability 0.2808

3. VDS (Drain-Source Voltage):
   - Q5, Q6:                                                                    uniform(17, 23)
   - All other VDS:                                                             uniform(1.5, 2.5)

4. VCE (Collector-Emitter Voltage):
   - Q10, Q14, Q17, Q19, Q20, Q22:                                              uniform(10, 15)
   - Q12, Q13, Q16, Q23, Q26, Q32, Q34, Q36, Q9:                                uniform(3, 3.6)

5. Operating Power:
   - U42, U23, U32, U41, U33, U34, U43:                                         uniform(3, 5)
   - L1, L2, L3, L4, L5:                                                        uniform(5, 15)
   - All other components:                                                      uniform(0.5, 1.5)

YOUR TASKS:
-----------
Vous trouverez si dessous la liste des composantes aléatoires de notre système. Avant de pouvoir passer au calcul de fiabilité, il nous faut calculer
les taux de défaillance des différents composants. Pour cela, vous pourrez utiliser les fonctions fournies dans reliability_math. Les valeurs de certains
paramètres qu'utilisent ses formules sont données par une norme qui se basent sur des essais laboratoires. Cela donne donc des valeurs qui ne sont pas
réellement précises et qui peuvent comporter un biais. Notre but ici est d'estimer au mieux ces différents paramètres "aléatoires", dont les lois vous sont
données si dessus. Estimer ces différents paramètres et calculer les taux de défaillance.
-----------

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math
import reliability_math as rm

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    INFO = '\033[10m'

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


# ============================================================================
# PARAMETER DISTRIBUTIONS DEFINITIONS
# ============================================================================

# LamB distributions (Table 18 - Package lambda values)
# Format: 'uniform' -> (low, high) or 'discrete' -> (val1, val2, prob_val1)
LAMB_DISTRIBUTIONS = {
    # uniform(5.7, 6.9)
    'Q5': ('uniform', 5.7, 6.9),
    'Q6': ('uniform', 5.7, 6.9),
    'D3': ('uniform', 5.7, 6.9),
    'D12': ('uniform', 5.7, 6.9),
    'D2': ('uniform', 5.7, 6.9),
    'Q24': ('uniform', 5.7, 6.9),
    'Q25': ('uniform', 5.7, 6.9),
    'Q27': ('uniform', 5.7, 6.9),
    'Q33': ('uniform', 5.7, 6.9),
    'Q35': ('uniform', 5.7, 6.9),
    'Q37': ('uniform', 5.7, 6.9),
    'Q7': ('uniform', 5.7, 6.9),
    # uniform(1, 6.9) - all others
    'D10': ('uniform', 1.0, 6.9),
    'D8': ('uniform', 1.0, 6.9),
    'Q10': ('uniform', 1.0, 6.9),
    'Q14': ('uniform', 1.0, 6.9),
    'Q17': ('uniform', 1.0, 6.9),
    'Q19': ('uniform', 1.0, 6.9),
    'Q20': ('uniform', 1.0, 6.9),
    'Q22': ('uniform', 1.0, 6.9),
    'Q12': ('uniform', 1.0, 6.9),
    'Q13': ('uniform', 1.0, 6.9),
    'Q16': ('uniform', 1.0, 6.9),
    'Q23': ('uniform', 1.0, 6.9),
    'Q26': ('uniform', 1.0, 6.9),
    'Q32': ('uniform', 1.0, 6.9),
    'Q34': ('uniform', 1.0, 6.9),
    'Q36': ('uniform', 1.0, 6.9),
    'Q9': ('uniform', 1.0, 6.9),
    'Q1': ('uniform', 1.0, 6.9),
    'Q11': ('uniform', 1.0, 6.9),
    'Q15': ('uniform', 1.0, 6.9),
    'Q18': ('uniform', 1.0, 6.9),
    'Q2': ('uniform', 1.0, 6.9),
    'Q21': ('uniform', 1.0, 6.9),
    'Q28': ('uniform', 1.0, 6.9),
    'Q3': ('uniform', 1.0, 6.9),
    'Q4': ('uniform', 1.0, 6.9),
    'Q8': ('uniform', 1.0, 6.9),
    'D4': ('uniform', 1.0, 6.9),
    'D5': ('uniform', 1.0, 6.9),
    'D6': ('uniform', 1.0, 6.9),
    'D7': ('uniform', 1.0, 6.9),
}

# Lam3 distributions (Lambda_3 for ICs - IEC page 34)
LAM3_DISTRIBUTIONS = {
    # 50% probability 6.479, 50% probability 1.3
    'U22': ('discrete', 6.479, 1.3, 0.5),
    # uniform(0.315, 0.627)
    'U17': ('uniform', 0.315, 0.627),
    'U19': ('uniform', 0.315, 0.627),
    # uniform(0.202, 0.371)
    'U11': ('uniform', 0.202, 0.371),
    'U21': ('uniform', 0.202, 0.371),
    'U3': ('uniform', 0.202, 0.371),
    'U7': ('uniform', 0.202, 0.371),
    # uniform(0.084, 0.118)
    'U42': ('uniform', 0.084, 0.118),
    # 50% probability 4.1, 50% probability 1.3
    'U10': ('discrete', 4.1, 1.3, 0.5),
    'U2': ('discrete', 4.1, 1.3, 0.5),
    'U6': ('discrete', 4.1, 1.3, 0.5),
    # uniform(1.3, 4.1)
    'U12': ('uniform', 1.3, 4.1),
    'U4': ('uniform', 1.3, 4.1),
    'U8': ('uniform', 1.3, 4.1),
    # uniform(1.3, 2.94)
    'U35': ('uniform', 1.3, 2.94),
    # 50% probability 1.164, 50% probability 0.2808
    'U23': ('discrete', 1.164, 0.2808, 0.5),
    'U32': ('discrete', 1.164, 0.2808, 0.5),
    'U14': ('discrete', 1.164, 0.2808, 0.5),
    'U20': ('discrete', 1.164, 0.2808, 0.5),
    'U25': ('discrete', 1.164, 0.2808, 0.5),
    'U27': ('discrete', 1.164, 0.2808, 0.5),
    'U29': ('discrete', 1.164, 0.2808, 0.5),
    'U31': ('discrete', 1.164, 0.2808, 0.5),
    'U36': ('discrete', 1.164, 0.2808, 0.5),
    'U39': ('discrete', 1.164, 0.2808, 0.5),
    'U40': ('discrete', 1.164, 0.2808, 0.5),
    'U41': ('discrete', 1.164, 0.2808, 0.5),
}

# VDS distributions (Drain-Source Voltage for MOS transistors)
VDS_DISTRIBUTIONS = {
    # uniform(17, 23)
    'Q5': ('uniform', 17.0, 23.0),
    'Q6': ('uniform', 17.0, 23.0),
    # All other VDS: uniform(1.5, 2.5) - handled as default
}
VDS_DEFAULT = ('uniform', 1.5, 2.5)

# VCE distributions (Collector-Emitter Voltage for Bipolar transistors)
VCE_DISTRIBUTIONS = {
    # uniform(10, 15)
    'Q10': ('uniform', 10.0, 15.0),
    'Q14': ('uniform', 10.0, 15.0),
    'Q17': ('uniform', 10.0, 15.0),
    'Q19': ('uniform', 10.0, 15.0),
    'Q20': ('uniform', 10.0, 15.0),
    'Q22': ('uniform', 10.0, 15.0),
    # uniform(3, 3.6)
    'Q12': ('uniform', 3.0, 3.6),
    'Q13': ('uniform', 3.0, 3.6),
    'Q16': ('uniform', 3.0, 3.6),
    'Q23': ('uniform', 3.0, 3.6),
    'Q26': ('uniform', 3.0, 3.6),
    'Q32': ('uniform', 3.0, 3.6),
    'Q34': ('uniform', 3.0, 3.6),
    'Q36': ('uniform', 3.0, 3.6),
    'Q9': ('uniform', 3.0, 3.6),
}

# Operating Power distributions
POWER_DISTRIBUTIONS = {
    # uniform(3, 5)
    'U42': ('uniform', 3.0, 5.0),
    'U23': ('uniform', 3.0, 5.0),
    'U32': ('uniform', 3.0, 5.0),
    'U41': ('uniform', 3.0, 5.0),
    'U33': ('uniform', 3.0, 5.0),
    'U34': ('uniform', 3.0, 5.0),
    'U43': ('uniform', 3.0, 5.0),
    # uniform(5, 15)
    'L1': ('uniform', 5.0, 15.0),
    'L2': ('uniform', 5.0, 15.0),
    'L3': ('uniform', 5.0, 15.0),
    'L4': ('uniform', 5.0, 15.0),
    'L5': ('uniform', 5.0, 15.0),
    # All other: uniform(0.5, 1.5) - handled as default
}
POWER_DEFAULT = ('uniform', 0.5, 1.5)




# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def sample_from_distribution(dist_spec: tuple, rng: np.random.Generator) -> float:
    """
    Sample a single value from a specified probability distribution.

    Supports uniform distributions (continuous between bounds) and discrete
    distributions (two possible values with given probability).

    Args:
        dist_spec: Tuple defining the distribution type and parameters.
            - ('uniform', low, high): Uniform distribution between low and high.
            - ('discrete', val1, val2, prob_val1): Discrete distribution where
              val1 is returned with probability prob_val1, otherwise val2.
        rng: NumPy random generator for reproducible sampling.

    Returns:
        float: A single sampled value from the specified distribution.

    Raises:
        ValueError: If the distribution type is not recognized.
    """
    dist_type = dist_spec[0]

    if dist_type == 'uniform':
        low, high = dist_spec[1], dist_spec[2]
        return rng.uniform(low, high)
    elif dist_type == 'discrete':
        val1, val2, prob_val1 = dist_spec[1], dist_spec[2], dist_spec[3]
        return val1 if rng.random() < prob_val1 else val2
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def sample_parameters(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample random parameters and apply them to a DataFrame copy.

    Creates a copy of the input DataFrame and replaces certain parameter values
    (LamB, Lam3, VDS, VCE, Operating Power) with values sampled from their
    predefined probability distributions. This enables Monte Carlo uncertainty
    quantification.

    Args:
        df: Original DataFrame containing component data with columns like
            'Reference', 'LamB_sampled', 'Lam3', 'Max applied VDS', etc.
        rng: NumPy random generator for reproducible sampling.

    Returns:
        pd.DataFrame: A new DataFrame with sampled parameter values replacing
            the original values for components defined in the distribution
            dictionaries (LAMB_DISTRIBUTIONS, LAM3_DISTRIBUTIONS, etc.).
    """
    df_sampled = df.copy()

    # Initialize LamB_sampled column if not present
    if 'LamB_sampled' not in df_sampled.columns:
        df_sampled['LamB_sampled'] = np.nan

    for idx, row in df_sampled.iterrows():
        ref = row.get('Reference', '')

        # Sample LamB (stored in 'Table 18' column as package type,
        # but we need to override the l_b function result)
        if ref in LAMB_DISTRIBUTIONS:
            df_sampled.at[idx, 'LamB_sampled'] = sample_from_distribution(
                LAMB_DISTRIBUTIONS[ref], rng
            )

        # Sample Lam3 for ICs
        if ref in LAM3_DISTRIBUTIONS:
            df_sampled.at[idx, 'Lam3'] = sample_from_distribution(
                LAM3_DISTRIBUTIONS[ref], rng
            )

        # Sample VDS for MOS transistors
        if ref in VDS_DISTRIBUTIONS:
            df_sampled.at[idx, 'Max applied VDS'] = sample_from_distribution(
                VDS_DISTRIBUTIONS[ref], rng
            )
        else:
            df_sampled.at[idx, 'Max applied VDS'] = sample_from_distribution(
                VDS_DEFAULT, rng
            )

        # Sample VCE for Bipolar transistors
        if ref in VCE_DISTRIBUTIONS:
            df_sampled.at[idx, 'Max repetitive VCE'] = sample_from_distribution(
                VCE_DISTRIBUTIONS[ref], rng
            )

        # Sample Operating Power
        if ref in POWER_DISTRIBUTIONS:
            df_sampled.at[idx, 'Operating_Power'] = sample_from_distribution(
                POWER_DISTRIBUTIONS[ref], rng
            )
        else:
            df_sampled.at[idx, 'Operating_Power'] = sample_from_distribution(
                POWER_DEFAULT, rng
            )

    return df_sampled


# ============================================================================
# MONTE CARLO CALCULATION FUNCTIONS
# ============================================================================

def calculate_block_reliability_mc(df: pd.DataFrame, sheet_name: str,
                                    ni: int = rm.NI, dt: float = rm.DT,
                                    t_mission: float = rm.T_MISSION,
                                    pi_i: float = rm.PI_I, leos: float = rm.LEOS
                                    ) -> Tuple[float, float, Dict[str, Tuple[float, str]]]:
    """
    Calculate reliability for a block using Monte Carlo sampled parameters.

    Computes the failure rate (lambda) for each component in the block based on
    its class and parameters, then combines them in series to get block reliability.
    Uses the exponential reliability model: R = exp(-lambda_total * t_mission).

    Args:
        df: DataFrame containing component data with sampled parameters.
        sheet_name: Name of the block/sheet to analyze.
        ni: Number of thermal cycles per year.
        dt: Temperature cycle amplitude in degrees Celsius.
        t_mission: Mission duration in hours.
        pi_i: Integration factor for reliability calculation.
        leos: Low earth orbit stress factor.

    Returns:
        tuple: A tuple containing:
            - lambda_total (float): Sum of all component failure rates in the block.
            - reliability (float): Block reliability R = exp(-lambda_total * t_mission).
            - component_lambdas (dict): Dictionary mapping component reference to
              a tuple of (lambda_value, class_name).
    """
    block_df = df[df['Sheet'] == sheet_name].copy()

    if block_df.empty:
        print(f"{Colors.YELLOW}WARNING: No components found for sheet '{sheet_name}'{Colors.ENDC}")
        return 0.0, 1.0, {}

    lambdas = []
    component_lambdas = {}  # {reference: (lambda, class)}

    for idx, row in block_df.iterrows():
        component_class = row.get('Class', '')
        reference = row.get('Reference', 'Unknown')
        lam = None

        if pd.isna(component_class) or component_class == '':
            lam = 0.0
            lambdas.append(lam)
            component_lambdas[reference] = (lam, str(component_class))
            # Warning suppressed for MC simulations to avoid spam
            continue

        try:
            # Transistors
            if component_class in ['Low Power transistor (8.4)', 'Power Transistor (8.5)']:
                transistor_type = row.get('Transistor type', '')
                typ1 = "MOS" if 'MOS' in str(transistor_type) else "Bipolar"
                typ2 = "low" if component_class == 'Low Power transistor (8.4)' else "not low"

                if pd.isna(row.get('Temperature_Junction')):
                    lam = 0.0
                else:
                    if not pd.isna(row.get('LamB_sampled')):
                        lb = row['LamB_sampled']
                    else:
                        lb = rm.l_b(row.get('Table 18'))

                    vce_max = 0 if pd.isna(row.get('Max repetitive VCE')) else row['Max repetitive VCE']
                    vce_min = 1 if pd.isna(row.get('Min specified VCE')) else row['Min specified VCE']
                    vds_max = 0 if pd.isna(row.get('Max applied VDS')) else row['Max applied VDS']
                    vds_min = 1 if pd.isna(row.get('Min specified VDS')) else row['Min specified VDS']
                    vgs_max = 0 if pd.isna(row.get('Max applied VGS')) else row['Max applied VGS']
                    vgs_min = 1 if pd.isna(row.get('Min specified VGS')) else row['Min specified VGS']

                    lam = rm.lambda_transistors(
                        ni, row['Temperature_Junction'], typ1, typ2, dt,
                        lb, pi_i, leos,
                        vce_max, vce_min, vds_max, vds_min, vgs_max, vgs_min
                    )

            # Capacitors
            elif component_class == 'Ceramic Capacitor (10.3)':
                if pd.isna(row.get('Temperature_Ambiant')):
                    lam = 0.0
                else:
                    lam = rm.lambda_capacitors(ni, row['Temperature_Ambiant'], dt, "dielectrique")

            elif component_class == 'Tantlum Capacitor (10.4)':
                if pd.isna(row.get('Temperature_Ambiant')):
                    lam = 0.0
                else:
                    lam = rm.lambda_capacitors(ni, row['Temperature_Ambiant'], dt, "tantlum")

            # Resistors
            elif component_class == 'Resistor (11.1)':
                required = {
                    'Temperature_Ambiant': row.get('Temperature_Ambiant'),
                    'Operating_Power': row.get('Operating_Power'),
                    'Rated_Power': row.get('Rated_Power')
                }
                missing = [name for name, val in required.items() if pd.isna(val)]

                if missing:
                    lam = 0.0
                else:
                    lam = rm.lambda_resistors(
                        row['Temperature_Ambiant'],
                        row['Operating_Power'],
                        row['Rated_Power'],
                        dt, ni
                    )

            # Inductors
            elif component_class == 'Inductor (12)':
                required = {
                    'Temperature_Ambiant': row.get('Temperature_Ambiant'),
                    'Power loss': row.get('Power loss'),
                    'Radiating surface': row.get('Radiating surface')
                }
                missing = [name for name, val in required.items() if pd.isna(val)]

                if missing:
                    lam = 0.0
                else:
                    surface_str = str(row['Radiating surface'])
                    try:
                        if 'x' in surface_str:
                            parts = surface_str.split('x')
                            w = float(parts[0].strip())
                            h = float(parts[1].strip())
                            sur = (w / 100) * (h / 100)
                        else:
                            sur = 0.0132
                    except:
                        sur = 0.0132

                    lam = rm.lambda_inductors(
                        "inductor",
                        row.get("Inductor type", "Power Inductor"),
                        ni, dt,
                        row['Temperature_Ambiant'],
                        row['Power loss'],
                        sur
                    )

            # Converters
            elif component_class == 'Converter <10W (19.6)':
                lam = rm.lambda_converters("W<10", ni, dt)

            elif component_class == 'Converter >10W (19.6)':
                lam = rm.lambda_converters("W>10", ni, dt)

            # Diodes
            elif component_class in ['Low power diode (8.2)', 'Power diodes (8.3)']:
                required = {
                    'diode_type': row.get('diode_type'),
                    'Temperature_Junction': row.get('Temperature_Junction')
                }
                missing = [name for name, val in required.items() if pd.isna(val)]

                if missing:
                    lam = 0.0
                else:
                    if not pd.isna(row.get('LamB_sampled')):
                        lb = row['LamB_sampled']
                    else:
                        lb = rm.l_b(row.get('Table 18'))

                    lam = rm.lambda_diode(
                        row['diode_type'],
                        row['Temperature_Junction'],
                        ni, dt, lb,
                        pi_i, leos,
                        component_class
                    )

            # Primary batteries
            elif component_class == 'Primary batteries (19.1)':
                lam = rm.lambda_primary(component_class)

            # Integrated circuits
            elif component_class == 'Integrated Circuit (7)':
                required_params = {
                    'Construction Date': row.get('Construction Date'),
                    'Temperature_Junction': row.get('Temperature_Junction'),
                    'alpha_s': row.get('alpha_s'),
                    'alpha_c': row.get('alpha_c'),
                    'Table 16': row.get('Table 16'),
                    'Table 17a': row.get('Table 17a')
                }
                missing = [name for name, val in required_params.items() if pd.isna(val)]

                if missing:
                    lam = 0.0
                else:
                    lam = rm.lambda_int(
                        row['Construction Date'],
                        row['Temperature_Junction'],
                        row['alpha_s'],
                        row['alpha_c'],
                        ni, dt,
                        row['Table 16'],
                        row['Table 17a'],
                        row.get('Lam3', 1.3)
                    )

            else:
                # Unknown component class - lambda set to 0
                lam = 0.0

        except Exception:
            # Calculation error - lambda set to 0 as fallback
            lam = 0.0

        lam = lam if lam is not None else 0.0
        lambdas.append(lam)
        component_lambdas[reference] = (lam, str(component_class))

    # Series combination - only include valid lambdas
    valid_lambdas = [lam for lam in lambdas if lam > 0]
    lambda_total = sum(valid_lambdas)
    R_block = math.exp(-lambda_total * t_mission)

    return lambda_total, R_block, component_lambdas


def run_single_simulation(df: pd.DataFrame, matching_sheets: List[str],
                          rng: np.random.Generator
                          ) -> Tuple[float, float, Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Run a single Monte Carlo simulation iteration.

    Samples random parameters from their distributions, calculates reliability
    for each block, and combines them in series for system reliability.

    Args:
        df: DataFrame containing component data.
        matching_sheets: List of sheet names (blocks) to process.
        rng: NumPy random generator for reproducible sampling.

    Returns:
        tuple: A tuple containing:
            - lambda_total (float): Total system failure rate.
            - R_total (float): Total system reliability (series combination).
            - block_lambdas (dict): {sheet_name: lambda_value} for each block.
            - all_component_lambdas (dict): {sheet_name: {reference: lambda_value}}
              containing lambda values for all components in each block.
    """
    # Sample parameters
    df_sampled = sample_parameters(df, rng)

    # Calculate reliability for each block
    block_lambdas = {}
    block_reliabilities = []
    all_component_lambdas = {}

    for sheet in matching_sheets:
        lam, R, comp_lambdas = calculate_block_reliability_mc(df_sampled, sheet)
        block_lambdas[sheet] = lam
        block_reliabilities.append(R)
        # Store only lambda values for components
        all_component_lambdas[sheet] = {ref: val[0] for ref, val in comp_lambdas.items()}

    # Series combination
    lambda_total = sum(block_lambdas.values())
    R_total = rm.series_reliability(block_reliabilities)

    return lambda_total, R_total, block_lambdas, all_component_lambdas


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_component_details(component_stats: Dict[str, Dict], component_classes: Dict[str, str],
                            t_mission: float = rm.T_MISSION):
    """
    Print a formatted table showing Monte Carlo statistics for each component.

    Displays lambda (failure rate) and reliability statistics including mean,
    standard deviation for each component, color-coded by reliability level.

    Args:
        component_stats: Dictionary mapping component reference to statistics dict
            containing keys: 'mean', 'std', 'min', 'max', 'R_mean', 'R_std'.
        component_classes: Dictionary mapping component reference to class name.
        t_mission: Mission duration in hours for reliability calculation.

    Returns:
        None: Prints the table to stdout.
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*110}")
    print(f"COMPONENT BREAKDOWN (Monte Carlo Statistics)")
    print(f"{'='*110}{Colors.ENDC}\n")

    # Header
    print(f"{Colors.BOLD}{'Ref':<10} {'Class':<30} {'Lambda Mean':>14} {'Lambda Std':>14} {'R Mean':>12} {'R Std':>12}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*110}{Colors.ENDC}")

    total_lambda_mean = 0.0
    valid_count = 0

    for ref in sorted(component_stats.keys()):
        stats = component_stats[ref]
        comp_class = component_classes.get(ref, 'Unknown')

        # Truncate class name if too long
        if len(comp_class) > 28:
            comp_class = comp_class[:25] + '...'

        mean_lam = stats['mean']
        std_lam = stats['std']
        mean_R = stats.get('R_mean', 0)
        std_R = stats.get('R_std', 0)

        if mean_lam > 0:
            valid_count += 1
            total_lambda_mean += mean_lam

            # Color code by reliability
            if mean_R > 0.999:
                color = Colors.GREEN
            elif mean_R > 0.99:
                color = Colors.CYAN
            elif mean_R > 0.95:
                color = Colors.YELLOW
            else:
                color = Colors.RED

            print(f"{ref:<10} {comp_class:<30} {mean_lam:>14.6e} {std_lam:>14.6e} {color}{mean_R:>12.6f}{Colors.ENDC} {std_R:>12.6f}")
        else:
            print(f"{ref:<10} {comp_class:<30} {'N/A':>14} {'--':>14} {Colors.RED}{'N/A':>12}{Colors.ENDC} {'--':>12}")

    print(f"{Colors.CYAN}{'-'*110}{Colors.ENDC}")

    # Block summary
    total_R_mean = math.exp(-total_lambda_mean * t_mission) if total_lambda_mean > 0 else 1.0

    if total_R_mean > 0.99:
        color = Colors.GREEN
    elif total_R_mean > 0.95:
        color = Colors.CYAN
    elif total_R_mean > 0.90:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    print(f"\n{Colors.BOLD}Block Summary:{Colors.ENDC}")
    print(f"  Valid components: {valid_count}/{len(component_stats)}")
    print(f"  Total lambda (mean): {total_lambda_mean:.6e} failures/hour")
    print(f"  Block reliability (mean): {color}{total_R_mean:.6f}{Colors.ENDC}")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_lambda_distributions(lambda_results: np.ndarray, R_results: np.ndarray,
                               block_lambda_results: Optional[Dict[str, np.ndarray]] = None,
                               save_path: Optional[str] = None):
    """
    Plot histograms of lambda and reliability distributions from Monte Carlo results.

    Creates a multi-panel figure showing the system reliability distribution and
    individual block lambda distributions with mean and percentile indicators.

    Args:
        lambda_results: Array of total system lambda values from each simulation.
        R_results: Array of system reliability values from each simulation.
        block_lambda_results: Optional dictionary mapping block names to arrays
            of lambda values for block-level distribution plots.
        save_path: Optional file path to save the figure.

    Returns:
        None: Displays the plot and optionally saves to file.
    """
    # Calculate number of subplots needed
    n_blocks = len(block_lambda_results) if block_lambda_results else 0
    n_rows = 2 + (n_blocks + 1) // 2

    fig = plt.figure(figsize=(14, 4 * n_rows))



    # Main reliability distribution
    ax2 = fig.add_subplot(n_rows, 2, 2)
    ax2.hist(R_results, bins=50, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
    ax2.axvline(np.mean(R_results), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(R_results):.6f}')
    ax2.axvline(np.percentile(R_results, 5), color='orange', linestyle=':', linewidth=2,
                label=f'5th percentile: {np.percentile(R_results, 5):.6f}')
    ax2.axvline(np.percentile(R_results, 95), color='orange', linestyle=':', linewidth=2,
                label=f'95th percentile: {np.percentile(R_results, 95):.6f}')
    ax2.set_xlabel('Reliability')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of System Reliability')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Block-level lambda distributions
    if block_lambda_results:
        colors = plt.cm.tab10(np.linspace(0, 1, len(block_lambda_results)))
        for i, (block_name, block_lambdas) in enumerate(block_lambda_results.items()):
            ax = fig.add_subplot(n_rows, 2, 3 + i)
            display_name = block_name.split('/')[-2] if block_name.endswith('/') else block_name.split('/')[-1]
            ax.hist(block_lambdas * 1e9, bins=30, density=True, alpha=0.7,
                    color=colors[i % len(colors)], edgecolor='black')
            ax.axvline(np.mean(block_lambdas) * 1e9, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Lambda (x10^-9 failures/hour)')
            ax.set_ylabel('Density')
            ax.set_title(f'Lambda Distribution: {display_name}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_success(f"Figure saved to {save_path}")

    plt.show()


def plot_component_lambda_distributions(component_lambda_results: Dict[str, np.ndarray],
                                         save_path: Optional[str] = None):
    """
    Plot histograms of lambda distributions for each component.

    Creates a grid of histograms showing the distribution of failure rates
    for each component across Monte Carlo simulations. Components with
    zero mean lambda are filtered out.

    Args:
        component_lambda_results: Dictionary mapping component reference to
            an array of lambda values from Monte Carlo simulations.
        save_path: Optional file path to save the figure.

    Returns:
        None: Displays the plot and optionally saves to file.
    """
    # Filter components with valid data (mean lambda > 0)
    valid_components = {ref: lam_arr for ref, lam_arr in component_lambda_results.items()
                        if np.mean(lam_arr) > 0}

    if not valid_components:
        print_warning("No components with valid lambda data to plot")
        return

    n_components = len(valid_components)
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_components == 1:
        axes = np.array([[axes]])

    colors = plt.cm.tab20(np.linspace(0, 1, n_components))

    for i, (ref, lam_arr) in enumerate(sorted(valid_components.items())):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        # Convert to 10^-9 for better readability
        lam_arr_scaled = lam_arr * 1e9

        ax.hist(lam_arr_scaled, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        ax.axvline(np.mean(lam_arr_scaled), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(lam_arr_scaled):.4f}')
        ax.axvline(np.percentile(lam_arr_scaled, 5), color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(np.percentile(lam_arr_scaled, 95), color='orange', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Lambda (x10^-9 /h)')
        ax.set_ylabel('Count')
        ax.set_title(f'{ref} (std={np.std(lam_arr):.2e})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_components, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Lambda Distributions by Component', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_success(f"Figure saved to {save_path}")

    plt.show()


def plot_component_reliability_distributions(component_R_results: Dict[str, np.ndarray],
                                              save_path: Optional[str] = None):
    """
    Plot histograms of reliability distributions for each component.

    Creates a grid of histograms showing the distribution of reliability values
    for each component across Monte Carlo simulations. Components with
    zero mean reliability are filtered out.

    Args:
        component_R_results: Dictionary mapping component reference to
            an array of reliability values from Monte Carlo simulations.
        save_path: Optional file path to save the figure.

    Returns:
        None: Displays the plot and optionally saves to file.
    """
    # Filter components with valid data (mean R > 0)
    valid_components = {ref: R_arr for ref, R_arr in component_R_results.items()
                        if np.mean(R_arr) > 0}

    if not valid_components:
        print_warning("No components with valid data to plot")
        return

    n_components = len(valid_components)
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_components == 1:
        axes = np.array([[axes]])

    colors = plt.cm.tab20(np.linspace(0, 1, n_components))

    for i, (ref, R_arr) in enumerate(sorted(valid_components.items())):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        ax.hist(R_arr, bins=30,  alpha=0.7, color=colors[i], edgecolor='black')
        ax.axvline(np.mean(R_arr), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(R_arr):.6f}')
        ax.axvline(np.percentile(R_arr, 5), color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(np.percentile(R_arr, 95), color='orange', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Reliability')
        ax.set_ylabel('Density')
        ax.set_title(f'{ref} (std={np.std(R_arr):.2e})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_components, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Reliability Distributions by Component', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print_success(f"Figure saved to {save_path}")

    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_monte_carlo_analysis(excel_file: str, sheet_name: str,
                              n_simulations: int = 100,
                              process_subblocks: bool = True,
                              seed: Optional[int] = None):
    """
    Execute complete Monte Carlo reliability analysis workflow.

    Performs Monte Carlo simulation to quantify uncertainty in reliability
    calculations due to parameter variability. Samples parameters from their
    specified distributions, calculates reliability for each iteration, and
    provides statistical summaries and visualizations.

    Args:
        excel_file: Path to Excel file containing component data with columns
            including 'Sheet', 'Reference', 'Class', and various parameters.
        sheet_name: Name of the block/sheet to analyze. If process_subblocks
            is True, analyzes all sheets starting with this prefix.
        n_simulations: Number of Monte Carlo iterations to run.
        process_subblocks: If True, processes all sub-blocks starting with
            sheet_name as a system in series. If False, processes only the
            exact sheet_name.
        seed: Optional random seed for reproducible results.

    Returns:
        dict: Dictionary containing:
            - 'lambda_results': Array of total lambda values per simulation.
            - 'R_results': Array of reliability values per simulation.
            - 'block_lambda_results': Dict of lambda arrays per block.
            - 'component_stats': Component-level statistics (single block only).
            - 'component_R_results': Component reliability arrays (single block).
            - 'statistics': Summary statistics dict with keys 'lambda_mean',
              'lambda_std', 'R_mean', 'R_std', 'R_ci_90'.
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"MONTE CARLO RELIABILITY ANALYSIS")
    print(f"{'='*70}{Colors.ENDC}\n")

    # Load Excel file
    print_info(f"Loading data from {excel_file}...")
    df = pd.read_excel(excel_file)

    if 'Sheet' not in df.columns:
        raise ValueError("Excel file must have a 'Sheet' column")

    # Get matching sheets
    all_sheets = df['Sheet'].unique()
    if process_subblocks:
        matching_sheets = sorted([s for s in all_sheets if s.startswith(sheet_name)])
    else:
        matching_sheets = [sheet_name] if sheet_name in all_sheets else []

    if not matching_sheets:
        print(f"{Colors.RED}WARNING: No sheets found matching '{sheet_name}'{Colors.ENDC}")
        return

    # Determine if single block or group
    is_single_block = len(matching_sheets) == 1

    print_success(f"Found {len(matching_sheets)} matching block(s):")
    for sheet in matching_sheets:
        comp_count = len(df[df['Sheet'] == sheet])
        print(f"  {Colors.CYAN}- {sheet} ({comp_count} components){Colors.ENDC}")

    print_info(f"\nRunning {n_simulations} Monte Carlo simulations...")

    # Initialize random generator
    rng = np.random.default_rng(seed)

    # Storage for results
    lambda_results = np.zeros(n_simulations)
    R_results = np.zeros(n_simulations)
    block_lambda_results = {sheet: np.zeros(n_simulations) for sheet in matching_sheets}
    block_R_results = {sheet: np.zeros(n_simulations) for sheet in matching_sheets}

    # For single block: track component-level lambdas and reliabilities
    if is_single_block:
        block_df = df[df['Sheet'] == matching_sheets[0]]
        component_refs = block_df['Reference'].tolist()
        component_classes = dict(zip(block_df['Reference'], block_df['Class'].fillna('')))
        component_lambda_results = {ref: [] for ref in component_refs}
        component_R_results = {ref: [] for ref in component_refs}
    else:
        component_lambda_results = None
        component_R_results = None
        component_classes = None

    # Run Monte Carlo simulations
    for i in range(n_simulations):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i+1}/{n_simulations} simulations...", end='\r')

        lambda_total, R_total, block_lambdas, all_comp_lambdas = run_single_simulation(
            df, matching_sheets, rng
        )

        lambda_results[i] = lambda_total
        R_results[i] = R_total

        for sheet, lam in block_lambdas.items():
            block_lambda_results[sheet][i] = lam
            # Calculate R for this block
            block_R_results[sheet][i] = math.exp(-lam * rm.T_MISSION) if lam > 0 else 1.0

        # Store component lambdas and reliabilities for single block
        if is_single_block:
            sheet = matching_sheets[0]
            for ref, lam_val in all_comp_lambdas[sheet].items():
                if ref in component_lambda_results:
                    component_lambda_results[ref].append(lam_val)
                    # Calculate reliability for this component
                    R_comp = math.exp(-lam_val * rm.T_MISSION) if lam_val > 0 else 1.0
                    component_R_results[ref].append(R_comp)

    print(f"  {Colors.GREEN}✓ Completed {n_simulations} simulations{Colors.ENDC}" + " " * 30)

    # For single block: print component details with R stats
    if is_single_block:
        # Calculate component statistics
        component_stats = {}
        for ref, lam_list in component_lambda_results.items():
            lam_array = np.array(lam_list)
            R_array = np.array(component_R_results[ref])
            component_stats[ref] = {
                'mean': np.mean(lam_array),
                'std': np.std(lam_array),
                'min': np.min(lam_array),
                'max': np.max(lam_array),
                'R_mean': np.mean(R_array),
                'R_std': np.std(R_array)
            }

        print_component_details(component_stats, component_classes)

    # Calculate and print statistics
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"MONTE CARLO RESULTS")
    print(f"{'='*70}{Colors.ENDC}\n")

    # Lambda statistics
    print(f"{Colors.BOLD}Total Failure Rate (Lambda):{Colors.ENDC}")
    print(f"  Mean:               {np.mean(lambda_results):.6e} failures/hour")
    print(f"  Std Dev:            {np.std(lambda_results):.6e} failures/hour")
    print(f"  5th Percentile:     {np.percentile(lambda_results, 5):.6e} failures/hour")
    print(f"  95th Percentile:    {np.percentile(lambda_results, 95):.6e} failures/hour")
    print(f"  Min:                {np.min(lambda_results):.6e} failures/hour")
    print(f"  Max:                {np.max(lambda_results):.6e} failures/hour")

    # Reliability statistics
    print(f"\n{Colors.BOLD}System Reliability:{Colors.ENDC}")
    print(f"  Mean:               {np.mean(R_results):.6f}")
    print(f"  Std Dev:            {np.std(R_results):.6f}")
    print(f"  5th Percentile:     {np.percentile(R_results, 5):.6f}")
    print(f"  95th Percentile:    {np.percentile(R_results, 95):.6f}")
    print(f"  Min:                {np.min(R_results):.6f}")
    print(f"  Max:                {np.max(R_results):.6f}")

    # 90% Confidence Interval
    ci_lower = np.percentile(R_results, 5)
    ci_upper = np.percentile(R_results, 95)
    print(f"\n{Colors.BOLD}90% Confidence Interval for Reliability:{Colors.ENDC}")
    print(f"  [{ci_lower:.6f}, {ci_upper:.6f}]")

    # Block-level statistics (for groups)
    if not is_single_block:
        print(f"\n{Colors.BOLD}Block-Level Statistics:{Colors.ENDC}")
        print(f"{'Block':<40} {'Lambda Mean':>14} {'Lambda Std':>14} {'R Mean':>12} {'R Std':>12}")
        print(f"{Colors.CYAN}{'-'*95}{Colors.ENDC}")

        for sheet in matching_sheets:
            block_lam = block_lambda_results[sheet]
            block_R = block_R_results[sheet]
            display_name = sheet if len(sheet) < 40 else '...' + sheet[-37:]
            print(f"{display_name:<40} {np.mean(block_lam):>14.6e} {np.std(block_lam):>14.6e} {np.mean(block_R):>12.6f} {np.std(block_R):>12.6f}")

        # Total series
        print(f"{Colors.CYAN}{'-'*95}{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'SYSTEM TOTAL (Series)':<40} {np.mean(lambda_results):>14.6e} {np.std(lambda_results):>14.6e} {np.mean(R_results):>12.6f} {np.std(R_results):>12.6f}{Colors.ENDC}")

    print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}")

    # Mission parameters
    years = rm.T_MISSION / (365 * 24)
    print(f"\n{Colors.BOLD}Mission Parameters:{Colors.ENDC}")
    print(f"  Duration: {rm.T_MISSION:,} hours ({years:.2f} years)")
    print(f"  Cycles per year: {rm.NI:,}")
    print(f"  Temperature cycle amplitude: {rm.DT}°C")

    # Plot distributions
    print_info("\nGenerating distribution plots...")
    plot_lambda_distributions(lambda_results, R_results, block_lambda_results)

    # For single block: also plot component distributions
    if is_single_block:
        # Convert lists to arrays
        component_lambda_arrays = {ref: np.array(lam_list) for ref, lam_list in component_lambda_results.items()}
        component_R_arrays = {ref: np.array(R_list) for ref, R_list in component_R_results.items()}

        print_info("Generating component lambda distribution plots...")
        plot_component_lambda_distributions(component_lambda_arrays)

        print_info("Generating component reliability distribution plots...")
        plot_component_reliability_distributions(component_R_arrays)

    print("\nAnalysis complete")

    return {
        'lambda_results': lambda_results,
        'R_results': R_results,
        'block_lambda_results': block_lambda_results,
        'component_stats': component_stats if is_single_block else None,
        'component_R_results': component_R_results if is_single_block else None,
        'statistics': {
            'lambda_mean': np.mean(lambda_results),
            'lambda_std': np.std(lambda_results),
            'R_mean': np.mean(R_results),
            'R_std': np.std(R_results),
            'R_ci_90': (ci_lower, ci_upper)
        }
    }


