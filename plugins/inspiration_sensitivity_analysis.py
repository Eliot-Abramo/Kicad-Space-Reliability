"""
Task 2: Sensitivity Analysis for Reliability
==============================================

Sobol sensitivity analysis using the Pick-Freeze method to quantify the
contribution of each component's lambda uncertainty to the overall reliability
variance.

Pick-Freeze Formula (symmetrized estimator):
    Ŝᵢ = [E(Y·Y'ᵢ) - (E(Y+Y'ᵢ)/2)²] / [E(Y²+Y'ᵢ²)/2 - (E(Y+Y'ᵢ)/2)²]

    This formula estimates the first-order Sobol index by measuring how much
    of the output variance is explained by input variable i alone.

Analytical Variance Formula:
    σ²_T = Var((Y - Ȳ)(Y^pi - Ȳ) - Ŝᵢ/2*((Y - Ȳ)² + (Y^pi - Ȳ)²)) / (Var(Y))²

    This formula provides the asymptotic variance of the Sobol estimator,
    enabling construction of confidence intervals without bootstrap.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import reliability_math as rm

# Import distributions from task1
from task1_monte_carlo import (
     sample_parameters, calculate_block_reliability_mc
)


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
# SOBOL INDICES - PICK-FREEZE METHOD
# ============================================================================

def sobol_pick_freeze(f_func: Callable, X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate first-order Sobol indices using the Pick-Freeze method.

    Computes first-order Sobol indices that measure the contribution of each
    input variable to the output variance. Uses the symmetrized Pick-Freeze
    estimator for improved accuracy and also computes the analytical standard
    deviation for each index.

    Pick-Freeze Formula (symmetrized):
        Ŝᵢ = [E(Y·Y'ᵢ) - (E(Y+Y'ᵢ)/2)²] / [E(Y²+Y'ᵢ²)/2 - (E(Y+Y'ᵢ)/2)²]

        The numerator estimates Cov(Y, Y'_i) where Y'_i is the output when only
        input i is shared between samples. The denominator estimates Var(Y).

    Analytical Variance Formula:
        σ²_T = Var(T_i) / (Var(Y))²
        where T_i = (Y - Ȳ)(Y^pi - Ȳ) - Ŝᵢ/2 * ((Y - Ȳ)² + (Y^pi - Ȳ)²)

        This provides the asymptotic variance of the estimator.

    Args:
        f_func: Function that takes input array X of shape (N, d) and returns
            output array Y of shape (N,).
        X1: First independent sample array of shape (N, d) where N is the number
            of samples and d is the number of input dimensions.
        X2: Second independent sample array of shape (N, d).

    Returns:
        tuple: A tuple containing:
            - S (np.ndarray): Array of shape (d,) with first-order Sobol indices.
            - S_std (np.ndarray): Array of shape (d,) with standard deviation
              of each Sobol index estimator.
    """
    N, d = X1.shape

    # Compute Y for the first sample
    Y = f_func(X1)
    Y_mean = np.mean(Y)
    Y_centered = Y - Y_mean

    S = np.zeros(d)
    S_std = np.zeros(d)

    for i in range(d):
        # Create X1_prime: X2 with column i replaced by X1's column i
        X1_prime = X2.copy()
        X1_prime[:, i] = X1[:, i]

        # Compute Y' for the hybrid sample
        Y_pi = f_func(X1_prime)
        Y_pi_centered = Y_pi - Y_mean

        # Numerator: Cov(Y, Y'_i)
        term1 = np.mean(Y * Y_pi)
        term2 = 0.25 * (np.mean(Y + Y_pi))**2
        numerator = term1 - term2

        # Denominator: Var(Y) estimated symmetrically
        denominator = 0.5 * (np.mean(Y**2) + np.mean(Y_pi**2)) - term2

        
        S[i] = numerator / denominator

        # Analytical variance formula:
        # σ²_T = Var(T_i) / (Var(Y))²
        # where T_i = (Y - Ȳ)(Y^pi - Ȳ) - Ŝᵢ/2 * ((Y - Ȳ)² + (Y^pi - Ȳ)²)
        T_i = Y_centered * Y_pi_centered - (S[i] / 2) * (Y_centered**2 + Y_pi_centered**2)
        var_T = np.var(T_i, ddof=1)
        var_Y = denominator  # Already computed

        sigma_squared = var_T / (var_Y**2)
        S_std[i] = np.sqrt(sigma_squared)


    return S, S_std


# ============================================================================
# RELIABILITY FUNCTIONS
# ============================================================================

def reliability_function(lambdas: np.ndarray, t_mission: float) -> np.ndarray:
    """
    Compute system reliability from component failure rates.

    Uses the exponential reliability model for series systems:
        R = exp(-Σλᵢ * t)

    For components in series, the system fails if any component fails, so the
    total failure rate is the sum of individual failure rates.

    Args:
        lambdas: Array of shape (N, n_components) containing failure rates
            for each component across N Monte Carlo samples.
        t_mission: Mission duration in hours.

    Returns:
        np.ndarray: Array of shape (N,) containing reliability values for
            each sample, where R[i] = exp(-sum(lambdas[i, :]) * t_mission).
    """
    lambda_total = np.sum(lambdas, axis=1)
    return np.exp(-lambda_total * t_mission)


def make_reliability_func(t_mission: float) -> Callable:
    """
    Create a reliability function with fixed mission time.

    Factory function that returns a callable with the mission time bound,
    suitable for use with the Sobol analysis functions.

    Args:
        t_mission: Mission duration in hours to be fixed in the returned function.

    Returns:
        Callable: A function that takes lambdas array of shape (N, d) and
            returns reliability array R of shape (N,).
    """
    def f(lambdas):
        return reliability_function(lambdas, t_mission)
    return f


# ============================================================================
# SAMPLING FUNCTIONS FOR SENSITIVITY ANALYSIS
# ============================================================================

def sample_lambdas_for_block(df: pd.DataFrame, sheet_name: str,
                              rng: np.random.Generator, n_samples: int,
                              n_sets: int = 1
                              ) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate lambda samples for all components in a single block.

    Performs Monte Carlo sampling of component parameters and computes failure
    rates for each sample. Creates multiple independent sample sets as needed
    for the Pick-Freeze method.

    Args:
        df: DataFrame containing component data with columns like 'Reference',
            'Class', and various parameters.
        sheet_name: Name of the block/sheet to sample from.
        rng: NumPy random generator for reproducible sampling.
        n_samples: Number of Monte Carlo samples to generate per set.
        n_sets: Number of independent sample sets (2 for Pick-Freeze method).

    Returns:
        tuple: A tuple containing:
            - lambdas_list (list): List of n_sets arrays, each of shape
              (n_samples, n_components) containing sampled lambda values.
            - references (list): List of component reference strings.
    """
    block_df = df[df['Sheet'] == sheet_name].copy()

    if block_df.empty:
        print(f"{Colors.YELLOW}WARNING: No components found for sheet '{sheet_name}'{Colors.ENDC}")
        return [np.array([]) for _ in range(n_sets)], []

    references = block_df['Reference'].tolist()
    n_components = len(references)

    lambdas_list = [np.zeros((n_samples, n_components)) for _ in range(n_sets)]

    total_samples = n_samples * n_sets
    for i in range(total_samples):
        if (i + 1) % 100 == 0:
            print(f"  Sampling progress: {i+1}/{total_samples}...", end='\r')

        set_idx = i // n_samples
        sample_idx = i % n_samples

        # Sample parameters for this iteration
        df_sampled = sample_parameters(df, rng)

        # Calculate lambda for each component
        _, _, comp_lambdas = calculate_block_reliability_mc(df_sampled, sheet_name)

        # Store lambdas in order
        for j, ref in enumerate(references):
            if ref in comp_lambdas:
                lambdas_list[set_idx][sample_idx, j] = comp_lambdas[ref][0]

    print(" " * 50, end='\r')
    return lambdas_list, references


def sample_lambdas_for_system(df: pd.DataFrame, sheets: List[str],
                               rng: np.random.Generator, n_samples: int,
                               n_sets: int = 1
                               ) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Generate lambda samples for all components across multiple blocks.

    Performs Monte Carlo sampling of component parameters for a complete system
    consisting of multiple blocks. Creates a combined array with all components
    from all blocks for system-level sensitivity analysis.

    Args:
        df: DataFrame containing component data with columns like 'Reference',
            'Class', 'Sheet', and various parameters.
        sheets: List of sheet names (blocks) to include in the system.
        rng: NumPy random generator for reproducible sampling.
        n_samples: Number of Monte Carlo samples to generate per set.
        n_sets: Number of independent sample sets (2 for Pick-Freeze method).

    Returns:
        tuple: A tuple containing:
            - lambdas_list (list): List of n_sets arrays, each of shape
              (n_samples, n_total_components) containing sampled lambda values.
            - references (list): List of all component reference strings.
            - sheet_labels (list): List indicating which sheet/block each
              component belongs to, aligned with references.
    """
    # First, collect all component references and their sheets
    all_references = []
    sheet_labels = []

    for sheet in sheets:
        block_df = df[df['Sheet'] == sheet]
        refs = block_df['Reference'].tolist()
        all_references.extend(refs)
        sheet_labels.extend([sheet] * len(refs))

    n_total_components = len(all_references)
    lambdas_list = [np.zeros((n_samples, n_total_components)) for _ in range(n_sets)]

    total_samples = n_samples * n_sets
    for i in range(total_samples):
        if (i + 1) % 100 == 0:
            print(f"  Sampling progress: {i+1}/{total_samples}...", end='\r')

        set_idx = i // n_samples
        sample_idx = i % n_samples

        # Sample parameters for this iteration
        df_sampled = sample_parameters(df, rng)

        # Calculate lambda for each block and collect component lambdas
        idx = 0
        for sheet in sheets:
            _, _, comp_lambdas = calculate_block_reliability_mc(df_sampled, sheet)

            block_df = df[df['Sheet'] == sheet]
            for ref in block_df['Reference'].tolist():
                if ref in comp_lambdas:
                    lambdas_list[set_idx][sample_idx, idx] = comp_lambdas[ref][0]
                idx += 1

    print(" " * 50, end='\r')
    return lambdas_list, all_references, sheet_labels


# ============================================================================
# SOBOL ANALYSIS FUNCTIONS
# ============================================================================

def sobol_analysis_block(df: pd.DataFrame, sheet_name: str,
                          n_samples: int = 1000,
                          t_mission: float = rm.T_MISSION,
                          seed: Optional[int] = None
                          ) -> Dict:
    """
    Perform Sobol sensitivity analysis for a single block using Pick-Freeze method.

    Computes first-order Sobol indices to quantify each component's contribution
    to the reliability variance. Uses analytical variance estimation for 95%
    confidence intervals.

    Args:
        df: DataFrame containing component data with 'Sheet', 'Reference', 'Class'
            columns and various parameter columns.
        sheet_name: Name of the block/sheet to analyze.
        n_samples: Number of Monte Carlo samples for the Pick-Freeze estimation.
        t_mission: Mission duration in hours for reliability calculation.
        seed: Optional random seed for reproducible results.

    Returns:
        dict: Dictionary containing:
            - 'sobol_indices': Array of Sobol indices for each component.
            - 'sobol_std': Array of standard deviations for each index.
            - 'sobol_std_normalized': Array of σ/√N for each index.
            - 'ci_low', 'ci_high': 95% confidence interval bounds.
            - 'references': List of component references.
            - 'valid_references': List of components with non-zero variance.
            - 'valid_mask': Boolean mask for valid components.
            - 'classes': Dict mapping reference to class name.
            - 'lambdas': Sampled lambda values array.
            - 'reliability': Computed reliability values array.
            - 'n_samples': Number of samples used.
            - 't_mission': Mission time used.
        Returns empty dict if no valid components found.
    """
    print_info(f"Running Sobol analysis for block: {sheet_name}")
    print_info(f"Method: Pick-Freeze")
    print_info(f"Number of samples: {n_samples}")

    rng = np.random.default_rng(seed)

    # Pick-Freeze needs 2 sample sets
    n_sets = 2

    # Sample lambdas
    print_info("Sampling lambda values...")
    lambdas_list, references = sample_lambdas_for_block(df, sheet_name, rng, n_samples, n_sets)

    if len(references) == 0:
        print(f"{Colors.RED}WARNING: No components found in block '{sheet_name}'{Colors.ENDC}")
        return {}

    print_success(f"Total components: {len(references)}")

    # Filter out components with zero variance
    X1 = lambdas_list[0]
    X2 = lambdas_list[1]
    variances = np.var(X1, axis=0)

    valid_mask = variances > 1e-25
    valid_indices = np.where(valid_mask)[0]

    print_info(f"Components with non-zero variance: {len(valid_indices)}/{len(references)}")

    if len(valid_indices) == 0:
        print(f"{Colors.RED}WARNING: No components with variance - cannot compute Sobol indices{Colors.ENDC}")
        return {}

    # Filter lambda matrices to only include components with variance
    X1_filtered = X1[:, valid_mask]
    X2_filtered = X2[:, valid_mask]

    # Map filtered indices back to original references
    valid_references = [references[i] for i in valid_indices]

    # Compute Sobol indices with analytical std
    print_info("Computing Sobol indices with analytical variance...")

    f_reliability = make_reliability_func(t_mission)
    S_filtered, S_std_filtered = sobol_pick_freeze(f_reliability, X1_filtered, X2_filtered)
    R = f_reliability(X1)
    lambdas = X1

    # Normalized std: σ / √N
    S_std_normalized_filtered = S_std_filtered / np.sqrt(n_samples)

    # 95% Confidence interval: S ± 1.96 * σ_normalized
    CI_low_filtered = S_filtered - 1.96 * S_std_normalized_filtered
    CI_low_filtered = np.maximum(CI_low_filtered, 0.0)  # Borne inf >= 0
    CI_high_filtered = S_filtered + 1.96 * S_std_normalized_filtered

    # Map Sobol indices back to full array (zeros for components without variance)
    S = np.zeros(len(references))
    S[valid_mask] = S_filtered

    S_std = np.zeros(len(references))
    S_std[valid_mask] = S_std_filtered

    S_std_normalized = np.zeros(len(references))
    S_std_normalized[valid_mask] = S_std_normalized_filtered

    CI_low = np.zeros(len(references))
    CI_low[valid_mask] = CI_low_filtered

    CI_high = np.zeros(len(references))
    CI_high[valid_mask] = CI_high_filtered

    # Get component classes
    block_df = df[df['Sheet'] == sheet_name]
    classes = dict(zip(block_df['Reference'], block_df['Class'].fillna('')))

    return {
        'sobol_indices': S,
        'sobol_std': S_std,
        'sobol_std_normalized': S_std_normalized,
        'ci_low': CI_low,
        'ci_high': CI_high,
        'references': references,
        'valid_references': valid_references,
        'valid_mask': valid_mask,
        'classes': classes,
        'lambdas': lambdas,
        'reliability': R,
        'n_samples': n_samples,
        't_mission': t_mission
    }


def sobol_analysis_system(df: pd.DataFrame, sheets: List[str],
                           n_samples: int = 1000,
                           t_mission: float = rm.T_MISSION,
                           seed: Optional[int] = None
                           ) -> Dict:
    """
    Perform Sobol sensitivity analysis for a complete system (multiple blocks).

    Computes first-order Sobol indices for all components across multiple blocks
    in a series system configuration. Quantifies each component's contribution
    to the total system reliability variance.

    Args:
        df: DataFrame containing component data with 'Sheet', 'Reference', 'Class'
            columns and various parameter columns.
        sheets: List of sheet names (blocks) to include in the system analysis.
        n_samples: Number of Monte Carlo samples for the Pick-Freeze estimation.
        t_mission: Mission duration in hours for reliability calculation.
        seed: Optional random seed for reproducible results.

    Returns:
        dict: Dictionary containing:
            - 'sobol_indices': Array of Sobol indices for each component.
            - 'sobol_std': Array of standard deviations for each index.
            - 'sobol_std_normalized': Array of σ/√N for each index.
            - 'ci_low', 'ci_high': 95% confidence interval bounds.
            - 'references': List of all component references.
            - 'valid_references': List of components with non-zero variance.
            - 'valid_mask': Boolean mask for valid components.
            - 'sheet_labels': List of sheet names for each component.
            - 'classes': Dict mapping reference to class name.
            - 'lambdas': Sampled lambda values array.
            - 'reliability': Computed reliability values array.
            - 'n_samples': Number of samples used.
            - 't_mission': Mission time used.
            - 'sheets': List of sheets analyzed.
        Returns empty dict if no valid components found.
    """
    print_info(f"Running Sobol analysis for system with {len(sheets)} blocks")
    print_info(f"Method: Pick-Freeze")
    print_info(f"Number of samples: {n_samples}")

    rng = np.random.default_rng(seed)

    # Pick-Freeze needs 2 sample sets
    n_sets = 2

    # Sample lambdas for all components
    print_info("Sampling lambda values for all components...")
    lambdas_list, references, sheet_labels = sample_lambdas_for_system(df, sheets, rng, n_samples, n_sets)

    if len(references) == 0:
        print(f"{Colors.RED}WARNING: No components found in the system{Colors.ENDC}")
        return {}

    print_success(f"Total components: {len(references)}")

    # Filter out components with zero variance
    X1 = lambdas_list[0]
    X2 = lambdas_list[1]
    variances = np.var(X1, axis=0)
    valid_mask = variances > 1e-25
    valid_indices = np.where(valid_mask)[0]

    print_info(f"Components with non-zero variance: {len(valid_indices)}/{len(references)}")

    if len(valid_indices) == 0:
        print(f"{Colors.RED}WARNING: No components with variance - cannot compute Sobol indices{Colors.ENDC}")
        return {}

    # Filter lambda matrices to only include components with variance
    X1_filtered = X1[:, valid_mask]
    X2_filtered = X2[:, valid_mask]

    # Map filtered indices back to original references
    valid_references = [references[i] for i in valid_indices]
    valid_sheet_labels = [sheet_labels[i] for i in valid_indices]

    # Compute Sobol indices with analytical std
    print_info("Computing Sobol indices with analytical variance...")

    f_reliability = make_reliability_func(t_mission)
    S_filtered, S_std_filtered = sobol_pick_freeze(f_reliability, X1_filtered, X2_filtered)
    R = f_reliability(X1)
    lambdas = X1

    # Normalized std: σ / √N
    S_std_normalized_filtered = S_std_filtered / np.sqrt(n_samples)

    # 95% Confidence interval: S ± 1.96 * σ_normalized
    CI_low_filtered = S_filtered - 1.96 * S_std_normalized_filtered
    CI_low_filtered = np.maximum(CI_low_filtered, 0.0)  # Borne inf >= 0
    CI_high_filtered = S_filtered + 1.96 * S_std_normalized_filtered

    # Map Sobol indices back to full array (zeros for components without variance)
    S = np.zeros(len(references))
    S[valid_mask] = S_filtered

    S_std = np.zeros(len(references))
    S_std[valid_mask] = S_std_filtered

    S_std_normalized = np.zeros(len(references))
    S_std_normalized[valid_mask] = S_std_normalized_filtered

    CI_low = np.zeros(len(references))
    CI_low[valid_mask] = CI_low_filtered

    CI_high = np.zeros(len(references))
    CI_high[valid_mask] = CI_high_filtered

    # Get component classes
    classes = {}
    for sheet in sheets:
        block_df = df[df['Sheet'] == sheet]
        classes.update(dict(zip(block_df['Reference'], block_df['Class'].fillna(''))))

    return {
        'sobol_indices': S,
        'sobol_std': S_std,
        'sobol_std_normalized': S_std_normalized,
        'ci_low': CI_low,
        'ci_high': CI_high,
        'references': references,
        'valid_references': valid_references,
        'valid_mask': valid_mask,
        'sheet_labels': sheet_labels,
        'classes': classes,
        'lambdas': lambdas,
        'reliability': R,
        'n_samples': n_samples,
        't_mission': t_mission,
        'sheets': sheets
    }


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_sobol_results(results: Dict, top_n: int = 20):
    """
    Print Sobol sensitivity analysis results in a formatted table.

    Displays a ranked table of components by Sobol index, showing the estimated
    index value, normalized standard deviation, and 95% confidence interval
    for each component.

    Args:
        results: Dictionary returned by sobol_analysis_block or sobol_analysis_system
            containing 'sobol_indices', 'sobol_std_normalized', 'ci_low', 'ci_high',
            'references', 'classes', and optionally 'sheet_labels'.
        top_n: Maximum number of components to display in the ranked table.

    Returns:
        None: Prints the formatted table to stdout.
    """
    S = results['sobol_indices']
    S_std_norm = results.get('sobol_std_normalized', np.zeros_like(S))
    CI_low = results.get('ci_low', np.zeros_like(S))
    CI_high = results.get('ci_high', np.zeros_like(S))
    refs = results['references']
    classes = results['classes']
    sheet_labels = results.get('sheet_labels', None)
    valid_refs = results.get('valid_references', refs)
    n_samples = results.get('n_samples', 0)

    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*120}")
    print(f"SOBOL SENSITIVITY INDICES (First-Order) - Method: Pick-Freeze")
    print(f"{'='*120}{Colors.ENDC}\n")

    print(f"{Colors.BOLD}Analysis info:{Colors.ENDC}")
    print(f"  Total components: {len(refs)}")
    print(f"  Components with variance: {len(valid_refs)}")
    print(f"  Number of samples: {n_samples}")
    print()

    # Sort by Sobol index (descending)
    sorted_indices = np.argsort(S)[::-1]

    # Header
    if sheet_labels:
        print(f"{Colors.BOLD}{'Rank':<6} {'Ref':<10} {'Block':<20} {'Class':<15} {'Sobol estimé':>14} {'σ/√N':>10} {'IC 95%':>28}{Colors.ENDC}")
    else:
        print(f"{Colors.BOLD}{'Rank':<6} {'Ref':<10} {'Class':<25} {'Sobol estimé':>14} {'σ/√N':>12} {'IC 95%':>30}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*120}{Colors.ENDC}")

    # Display top components
    displayed = 0
    for rank, idx in enumerate(sorted_indices):
        if displayed >= top_n:
            break

        ref = refs[idx]
        sobol_val = S[idx]
        std_norm_val = S_std_norm[idx]
        ci_low_val = CI_low[idx]
        ci_high_val = CI_high[idx]
        comp_class = classes.get(ref, 'Unknown')

        if sobol_val == 0:
            continue

        # Truncate class name
        if sheet_labels:
            max_class_len = 13
        else:
            max_class_len = 23
        if len(comp_class) > max_class_len:
            comp_class = comp_class[:max_class_len-3] + '...'

        # Format IC 95%
        ic_str = f"[{ci_low_val:.8f}, {ci_high_val:.8f}]"

        if sheet_labels:
            block = sheet_labels[idx]
            block_short = block.split('/')[-2] if block.endswith('/') else block.split('/')[-1]
            if len(block_short) > 18:
                block_short = block_short[:15] + '...'
            print(f"{rank+1:<6} {ref:<10} {block_short:<20} {comp_class:<15} {sobol_val:>14.6f} {std_norm_val:>10.6f} {ic_str:>28}")
        else:
            print(f"{rank+1:<6} {ref:<10} {comp_class:<25} {sobol_val:>14.6f} {std_norm_val:>12.6f} {ic_str:>30}")

        displayed += 1

    print(f"{Colors.CYAN}{'-'*120}{Colors.ENDC}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_sensitivity_analysis(excel_file: str,
                              sheet_name: str,
                              n_samples: int = 1000,
                              process_subblocks: bool = True,
                              seed: Optional[int] = None):
    """
    Execute complete Sobol sensitivity analysis workflow.

    Loads component data from Excel, performs Pick-Freeze Sobol analysis to
    quantify the contribution of each component's lambda uncertainty to the
    overall reliability variance, and displays formatted results.

    Args:
        excel_file: Path to Excel file containing component data with columns
            including 'Sheet', 'Reference', 'Class', and various parameters.
        sheet_name: Name of the block/sheet to analyze. If process_subblocks
            is True, analyzes all sheets starting with this prefix.
        n_samples: Number of Monte Carlo samples for the Pick-Freeze estimation.
        process_subblocks: If True, processes all sub-blocks starting with
            sheet_name as a system in series. If False, processes only the
            exact sheet_name.
        seed: Optional random seed for reproducible results.

    Returns:
        dict: Results dictionary from sobol_analysis_block or sobol_analysis_system,
            or None if no matching sheets found or analysis fails.
    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"SOBOL SENSITIVITY ANALYSIS (Pick-Freeze)")
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
        return None

    is_single_block = len(matching_sheets) == 1

    print_success(f"Found {len(matching_sheets)} matching block(s):")
    for sheet in matching_sheets:
        comp_count = len(df[df['Sheet'] == sheet])
        print(f"  {Colors.CYAN}- {sheet} ({comp_count} components){Colors.ENDC}")

    # Run analysis
    if is_single_block:
        results = sobol_analysis_block(df, matching_sheets[0], n_samples, seed=seed)
    else:
        results = sobol_analysis_system(df, matching_sheets, n_samples, seed=seed)

    if not results:
        return None

    # Display results
    print_sobol_results(results)

    print(f"\n{Colors.GREEN}Analysis complete{Colors.ENDC}")

    return results


