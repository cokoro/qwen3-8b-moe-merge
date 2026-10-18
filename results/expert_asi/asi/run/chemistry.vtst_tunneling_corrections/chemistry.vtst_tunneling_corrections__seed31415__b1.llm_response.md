```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path)

def main():
    # Load all input files
    surface_samples = load_data('data/surface_samples.csv')
    curve_samples = load_data('data/curve_samples.csv')
    tube_scan = load_data('data/tube_scan.csv')
    kinetic_metric = load_data('data/kinetic_metric.csv')
    temp_grid = load_data('data/temperature_grid.csv')
    isotope_cases = load_data('data/isotope_cases.csv')
    manifest = load_data('data/system_manifest.json')

    # Step 1: Reconstruct potential along corridor
    # Fit each transverse scan to a local even polynomial
    # Compute omega_perp(s) and V0(s)
    # Assume `tube_scan` has columns `q_offset`, `potential_kj_mol`
    # For each `point_id`, fit a quadratic around q=0
    # Then compute omega_perp = frequency_scale * sqrt(k_perp / transverse_mass)
    # Implement this step with curve fitting
    # Output: V0(s), omega_perp(s), and V0(s) along the corridor

    # Step 2: Determine reactant, product, and saddle
    # Find the minimum of V0(s) on the reactant side
    # Compute the imaginary frequency at the saddle
    # Calculate crossover_T = (hc_over_kB_K_cm * |imag_freq|) / (2*pi)
    # Output: crossover_T, reactant_long_freq, etc.

    # Step 3: Compute F_vib(omega, T) for each temperature
    # Use hc_kJ_mol_per_cm and R_kJ_mol_K
    # Compute G(s, T) for all s and T

    # Step 4: Find the variational maximum for each T
    # Use a maximization routine to find the point that maximizes G(s, T)
    # Output: s_vtst, barrier_vtst, and conventional saddle

    # Step 5: Compute S_1d(E) and S_sct(E) using WKB
    # Use dynamic programming to find the shortest path through the forbidden region
    # Output: action profiles for each energy

    # Step 6: Compute tunneling factors for each T
    # Use the integral of P(E) * exp(-E/(R*T)) * exp(Vbarrier/(R*T)) / (R*T)
    # Output: kappa_sct, kappa_mep, and their ratio

    # Step 7: Repeat for each isotope case
    # Adjust mass scales for longitudinal and transverse frequencies
    # Compute all rate constants for each case

    # Save outputs to results/
    # Plot rate curves and tunneling factors

if __name__ == '__main__':
    main()
```