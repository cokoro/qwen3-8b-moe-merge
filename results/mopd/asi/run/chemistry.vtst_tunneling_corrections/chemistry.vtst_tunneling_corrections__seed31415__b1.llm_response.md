```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.signal import find_peaks

# Load data
surface_samples = pd.read_csv('data/surface_samples.csv')
curve_samples = pd.read_csv('data/curve_samples.csv')
tube_scan = pd.read_csv('data/tube_scan.csv')
kinetic_metric = pd.read_csv('data/kinetic_metric.csv')
temperature_grid = pd.read_csv('data/temperature_grid.csv')
isotope_cases = pd.read_csv('data/isotope_cases.csv')
system_manifest = pd.read_json('data/system_manifest.json')

# Constants
R_kJ_mol_K = system_manifest['R_kJ_mol_K']
kB_over_h_s_per_K = system_manifest['kB_over_h_s_per_K']
hc_kJ_mol_per_cm = system_manifest['hc_kJ_mol_per_cm']
hc_over_kB_K_cm = system_manifest['hc_over_kB_K_cm']
frequency_scale_cm1_per_sqrt_kj_mol_coord2 = system_manifest['frequency_scale_cm1_per_sqrt_kj_mol_coord2']
semiclassical_action_scale = system_manifest['semiclassical_action_scale']

# Step 1: Reconstruct the potential along the corridor
def fit_transverse_scan(q_offset, potential_kj_mol):
    q = q_offset
    V0 = potential_kj_mol[q == 0]
    k_perp = 2 * (potential_kj_mol[1:] - potential_kj_mol[:-1]) / (q[1:] - q[:-1])
    k_perp = np.interp(0, q, k_perp)
    omega_perp_cm1 = frequency_scale_cm1_per_sqrt_kj_mol_coord2 * np.sqrt(k_perp / kinetic_metric['transverse_reduced_mass_amu'])
    return V0, k_perp, omega_perp_cm1

# Step 2: Determine the lower-x reactant well, higher-x product well, and conventional saddle
def find_reactant_product_wells(V0):
    peaks, _ = find_peaks(V0)
    reactant_well = peaks[0]
    product_well = peaks[-1]
    return reactant_well, product_well

# Step 3: Compute the harmonic oscillator free energy
def compute_f_vib(omega, T):
    h_c_omega = hc_kJ_mol_per_cm * omega
    return 0.5 * h_c_omega + R_kJ_mol_K * T * np.log(1 - np.exp(-h_c_omega / (R_kJ_mol_K * T)))

# Step 4: Conventional TST and VTST rates
def compute_rates(V0, omega_perp, omega_reactant_long, omega_reactant_trans, T):
    G = V0 - V_reactant + compute_f_vib(omega_perp, T) - compute_f_vib(omega_reactant_long, T) - compute_f_vib(omega_reactant_trans, T)
    k_vtst = (kB_over_h_s_per_K * T) * np.exp(-G / (R_kJ_mol_K * T))
    return k_vtst

# Step 5: Compute energy-resolved tunneling
def compute_transmission(E, V0, T):
    S_1d = semiclassical_action_scale * quad(lambda s: np.sqrt(np.max(V0[s] - V_reactant - E, 0)), 0, len(V0))[0]
    S_2d = ... # Implement dynamic programming or shortest-path calculation
    P = 1 / (1 + np.exp(2 * S_2d))
    return P

# Step 6: Convert transmissions to canonical tunneling factor
def compute_kappa(T, E, P):
    Vbarrier = V_reactant - V0[reactant_well]
    integral = quad(lambda E: np.exp(-E / (R_kJ_mol_K * T)) * P(E), 0, Vbarrier)[0]
    kappa = 1 + np.exp(Vbarrier / (R_kJ_mol_K * T)) / (R_kJ_mol_K * T) * integral
    return kappa

# Step 7: Repeat calculations for isotope cases
# Write outputs
free_energy_profile = pd.DataFrame(columns=['T_K', 'point_id', 's_coord', 'x', 'y', 'V_mep_kj_mol', 'omega_perp_cm1', 'free_energy_rel_kj_mol', 'is_variational_max', 'is_conventional_saddle'])
rate_constants = pd.DataFrame(columns=['T_K', 'log10_k_tst_s_inv', 'log10_k_vtst_s_inv', 'log10_k_vtst_sct_s_inv', 'kappa_mep', 'kappa_sct', 's_vtst', 'barrier_tst_kj_mol', 'barrier_vtst_kj_mol', 'crossover_T_K'])
transmission_profile = pd.DataFrame(columns=['energy_id', 'energy_kj_mol', 'action_mep', 'action_sct', 'transmission_mep', 'transmission_sct', 'action_ratio_sct_over_mep'])
path_samples = pd.DataFrame(columns=['energy_id', 'path_step', 'energy_kj_mol', 'x', 'y', 'potential_kj_mol'])
isotope_effects = pd.DataFrame(columns=['case_id', 'isotopologue', 'longitudinal_mass_scale', 'transverse_mass_scale', 'T_K', 'log10_k_tst_s_inv', 'log10_k_vtst_s_inv', 'log10_k_vtst_sct_s_inv', 'kappa_sct', 's_vtst', 'barrier_vtst_kj_mol', 'crossover_T_K', 'log10_KIE_sct_vs_light'])
isotope_transmission_profile = pd.DataFrame(columns=['case_id', 'isotopologue', 'longitudinal_mass_scale', 'transverse_mass_scale', 'energy_id', 'energy_kj_mol', 'action_mep', 'action_sct', 'transmission_mep', 'transmission_sct', 'action_ratio_sct_over_mep'])
diagnostics = {}

# Write results
free_energy_profile.to_csv('results/free_energy_profile.csv', index=False)
rate_constants.to_csv('results/rate_constants.csv', index=False)
transmission_profile.to_csv('results/transmission_profile.csv', index=False)
path_samples.to_csv('results/path_samples.csv', index=False)
isotope_effects.to_csv('results/isotope_effects.csv', index=False)
isotope_transmission_profile.to_csv('results/isotope_transmission_profile.csv', index=False)
diagnostics.to_json('results/diagnostics.json')

# Plot
plt.plot(temperature_grid['T_K'], rate_constants['log10_k_tst_s_inv'], label='TST')
plt.plot(temperature_grid['T_K'], rate_constants['log10_k_vtst_s_inv'], label='VTST')
plt.plot(temperature_grid['T_K'], rate_constants['log10_k_vtst_sct_s_inv'], label='VTST+SCT')
plt.plot(temperature_grid['T_K'], rate_constants['kappa_mep'], label='kappa_mep')
plt.plot(temperature_grid['T_K'], rate_constants['kappa_sct'], label='kappa_sct')
plt.legend()
plt.savefig('results/rate_plot.png')
```