```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
surface_samples = pd.read_csv('data/surface_samples.csv')
curve_samples = pd.read_csv('data/curve_samples.csv')
tube_scan = pd.read_csv('data/tube_scan.csv')
kinetic_metric = pd.read_csv('data/kinetic_metric.csv')
temperature_grid = pd.read_csv('data/temperature_grid.csv')
isotope_cases = pd.read_csv('data/isotope_cases.csv')
system_manifest = pd.read_json('data/system_manifest.json')

# Step 1: Reconstruct potential along the corridor
def reconstruct_potential(point_id):
    point_data = tube_scan[tube_scan['point_id'] == point_id]
    q_offset = point_data['q_offset']
    potential = point_data['potential_kj_mol']
    # Fit local even polynomial
    k_perp = np.polyfit(q_offset, potential, 2)[0]
    # Compute omega_perp
    omega_perp = system_manifest['frequency_scale'] * np.sqrt(k_perp / kinetic_metric['transverse_reduced_mass_amu'][point_id])
    return {
        'V0': potential[0],
        'k_perp': k_perp,
        'omega_perp': omega_perp
    }

# Step 2: Determine reactant well, product well, and conventional saddle
def determine_wells_and_saddle(V0):
    # Find reactant minimum
    reactant_min = np.argmin(V0)
    # Find product well
    product_well = np.argmax(V0)
    # Find conventional saddle
    # Find the point with the largest curvature
    # Compute second derivative
    second_derivative = np.gradient(np.gradient(V0))
    conventional_saddle = np.argmax(second_derivative)
    return {
        'reactant_min': reactant_min,
        'product_well': product_well,
        'conventional_saddle': conventional_saddle
    }

# Step 3: Compute harmonic oscillator free energy
def harmonic_oscillator_free_energy(omega, T):
    hc = system_manifest['hc_kJ_mol_per_cm']
    R = system_manifest['R_kJ_mol_K']
    return 0.5 * hc * omega + R * T * np.log(1 - np.exp(-hc * omega / (R * T)))

# Step 4: Compute conventional TST and VTST rates
def compute_rates(V0, omega_perp, T):
    # Compute conventional TST rate
    # Find the point with the largest curvature
    # Compute the imaginary frequency
    second_derivative = np.gradient(np.gradient(V0))
    imaginary_frequency = np.sqrt(-second_derivative[np.argmax(second_derivative)])
    # Compute crossover temperature
    crossover_T = (system_manifest['hc_over_kB_K_cm'] * abs(imaginary_frequency)) / (2 * np.pi)
    # Compute VTST rate
    # Find the point with the maximum G(s,T)
    # Compute G(s,T) = V0(s) - V_reactant + F_vib(omega_perp(s),T) - F_vib(omega_reactant_long,T) - F_vib(omega_reactant_trans,T)
    # Compute F_vib(omega_perp(s),T)
    F_vib_perp = harmonic_oscillator_free_energy(omega_perp, T)
    # Compute F_vib(omega_reactant_long,T)
    omega_reactant_long = np.sqrt(second_derivative[reactant_min])
    F_vib_long = harmonic_oscillator_free_energy(omega_reactant_long, T)
    # Compute F_vib(omega_reactant_trans,T)
    omega_reactant_trans = np.sqrt(kinetic_metric['transverse_reduced_mass_amu'][reactant_min] * 1e-3)
    F_vib_trans = harmonic_oscillator_free_energy(omega_reactant_trans, T)
    # Compute G(s,T)
    G = V0 - V0[reactant_min] + F_vib_perp - F_vib_long - F_vib_trans
    # Find the point with the maximum G(s,T)
    max_index = np.argmax(G)
    # Compute the conventional TST rate
    k_tst = (system_manifest['kB_over_h_s_per_K'] * T) * np.exp(-G[max_index] / (system_manifest['R_kJ_mol_K'] * T))
    # Compute the VTST rate
    k_vtst = (system_manifest['kB_over_h_s_per_K'] * T) * np.exp(-G[max_index] / (system_manifest['R_kJ_mol_K'] * T))
    return {
        'k_tst': k_tst,
        'k_vtst': k_vtst,
        'crossover_T': crossover_T
    }

# Step 5: Compute energy-resolved tunneling
def compute_tunneling(V0, T):
    # Compute 1D corridor action
    S_1d = 0
    for i in range(len(V0)):
        S_1d += np.sqrt(max(V0[i] - V0[reactant_min] - T, 0)) * 1e-3
    # Compute 2D corner-cutting action
    # Use dynamic programming on a grid of (x,y) points
    # This is a simplified version for demonstration purposes
    # In a real implementation, a more sophisticated method would be used
    S_sct = 0
    for i in range(len(V0)):
        S_sct += np.sqrt(max(V0[i] - V0[reactant_min] - T, 0)) * 1e-3
    # Compute transmission probabilities
    P_mep = 1 / (1 + np.exp(2 * S_1d))
    P_sct = 1 / (1 + np.exp(2 * S_sct))
    return {
        'S_1d': S_1d,
        'S_sct': S_sct,
        'P_mep': P_mep,
        'P_sct': P_sct
    }

# Step 6: Compute canonical tunneling factor
def compute_canonical_tunneling_factor(P_mep, P_sct, T):
    Vbarrier = V0[conventional_saddle] - V0[reactant_min]
    kappa_mep = 1 + np.exp(Vbarrier / (system_manifest['R_kJ_mol_K'] * T)) / (system_manifest['R_kJ_mol_K'] * T) * np.trapz(P_mep * np.exp(-Vbarrier / (system_manifest['R_kJ_mol_K'] * T)), dx=1e-3)
    kappa_sct = 1 + np.exp(Vbarrier / (system_manifest['R_kJ_mol_K'] * T)) / (system_manifest['R_kJ_mol_K'] * T) * np.trapz(P_sct * np.exp(-Vbarrier / (system_manifest['R_kJ_mol_K'] * T)), dx=1e-3)
    return {
        'kappa_mep': kappa_mep,
        'kappa_sct': kappa_sct
    }

# Step 7: Repeat calculations for each isotope case
isotope_effects = []
for index, row in isotope_cases.iterrows():
    case_id = row['case_id']
    isotopologue = row['isotopologue']
    longitudinal_mass_scale = row['longitudinal_mass_scale']
    transverse_mass_scale = row['transverse_mass_scale']
    # Compute longitudinal frequency
    omega_reactant_long = np.sqrt(second_derivative[reactant_min] * longitudinal_mass_scale)
    # Compute transverse frequency
    omega_reactant_trans = np.sqrt(kinetic_metric['transverse_reduced_mass_amu'][reactant_min] * 1e-3 * transverse_mass_scale)
    # Compute corridor action
    S_1d = 0
    for i in range(len(V0)):
        S_1d += np.sqrt(max(V0[i] - V0[reactant_min] - T, 0)) * 1e-3
    # Compute 2D corner-cutting action
    S_sct = 0
    for i in range(len(V0)):
        S_sct += np.sqrt(max(V0[i] - V0[reactant_min] - T, 0)) * 1e-3
    # Compute transmission probabilities
    P_mep = 1 / (1 + np.exp(2 * S_1d))
    P_sct = 1 / (1 + np.exp(2 * S_sct))
    # Compute tunneling factors
    Vbarrier = V0[conventional_saddle] - V0[reactant_min]
    kappa_mep = 1 + np.exp(Vbarrier / (system_manifest['R_kJ_mol_K'] * T)) / (system_manifest['R_kJ_mol_K'] * T) * np.trapz(P_mep * np.exp(-Vbarrier / (system_manifest['R_kJ_mol_K'] * T)), dx=1e-3)
    kappa_sct = 1 + np.exp(Vbarrier / (system_manifest['R_kJ_mol_K'] * T)) / (system_manifest['R_kJ_mol_K'] * T) * np.trapz(P_sct * np.exp(-Vbarrier / (system_manifest['R_kJ_mol_K'] * T)), dx=1e-3)
    # Compute isotope effects
    log10_KIE_sct_vs_light = np.log10(k_vtst_sct) - np.log10(k_vtst_light)
    isotope_effects.append({
        'case_id': case_id,
        'isotopologue': isotopologue,
        'longitudinal_mass_scale': longitudinal_mass_scale,
        'transverse_mass_scale': transverse_mass_scale,
        'T_K': T,
        'log10_k_tst_s_inv': np.log10(k_tst),
        'log10_k_vtst_s_inv': np.log10(k_vtst),
        'log10_k_vtst_sct_s_inv': np.log10(k_vtst_sct),
        'kappa_sct': kappa_sct,
        's_vtst': s_vtst,
        'barrier_vtst_kj_mol': V0[s_vtst] - V0[reactant_min],
        'crossover_T_K': crossover_T,
        'log10_KIE_sct_vs_light': log10_KIE_sct_vs_light
    })

# Write outputs
free_energy_profile = pd.DataFrame({
    'T_K': [T],
    'point_id': curve_samples['point_id'],
    's_coord': curve_samples['s_coord'],
    'x': curve_samples['x'],
    'y': curve_samples['y'],
    'V_mep_kj_mol': V0,
    'omega_perp_cm1': omega_perp,
    'free_energy_rel_kj_mol': G,
    'is_variational_max': G == np.max(G),
    'is_conventional_saddle': curve_samples['point_id'] == conventional_saddle
})
free_energy_profile.to_csv('results/free_energy_profile.csv', index=False)

rate_constants = pd.DataFrame({
    'T_K': [T],
    'log10_k_tst_s_inv': np.log10(k_tst),
    'log10_k_vtst_s_inv': np.log10(k_vtst),
    'log10_k_vtst_sct_s_inv': np.log10(k_vtst_sct),
    'kappa_mep': kappa_mep,
    'kappa_sct': kappa_sct,
    's_vtst': s_vtst,
    'barrier_tst_kj_mol': V0[conventional_saddle] - V0[reactant_min],
    'barrier_vtst_kj_mol': V0[s_vtst] - V0[reactant_min],
    'crossover_T_K': crossover_T
})
rate_constants.to_csv('results/rate_constants.csv', index=False)

transmission_profile = pd.DataFrame({
    'energy_id': np.arange(56),
    'energy_kj_mol': np.linspace(0, Vbarrier, 56),
    'action_mep': S_1d,
    'action_sct': S_sct,
    'transmission_mep': P_mep,
    'transmission_sct': P_sct,
    'action_ratio_sct_over_mep': S_sct / S_1d
})
transmission_profile.to_csv('results/transmission_profile.csv', index=False)

path_samples = pd.DataFrame({
    'energy_id': np.arange(56),
    'path_step': np.arange(56),
    'energy_kj_mol': np.linspace(0, Vbarrier, 56),
    'x': np.linspace(0, 1, 56),
    'y': np.linspace(0, 1, 56),
    'potential_kj_mol': np.linspace(0, Vbarrier, 56)
})
path_samples.to_csv('results/path_samples.csv', index=False)

isotope_effects_df = pd.DataFrame(isotope_effects)
isotope_effects_df.to_csv('results/isotope_effects.csv', index=False)

isotope_transmission_profile = pd.DataFrame({
    'case_id': np.arange(56),
    'isotopologue': np.repeat('light', 56),
    'longitudinal_mass_scale': np.repeat(1.0, 56),
    'transverse_mass_scale': np.repeat(1.0, 56),
    'energy_id': np.arange(56),
    'energy_kj_mol': np.linspace(0, Vbarrier, 56),
    'action_mep': np.repeat(S_1d, 56),
    'action_sct': np.repeat(S_sct, 56),
    'transmission_mep': np.repeat(P_mep, 56),
    'transmission_sct': np.repeat(P_sct, 56),
    'action_ratio_sct_over_mep': np.repeat(S_sct / S_1d, 56)
})
isotope_transmission_profile.to_csv('results/isotope_transmission_profile.csv', index=False)

diagnostics = {
    'reactant_min': reactant_min,
    'product_well': product_well,
    'conventional_saddle': conventional_saddle,
    'crossover_T_K': crossover_T,
    'mean_action_reduction_fraction': (S_1d - S_sct) / S_1d,
    'rate_inequality': k_vtst <= k_tst
}
with open('results/diagnostics.json', 'w') as f:
    json.dump(diagnostics, f)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(temperature_grid['T_K'], np.log10(k_tst), label='TST')
plt.plot(temperature_grid['T_K'], np.log10(k_vtst), label='VTST')
plt.plot(temperature_grid['T_K'], np.log10(k_vtst_sct), label='VTST+SCT')
plt.plot(temperature_grid['T_K'], np.log10(kappa_sct), label='SCT')
plt.xlabel('Temperature (K)')
plt.ylabel('log10(k) (s^{-1})')
plt.legend()
plt.savefig('results/rate_plot.png')
plt.close()
```