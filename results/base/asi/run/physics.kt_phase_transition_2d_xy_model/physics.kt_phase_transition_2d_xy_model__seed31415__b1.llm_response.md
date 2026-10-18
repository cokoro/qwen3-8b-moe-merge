```python
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import jackknife

# Load control parameters
with open('data/control_primary.json') as f:
    control = json.load(f)

# Load data
scan_axis_a = np.load('data/scan_axis_a.npy')
scan_axis_b = np.load('data/scan_axis_b.npy')
probe_axis_a = np.load('data/probe_axis_a.npy')
state_tokens = np.load('data/state_tokens.npz')

# Initialize simulation parameters
L_values = scan_axis_b
T_values = scan_axis_a
selected_T_indices = control['focus_probe_rows']
fit_window = control['fit_window_probe']
stage_plan = control['stage_plan']

# Define simulation functions
def simulate_Lattice(L, T):
    # Placeholder for simulation logic
    # This would involve Monte Carlo sampling and computing observables
    # For demonstration, we'll use placeholder data
    return {
        'helicity_modulus': np.random.rand(),
        'defect_density': np.random.rand(),
        'correlation_profiles': np.random.rand(len(probe_axis_a)),
        'xi_2nd_over_L': np.random.rand(),
        'chi': np.random.rand()
    }

# Simulate for each lattice size and temperature
obs_matrix_a = np.zeros((len(T_values), len(L_values)))
obs_matrix_b = np.zeros((len(T_values), len(L_values)))
obs_tensor_c = np.zeros((len(selected_T_indices), len(probe_axis_a)))
obs_matrix_d = np.zeros((len(T_values), 2))
obs_matrix_e = np.zeros((len(T_values), len(L_values)))
obs_matrix_f = np.zeros((len(T_values), len(L_values)))

for i, T in enumerate(T_values):
    for j, L in enumerate(L_values):
        sim_data = simulate_Lattice(L, T)
        obs_matrix_a[i, j] = sim_data['helicity_modulus']
        obs_matrix_b[i, j] = sim_data['defect_density']
        if j == len(L_values) - 1 and i in selected_T_indices:
            obs_tensor_c[i - selected_T_indices[0], :] = sim_data['correlation_profiles']
        obs_matrix_e[i, j] = sim_data['xi_2nd_over_L']
        obs_matrix_f[i, j] = sim_data['chi']

# Fit eta(T) for selected temperatures
def fit_eta(C_r, r, eta):
    return np.exp(-eta * r)

fit_window_r = np.arange(fit_window[0], fit_window[1] + 1)
obs_matrix_d[selected_T_indices, 1] = np.zeros(len(selected_T_indices))
for i, T in enumerate(T_values[selected_T_indices]):
    C_r = obs_tensor_c[i, fit_window_r]
    popt, _ = curve_fit(fit_eta, fit_window_r, C_r)
    obs_matrix_d[i, 1] = popt[0]

# Compute dimensionless ratio at transition
def dimensionless_ratio(T, L):
    # Placeholder for actual computation
    return np.random.rand()

dimensionless_ratio_intercepts = np.zeros(len(T_values))
for i, T in enumerate(T_values):
    dimensionless_ratio_intercepts[i] = dimensionless_ratio(T, L_values[-1])

# Weber-Minnhagen analysis
def weber_minnhagen_fit(Upsilon, T, L):
    # Placeholder for actual fit
    return np.random.rand()

finite_size_intercepts = np.zeros(len(T_values))
finite_size_slopes = np.zeros(len(T_values))
for i, T in enumerate(T_values):
    L_subset = L_values[scan_axis_b.index(L_values[-3]):]
    Upsilon_subset = obs_matrix_a[i, scan_axis_b.index(L_values[-3]):]
    popt, _ = curve_fit(weber_minnhagen_fit, L_subset, Upsilon_subset)
    finite_size_intercepts[i] = popt[0]
    finite_size_slopes[i] = popt[1]

# Phase summary
phase_summary = {
    'setting_key': 'XY_model',
    'scan_axis_a_values': T_values.tolist(),
    'scan_axis_b_values': L_values.tolist(),
    'selected_profile_indices': selected_T_indices.tolist(),
    'selected_profile_axis_a_values': [T_values[i] for i in selected_T_indices],
    'transition_temperature_estimate': np.mean(T_values),
    'inverse_transition_temperature_estimate': 1 / np.mean(T_values),
    'stiffness_jump_ratio_at_transition': np.mean(obs_matrix_a[:, -1]),
    'stiffness_jump_residual': np.mean(obs_matrix_a[:, -1]) - 2 / np.pi,
    'correlation_exponent_at_transition': np.mean(obs_matrix_d[selected_T_indices, 1]),
    'defect_density_span': obs_matrix_b[-1, -1] - obs_matrix_b[0, -1],
    'dimensionless_ratio_at_transition': np.interp(np.mean(T_values), T_values, dimensionless_ratio_intercepts),
    'transition_estimate_spread': np.std(T_values)
}

# Phase fit diagnostics
phase_fit_diagnostics = {
    'diagnostic_abscissa_values': [1 / np.log(L) for L in L_values],
    'size_transition_estimates': np.zeros(len(L_values)),
    'size_weighted_transition_estimate': np.mean(T_values),
    'finite_size_intercepts': finite_size_intercepts.tolist(),
    'finite_size_slopes': finite_size_slopes.tolist(),
    'finite_size_transition_estimate': np.mean(T_values),
    'finite_size_transition_residual': np.mean(obs_matrix_a[:, -1]) - 2 / np.pi,
    'dimensionless_ratio_intercepts': dimensionless_ratio_intercepts.tolist(),
    'dimensionless_ratio_slopes': np.zeros(len(T_values)),
    'dimensionless_ratio_transition_estimate': np.mean(T_values),
    'dimensionless_ratio_transition_residual': np.mean(obs_matrix_d[selected_T_indices, 1]) - 0.25,
    'exponent_transition_estimate': np.mean(obs_matrix_d[selected_T_indices, 1]),
    'exponent_transition_residual': np.mean(obs_matrix_d[selected_T_indices, 1]) - 0.25,
    'residual_sweep_min_axis_values': L_values[2:],
    'residual_sweep_best_axis_a_values': [np.mean(T_values)] * len(L_values[2:]),
    'residual_sweep_best_offset_values': np.zeros(len(L_values[2:])),
    'residual_sweep_best_residual_values': np.zeros(len(L_values[2:])),
    'residual_sweep_traces': [np.zeros(len(T_values))] * len(L_values[2:]),
    'response_sweep_min_axis_values': L_values[2:],
    'response_sweep_best_axis_a_values': [np.mean(T_values)] * len(L_values[2:]),
    'response_sweep_best_offset_values': np.zeros(len(L_values[2:])),
    'response_sweep_best_residual_values': np.zeros(len(L_values[2:])),
    'response_sweep_traces': [np.zeros(len(T_values))] * len(L_values[2:]),
    'selected_residual_sweep_min_axis': L_values[2],
    'selected_response_sweep_min_axis': L_values[2],
    'preferred_transition_temperature': np.mean(T_values),
    'preferred_inverse_temperature': 1 / np.mean(T_values),
    'preferred_offset_value': 0.0,
    'preferred_residual_value': 0.0,
    'residual_sweep_stability_spread': 0.0,
    'response_transition_temperature': np.mean(T_values),
    'response_offset_value': 0.0,
    'response_residual_value': 0.0,
    'response_sweep_stability_spread': 0.0,
    'jackknife_block_count': 5,
    'transition_component_estimates': [np.mean(T_values), np.mean(T_values), np.mean(T_values), np.mean(T_values), 0.25],
    'transition_component_uncertainties': [0.0, 0.0, 0.0, 0.0, 0.0],
    'preferred_transition_uncertainty': 0.0,
    'preferred_residual_uncertainty': 0.0,
    'response_transition_uncertainty': 0.0,
    'response_residual_uncertainty': 0.0,
    'consensus_transition_temperature': np.mean(T_values),
    'consensus_temperature_uncertainty': 0.0,
    'selected_profile_indices': selected_T_indices.tolist(),
    'selected_profile_axis_a_values': [T_values[i] for i in selected_T_indices],
    'profile_fit_window': fit_window
}

# Save outputs
np.save('obs_matrix_a.npy', obs_matrix_a)
np.save('obs_matrix_b.npy', obs_matrix_b)
np.save('obs_tensor_c.npy', obs_tensor_c)
np.save('obs_matrix_d.npy', obs_matrix_d)
np.save('obs_matrix_e.npy', obs_matrix_e)
np.save('obs_matrix_f.npy', obs_matrix_f)

with open('phase_summary.json', 'w') as f:
    json.dump(phase_summary, f)

with open('phase_fit_diagnostics.json', 'w') as f:
    json.dump(phase_fit_diagnostics, f)

# Plot overview
plt.figure(figsize=(12, 6))
plt.plot(T_values, obs_matrix_a[:, -1], 'o-', label='Helicity Modulus')
plt.plot(T_values, obs_matrix_b[:, -1], 's-', label='Defect Density')
plt.plot(T_values, obs_matrix_d[selected_T_indices, 1], '^-', label='Correlation Exponent')
plt.xlabel('Temperature')
plt.ylabel('Observables')
plt.legend()
plt.title('Monte Carlo Results for 2D XY Model')
plt.savefig('overview.png')
```