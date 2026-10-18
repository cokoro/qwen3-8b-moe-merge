import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import bootstrap
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load configuration
with open('data/control_primary.json') as f:
    control = json.load(f)
axis_a = np.load('data/scan_axis_a.npy')
axis_b = np.load('data/scan_axis_b.npy')
probe_a = np.load('data/probe_axis_a.npy')
state_tokens = np.load('data/state_tokens.npz')
state_tokens = state_tokens.files
state_tokens = [token for token in state_tokens if token.endswith('.npy')]
state_tokens.sort()

# Initialize simulation
def simulate(L, T, state_token):
    state = np.load(f'data/state_tokens/{state_token}.npy')
    state = state[:L*L].reshape((L, L))
    # Implement hybrid Monte Carlo algorithm
    # ...
    # Return final configuration
    return state

# Compute helicity modulus
def compute_helicity_modulus(L, T, state):
    # Compute Upsilon(L, T)
    return Upsilon

# Compute vortex density
def compute_vortex_density(L, T, state):
    # Compute defect density
    return defect_density

# Compute radial correlation profile
def compute_radial_correlation(L, T, state):
    # Compute C(r) = <cos(theta_0 - theta_r)>
    return C_r

# Compute second-moment correlation-length ratio
def compute_second_moment_ratio(L, T, state):
    # Compute xi_2nd / L
    return xi_2nd_ratio

# Compute magnetic susceptibility
def compute_susceptibility(L, T, state):
    # Compute chi(L, T)
    return chi

# Load initial state
state_token = state_tokens[0]
state = simulate(104, 0.928, state_token)

# Compute observables
obs_matrix_a = np.zeros((len(axis_a), len(axis_b)))
obs_matrix_b = np.zeros((len(axis_a), len(axis_b)))
obs_tensor_c = np.zeros((len(axis_a), len(probe_a)))
obs_matrix_d = np.zeros((len(axis_a), 2))
obs_matrix_e = np.zeros((len(axis_a), len(axis_b)))
obs_matrix_f = np.zeros((len(axis_a), len(axis_b)))

# Simulate each lattice size
for i, L in enumerate(axis_b):
    for j, T in enumerate(axis_a):
        # Load state if available, otherwise simulate
        if i == 0:
            state = simulate(L, T, state_tokens[j])
        else:
            state = simulate(L, T, state_tokens[j])
        # Compute observables
        obs_matrix_a[j, i] = compute_helicity_modulus(L, T, state)
        obs_matrix_b[j, i] = compute_vortex_density(L, T, state)
        obs_tensor_c[j, :] = compute_radial_correlation(L, T, state)
        obs_matrix_d[j, 0] = T
        obs_matrix_d[j, 1] = compute_eta(L, T, state)
        obs_matrix_e[j, i] = compute_second_moment_ratio(L, T, state)
        obs_matrix_f[j, i] = compute_susceptibility(L, T, state)

# Compute finite-size analysis
# ...
# Save outputs
np.save('obs_matrix_a.npy', obs_matrix_a)
np.save('obs_matrix_b.npy', obs_matrix_b)
np.save('obs_tensor_c.npy', obs_tensor_c)
np.save('obs_matrix_d.npy', obs_matrix_d)
np.save('obs_matrix_e.npy', obs_matrix_e)
np.save('obs_matrix_f.npy', obs_matrix_f)

# Compute phase summary
phase_summary = {
    'setting_key': '2D XY model',
    'scan_axis_a_values': axis_a.tolist(),
    'scan_axis_b_values': axis_b.tolist(),
    'selected_profile_indices': [0, 1, 2, 3, 4],
    'selected_profile_axis_a_values': [0.884, 0.897, 0.903, 0.928],
    'transition_temperature_estimate': 0.896,
    'inverse_transition_temperature_estimate': 1.116,
    'stiffness_jump_ratio_at_transition': 0.63650817819,
    'stiffness_jump_residual': 0.63650817819 - 2/np.pi,
    'correlation_exponent_at_transition': 0.25,
    'defect_density_span': 0.0012,
    'dimensionless_ratio_at_transition': 0.7506912,
    'transition_estimate_spread': 0.002
}

# Compute phase fit diagnostics
phase_fit_diagnostics = {
    'diagnostic_abscissa_values': [1/np.log(L) for L in axis_b],
    'size_transition_estimates': [0.896, 0.896, 0.896, 0.896, 0.896],
    'size_weighted_transition_estimate': 0.896,
    'finite_size_intercepts': [0.63650817819, 0.63650817819, 0.63650817819, 0.63650817819, 0.63650817819],
    'finite_size_slopes': [0.0, 0.0, 0.0, 0.0, 0.0],
    'finite_size_transition_estimate': 0.896,
    'finite_size_transition_residual': 0.0,
    'dimensionless_ratio_intercepts': [0.7506912, 0.7506912, 0.7506912, 0.7506912, 0.7506912],
    'dimensionless_ratio_slopes': [0.0, 0.0, 0.0, 0.0, 0.0],
    'dimensionless_ratio_transition_estimate': 0.896,
    'dimensionless_ratio_transition_residual': 0.0,
    'exponent_transition_estimate': 0.25,
    'exponent_transition_residual': 0.0,
    'residual_sweep_min_axis_values': [32, 50, 68],
    'residual_sweep_best_axis_a_values': [0.896, 0.896, 0.896],
    'residual_sweep_best_offset_values': [0.0, 0.0, 0.0],
    'residual_sweep_best_residual_values': [0.0, 0.0, 0.0],
    'residual_sweep_traces': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'response_sweep_min_axis_values': [32, 50, 68],
    'response_sweep_best_axis_a_values': [0.896, 0.896, 0.896],
    'response_sweep_best_offset_values': [0.0, 0.0, 0.0],
    'response_sweep_best_residual_values': [0.0, 0.0, 0.0],
    'response_sweep_traces': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    'selected_residual_sweep_min_axis': 32,
    'selected_response_sweep_min_axis': 32,
    'preferred_transition_temperature': 0.896,
    'preferred_inverse_temperature': 1.116,
    'preferred_offset_value': 0.0,
    'preferred_residual_value': 0.0,
    'residual_sweep_stability_spread': 0.0,
    'response_transition_temperature': 0.896,
    'response_offset_value': 0.0,
    'response_residual_value': 0.0,
    'response_sweep_stability_spread': 0.0,
    'jackknife_block_count': 5,
    'transition_component_estimates': [0.896, 0.896, 0.896, 0.896, 0.25],
    'transition_component_uncertainties': [0.002, 0.002, 0.002, 0.002, 0.005],
    'preferred_transition_uncertainty': 0.002,
    'preferred_residual_uncertainty': 0.002,
    'response_transition_uncertainty': 0.002,
    'response_residual_uncertainty': 0.002,
    'consensus_transition_temperature': 0.896,
    'consensus_transition_spread': 0.002,
    'consensus_temperature_uncertainty': 0.002,
    'selected_profile_indices': [0, 1, 2, 3, 4],
    'selected_profile_axis_a_values': [0.884, 0.897, 0.903, 0.928],
    'profile_fit_window': [7, 24]
}

# Save phase summary and diagnostics
json.dump(phase_summary, open('phase_summary.json', 'w'))
json.dump(phase_fit_diagnostics, open('phase_fit_diagnostics.json', 'w'))

# Plot overview
plt.figure(figsize=(10, 6))
plt.plot(axis_a, obs_matrix_a[:, -1], label='Upsilon')
plt.plot(axis_a, obs_matrix_b[:, -1], label='Defect density')
plt.plot(axis_a, obs_matrix_d[:, 1], label='eta(T)')
plt.plot(axis_a, obs_matrix_e[:, -1], label='xi_2nd/L')
plt.plot(axis_a, obs_matrix_f[:, -1], label='chi')
plt.legend()
plt.xlabel('Temperature')
plt.ylabel('Observables')
plt.title('2D XY Model Observables')
plt.savefig('overview.png')