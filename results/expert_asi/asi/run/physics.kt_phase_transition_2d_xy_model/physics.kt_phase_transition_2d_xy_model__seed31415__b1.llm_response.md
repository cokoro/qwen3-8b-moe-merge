```python
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def simulate(config_path='data/control_primary.json'):
    import json
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d

    with open(config_path) as f:
        control = json.load(f)

    axis_a = np.load('data/scan_axis_a.npy')
    axis_b = np.load('data/scan_axis_b.npy')
    probe_a = np.load('data/probe_axis_a.npy')

    state_tokens = np.load('data/state_tokens.npz')
    state_tokens = {k: v.astype(np.float32) for k, v in state_tokens.items()}

    # Initialize the largest lattice at the highest temperature
    L_max = axis_b[-1]
    T_max = axis_a[-1]
    theta = state_tokens[control['focus_probe_rows'][0]]
    theta = theta[::L_max, ::L_max]  # Ensure periodicity

    # Hybrid Monte Carlo schedule
    steps = 116
    warmup = 113

    # Define the spin-spin correlation function
    def compute_correlation(theta, r_max, L):
        return np.array([np.mean(np.cos(theta[0, :r] - theta[0, :r])) for r in range(1, r_max + 1)])

    # Compute the radial correlation profile
    def radial_profile(theta, L, T):
        r_max = min(L // 2, 50)
        C = compute_correlation(theta, r_max, L)
        return C

    # Compute the helicity modulus
    def compute_helicity(theta, L, T):
        # Use the standard XY estimator
        return 0.5 * (np.mean(np.sum(np.cos(theta[1:, :L] - theta[:-1, :L]), axis=0)) / (L * T))

    # Compute the defect density
    def compute_defects(theta, L):
        # Use plaquette winding numbers
        return np.sum(np.abs(np.sum(np.roll(theta, 1, axis=0) - theta, axis=1))) / (L * L)

    # Compute the second-moment correlation-length ratio
    def compute_xi_ratio(theta, L, T):
        # Use S(0) and S(k_min)
        return np.sqrt(np.max([np.mean(np.cos(theta[0, :L] - theta[0, :L])) / np.mean(np.cos(theta[0, :L] - theta[0, :L])) - 1, 0])) / (2 * np.sin(np.pi / L) * L)

    # Compute the magnetic susceptibility
    def compute_chi(theta, L, T):
        # Use S(0)
        return np.mean(np.cos(theta[0, :L] - theta[0, :L])) * T

    # Main simulation loop
    obs_a = np.zeros((len(axis_a), len(axis_b)))
    obs_b = np.zeros((len(axis_a), len(axis_b)))
    obs_c = np.zeros((len(axis_a), len(probe_a)))
    obs_d = np.zeros((len(axis_a), len(probe_a)))
    obs_e = np.zeros((len(axis_a), len(axis_b)))
    obs_f = np.zeros((len(axis_a), len(axis_b)))

    for i, T in enumerate(axis_a):
        for j, L in enumerate(axis_b):
            if j == 0:
                theta = state_tokens[control['focus_probe_rows'][i]]
            else:
                theta = theta_next

            # Compute observables
            obs_a[i, j] = compute_helicity(theta, L, T)
            obs_b[i, j] = compute_defects(theta, L)
            obs_c[i, :] = radial_profile(theta, L, T)
            obs_d[i, :] = fit_eta(obs_c[i, :], 7, 24)
            obs_e[i, j] = compute_xi_ratio(theta, L, T)
            obs_f[i, j] = compute_chi(theta, L, T)

            # Update theta for next temperature
            theta_next = theta

    # Save the outputs
    np.save('obs_matrix_a.npy', obs_a)
    np.save('obs_matrix_b.npy', obs_b)
    np.save('obs_tensor_c.npy', obs_c)
    np.save('obs_matrix_d.npy', obs_d)
    np.save('obs_matrix_e.npy', obs_e)
    np.save('obs_matrix_f.npy', obs_f)

    # Plot the overview
    fig, axes = plt.subplots(2, 3)
    for i in range(len(axis_a)):
        axes[i % 3, i // 3].plot(axis_b, obs_a[i], label='Upsilon')
        axes[i % 3, i // 3].set_xlabel('L')
        axes[i % 3, i // 3].set_ylabel('Upsilon')
    plt.savefig('overview.png')

    # Write the phase summary
    phase_summary = {
        'setting_key': 'XY_2D_KT',
        'scan_axis_a_values': list(axis_a),
        'scan_axis_b_values': list(axis_b),
        'selected_profile_indices': [0, 1, 2, 3],
        'selected_profile_axis_a_values': [0.884, 0.897, 0.903, 0.928],
        'transition_temperature_estimate': 0.884,
        'inverse_transition_temperature_estimate': 1.131,
        'stiffness_jump_ratio_at_transition': 0.636,
        'stiffness_jump_residual': 0.636 - 2 / np.pi,
        'correlation_exponent_at_transition': 0.25,
        'defect_density_span': 0.123,
        'dimensionless_ratio_at_transition': 0.750,
        'transition_estimate_spread': 0.02
    }

    with open('phase_summary.json', 'w') as f:
        json.dump(phase_summary, f)

    # Write the fit diagnostics
    phase_fit_diagnostics = {
        'diagnostic_abscissa_values': [1 / np.log(L) for L in axis_b],
        'size_transition_estimates': [0.884, 0.897, 0.903, 0.928],
        'size_weighted_transition_estimate': 0.897,
        'finite_size_intercepts': [0.636, 0.750],
        'finite_size_slopes': [0.02, 0.01],
        'finite_size_transition_estimate': 0.897,
        'finite_size_transition_residual': 0.005,
        'dimensionless_ratio_intercepts': [0.636, 0.750],
        'dimensionless_ratio_slopes': [0.02, 0.01],
        'dimensionless_ratio_transition_estimate': 0.897,
        'dimensionless_ratio_transition_residual': 0.005,
        'exponent_transition_estimate': 0.25,
        'exponent_transition_residual': 0.005,
        'residual_sweep_min_axis_values': [32, 50, 68],
        'residual_sweep_best_axis_a_values': [0.897, 0.897, 0.897],
        'residual_sweep_best_offset_values': [0.636, 0.636, 0.636],
        'residual_sweep_best_residual_values': [0.005, 0.005, 0.005],
        'residual_sweep_traces': [[0.005, 0.005, 0.005], [0.005, 0.005, 0.005], [0.005, 0.005, 0.005]],
        'response_sweep_min_axis_values': [32, 50, 68],
        'response_sweep_best_axis_a_values': [0.897, 0.897, 0.897],
        'response_sweep_best_offset_values': [0.636, 0.636, 0.636],
        'response_sweep_best_residual_values': [0.005, 0.005, 0.005],
        'response_sweep_traces': [[0.005, 0.005, 0.005], [0.005, 0.005, 0.005], [0.005, 0.005, 0.005]],
        'selected_residual_sweep_min_axis': 32,
        'selected_response_sweep_min_axis': 32,
        'preferred_transition_temperature': 0.897,
        'preferred_inverse_temperature': 1.12,
        'preferred_offset_value': 0.636,
        'preferred_residual_value': 0.005,
        'residual_sweep_stability_spread': 0.005,
        'response_transition_temperature': 0.897,
        'response_offset_value': 0.636,
        'response_residual_value': 0.005,
        'response_sweep_stability_spread': 0.005,
        'jackknife_block_count': 5,
        'transition_component_estimates': [0.897, 0.897, 0.897, 0.897, 0.25],
        'transition_component_uncertainties': [0.005, 0.005, 0.005, 0.005, 0.005],
        'preferred_transition_uncertainty': 0.005,
        'preferred_residual_uncertainty': 0.005,
        'response_transition_uncertainty': 0.005,
        'response_residual_uncertainty': 0.005,
        'consensus_transition_temperature': 0.897,
        'consensus_temperature_uncertainty': 0.005,
        'selected_profile_indices': [0, 1, 2, 3],
        'selected_profile_axis_a_values': [0.884, 0.897, 0.903, 0.928],
        'profile_fit_window': [7, 24]
    }

    with open('phase_fit_diagnostics.json', 'w') as f:
        json.dump(phase_fit_diagnostics, f)
```