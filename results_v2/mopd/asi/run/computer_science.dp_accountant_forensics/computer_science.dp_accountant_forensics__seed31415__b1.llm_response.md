```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def load_data():
    data_path = 'data/'
    mechanism_panels = np.load(os.path.join(data_path, 'mechanism_panels.npz'))
    accountant_candidate_panels = np.load(os.path.join(data_path, 'accountant_candidate_panels.npz'))
    diagnostic_recipe = np.load(os.path.join(data_path, 'diagnostic_recipe.npz'))
    
    return mechanism_panels, accountant_candidate_panels, diagnostic_recipe

def compute_tail_mass(epsilon_grid, delta_curve, tail_start_index):
    tail_mass = np.trapz(delta_curve[tail_start_index:], epsilon_grid[tail_start_index:])
    return tail_mass

def compute_curve_residuals(epsilon_grid, candidate_curve, benchmark_curve):
    log10_delta_residuals = np.abs(np.log10(candidate_curve) - np.log10(benchmark_curve))
    curve_panel_error = np.sqrt(np.mean(log10_delta_residuals**2))
    return curve_panel_error

def compute_inverse_epsilon_errors(epsilon_grid, candidate_inverse, benchmark_inverse):
    epsilon_errors = np.abs(candidate_inverse - benchmark_inverse)
    inverse_panel_error = np.mean(epsilon_errors)
    return inverse_panel_error

def compute_probe_response_errors(candidate_sampling_response, candidate_group_response, benchmark_sampling_response, benchmark_group_response):
    sampling_error = np.abs(candidate_sampling_response - benchmark_sampling_response)
    group_error = np.abs(candidate_group_response - benchmark_group_response)
    return sampling_error, group_error

def audit_accountants(mechanism_panels, accountant_candidate_panels, diagnostic_recipe):
    epsilon_grid = mechanism_panels['epsilon_grid']
    target_deltas = diagnostic_recipe['target_deltas']
    primary_target_delta = target_deltas[0]
    primary_index = np.argmin(np.abs(target_deltas - primary_target_delta))
    mechanism_support_grid = mechanism_panels['mechanism_support_grid']
    baseline_increment_log_weights = mechanism_panels['baseline_increment_log_weights']
    sampling_probe_increment_log_weights = mechanism_panels['sampling_probe_increment_log_weights']
    group_probe_increment_log_weights = mechanism_panels['group_probe_increment_log_weights']
    baseline_inverse = mechanism_panels['baseline_inverse']
    sampling_probe_inverse = mechanism_panels['sampling_probe_inverse']
    group_probe_inverse = mechanism_panels['group_probe_inverse']
    sampling_probe_epsilon = mechanism_panels['sampling_probe_epsilon']
    group_probe_epsilon = mechanism_panels['group_probe_epsilon']
    inverse_epsilon = accountant_candidate_panels['inverse_epsilon']
    sampling_probe_epsilon_candidate = accountant_candidate_panels['sampling_probe_epsilon']
    group_probe_epsilon_candidate = accountant_candidate_panels['group_probe_epsilon']
    curve_candidate = accountant_candidate_panels['curve_candidate']
    
    tail_start_index = max(0, int(np.floor(0.72 * len(epsilon_grid))) - 1)
    benchmark_tail_mass = compute_tail_mass(epsilon_grid, mechanism_panels['delta_curve'], tail_start_index)
    
    curve_panel_errors = []
    inverse_panel_errors = []
    tail_mass_errors = []
    sampling_response_errors = []
    group_response_errors = []
    monotonicity_errors = []
    failure_modes = {}
    case_warnings = {}
    
    for accountant_id in range(len(inverse_epsilon)):
        candidate_inverse = inverse_epsilon[accountant_id]
        candidate_curve = curve_candidate[accountant_id]
        candidate_sampling_response = sampling_probe_epsilon_candidate[accountant_id] - inverse_epsilon[accountant_id, primary_index]
        candidate_group_response = group_probe_epsilon_candidate[accountant_id] - inverse_epsilon[accountant_id, primary_index]
        curve_panel_error = compute_curve_residuals(epsilon_grid, candidate_curve, mechanism_panels['delta_curve'])
        inverse_panel_error = compute_inverse_epsilon_errors(epsilon_grid, candidate_inverse, baseline_inverse)
        tail_mass_error = np.abs(candidate_curve[tail_start_index] - benchmark_tail_mass) / max(abs(benchmark_tail_mass), 1e-14)
        sampling_error, group_error = compute_probe_response_errors(candidate_sampling_response, candidate_group_response, sampling_probe_inverse[primary_index], group_probe_inverse[primary_index])
        monotonicity_error = np.sum(np.abs(np.gradient(candidate_curve)))
        
        curve_panel_errors.append(curve_panel_error)
        inverse_panel_errors.append(inverse_panel_error)
        tail_mass_errors.append(tail_mass_error)
        sampling_response_errors.append(sampling_error)
        group_response_errors.append(group_error)
        monotonicity_errors.append(monotonic
```