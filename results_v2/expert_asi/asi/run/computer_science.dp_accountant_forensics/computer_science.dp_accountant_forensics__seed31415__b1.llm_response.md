```python
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load data
mechanism_data = np.load('data/mechanism_panels.npz')
accountant_data = np.load('data/accountant_candidate_panels.npz')

epsilon_grid = mechanism_data['epsilon_grid']
delta_grid = mechanism_data['delta_grid']
target_deltas = mechanism_data['target_deltas']
primary_index = np.searchsorted(delta_grid, target_deltas[0])

# Benchmark tail mass
tail_start_index = max(0, int(0.72 * len(epsilon_grid)) - 1)
tail_mass = np.trapz(delta_grid[tail_start_index:], epsilon_grid[tail_start_index:])

# Compute candidate tail mass
def compute_tail_mass(candidate_delta_curve):
    return np.trapz(candidate_delta_curve[tail_start_index:], epsilon_grid[tail_start_index:])

# Compute inverse estimates
def compute_inverse_estimate(candidate_delta_curve, delta_target, epsilon_grid):
    # Interpolate on the public epsilon_grid
    f = interp1d(delta_grid, epsilon_grid, kind='linear', fill_value='extrapolate')
    return f(delta_target)

# Compute probe responses
def compute_probe_response(candidate_probe_epsilon, baseline_inverse):
    return candidate_probe_epsilon - baseline_inverse

# Evaluate all candidates
results = {}
for accountant in accountant_data.files:
    candidate_delta_curve = accountant_data[accountant]
    # Compute tail mass
    candidate_tail_mass = compute_tail_mass(candidate_delta_curve)
    # Compute inverse estimates
    candidate_inverse = compute_inverse_estimate(candidate_delta_curve, target_deltas[0], epsilon_grid)
    # Compute probe responses
    candidate_sampling_response = compute_probe_response(accountant_data['sampling_probe_epsilon'][accountant], candidate_inverse)
    candidate_group_response = compute_probe_response(accountant_data['group_probe_epsilon'][accountant], candidate_inverse)
    # Compute diagnostics
    curve_error = np.sqrt(np.mean(np.log10(candidate_delta_curve / delta_grid) ** 2))
    inverse_error = np.abs(candidate_inverse - target_deltas[0])
    tail_mass_error = abs(candidate_tail_mass - tail_mass) / max(abs(tail_mass), 1e-14)
    results[accountant] = {
        'curve_error': curve_error,
        'inverse_error': inverse_error,
        'tail_mass_error': tail_mass_error,
        'sampling_response': candidate_sampling_response,
        'group_response': candidate_group_response
    }

# Aggregate diagnostics
accountant_ranking = sorted(results.keys(), key=lambda x: results[x]['curve_error'] + results[x]['inverse_error'])

# Select the most reliable accountant
selected_accountant = accountant_ranking[0]
selected_accountant_score = 100 - (results[selected_accountant]['curve_error'] + results[selected_accountant]['inverse_error']) * 100

# Write results
pd.DataFrame(results).to_csv('results/accountant_audit.csv', index=False)
pd.DataFrame({'accountant': accountant_ranking, 'score': [100 - (results[a]['curve_error'] + results[a]['inverse_error']) * 100 for a in accountant_ranking]}).to_csv('results/accountant_overview.csv', index=False)
```