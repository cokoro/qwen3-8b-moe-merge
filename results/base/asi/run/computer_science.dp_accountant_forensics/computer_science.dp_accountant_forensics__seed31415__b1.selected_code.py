import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar

# Define paths
data_dir = 'data/'
results_dir = 'results/'
os.makedirs(results_dir, exist_ok=True)

# Load data
mechanism_data = np.load(os.path.join(data_dir, 'mechanism_panels.npz'))
accountant_data = np.load(os.path.join(data_dir, 'accountant_candidate_panels.npz'))
diagnostic_recipe = np.load(os.path.join(data_dir, 'diagnostic_recipe.npz'))

# Extract public data
epsilon_grid = mechanism_data['epsilon_grid']
target_deltas = mechanism_data['target_deltas']
primary_target_delta = target_deltas[0]
primary_index = np.where(target_deltas == primary_target_delta)[0][0]

# Benchmark curves and inverse queries
baseline_increment_log_weights = mechanism_data['baseline_increment_log_weights']
sampling_probe_increment_log_weights = mechanism_data['sampling_probe_increment_log_weights']
group_probe_increment_log_weights = mechanism_data['group_probe_increment_log_weights']

# Compute benchmark curves
def compute_tail_mass(epsilon_grid, support_grid, log_weights, temperature):
    tail_start_index = max(0, int(0.72 * len(epsilon_grid)) - 1)
    weights = np.exp(log_weights)
    tail_weights = weights[tail_start_index:]
    tail_epsilon = epsilon_grid[tail_start_index:]
    tail_mass = trapezoid(tail_weights, tail_epsilon)
    return tail_mass

def compute_benchmark_curve(log_weights, epsilon_grid):
    support_grid = epsilon_grid
    temperature = 0.115 + 0.010 * np.sqrt(np.maximum(support_grid, 0) + 1)
    tail = np.sum(np.exp(log_weights) / (1 + np.exp((support_grid - epsilon_grid) / temperature)))
    curve = np.clip(tail ** 1.08, 0, 1)
    return curve

benchmark_curve = compute_benchmark_curve(baseline_increment_log_weights, epsilon_grid)
benchmark_inverse = []
for i in range(len(target_deltas)):
    delta = target_deltas[i]
    # Interpolate benchmark curve at delta
    f = interp1d(epsilon_grid, benchmark_curve, kind='linear', fill_value='extrapolate')
    epsilon = f(delta)
    benchmark_inverse.append(epsilon)

# Compute candidate curves and inverse queries
accountant_ids = accountant_data.files
accountant_curves = {}
accountant_inverses = {}
accountant_sampling_responses = {}
accountant_group_responses = {}
accountant_tail_masses = {}

for accountant_id in accountant_ids:
    curves = accountant_data[accountant_id]
    inverses = accountant_data[f'{accountant_id}_inverse']
    sampling_probes = accountant_data[f'{accountant_id}_sampling_probe']
    group_probes = accountant_data[f'{accountant_id}_group_probe']
    tail_mass = compute_tail_mass(epsilon_grid, epsilon_grid, curves, 0.115 + 0.010 * np.sqrt(np.maximum(epsilon_grid, 0) + 1))
    
    accountant_curves[accountant_id] = curves
    accountant_inverses[accountant_id] = inverses
    accountant_sampling_responses[accountant_id] = sampling_probes
    accountant_group_responses[accountant_id] = group_probes
    accountant_tail_masses[accountant_id] = tail_mass

# Compute diagnostics
def compute_log10_delta_residual(candidate_curve, benchmark_curve, epsilon_grid):
    residuals = np.abs(np.log10(candidate_curve) - np.log10(benchmark_curve))
    return residuals

def compute_epsilon_error(candidate_inverse, benchmark_inverse):
    return np.abs(candidate_inverse - benchmark_inverse)

def compute_tail_mass_error(candidate_tail_mass, benchmark_tail_mass):
    return np.abs(candidate_tail_mass - benchmark_tail_mass) / max(np.abs(benchmark_tail_mass), 1e-14)

def compute_curve_panel_error(log10_delta_residuals):
    return np.sqrt(np.mean(log10_delta_residuals ** 2))

def compute_inverse_panel_error(epsilon_errors):
    return np.mean(epsilon_errors)

def compute_probe_response_error(candidate_response, benchmark_response):
    return np.abs(candidate_response - benchmark_response)

# Compute diagnostics for each accountant
diagnostics = {}
for accountant_id in accountant_ids:
    candidate_curve = accountant_curves[accountant_id]
    candidate_inverse = accountant_inverses[accountant_id]
    candidate_sampling_response = accountant_sampling_responses[accountant_id]
    candidate_group_response = accountant_group_responses[accountant_id]
    
    # Benchmark inverse
    benchmark_inverse_value = benchmark_inverse[primary_index]
    
    # Candidate probe responses
    candidate_sampling_response = candidate_sampling_response[primary_index]
    candidate_group_response = candidate_group_response[primary_index]
    
    # Compute log10 delta residuals
    log10_delta_residuals = compute_log10_delta_residual(candidate_curve, benchmark_curve, epsilon_grid)
    
    # Compute curve panel error
    curve_panel_error = compute_curve_panel_error(log10_delta_residuals)
    
    # Compute inverse panel error
    epsilon_errors = compute_epsilon_error(candidate_inverse, benchmark_inverse)
    inverse_panel_error = compute_inverse_panel_error(epsilon_errors)
    
    # Compute tail mass error
    tail_mass_error = compute_tail_mass_error(accountant_tail_masses[accountant_id], benchmark_tail_mass)
    
    # Compute probe response errors
    sampling_response_error = compute_probe_response_error(candidate_sampling_response, benchmark_inverse_value)
    group_response_error = compute_probe_response_error(candidate_group_response, benchmark_inverse_value)
    
    diagnostics[accountant_id] = {
        'curve_panel_error': curve_panel_error,
        'inverse_panel_error': inverse_panel_error,
        'tail_mass_error': tail_mass_error,
        'sampling_response_error': sampling_response_error,
        'group_response_error': group_response_error
    }

# Compute reliability scores
accountant_scores = {}
for accountant_id in accountant_ids:
    curve_error = diagnostics[accountant_id]['curve_panel_error']
    inverse_error = diagnostics[accountant_id]['inverse_panel_error']
    tail_error = diagnostics[accountant_id]['tail_mass_error']
    sampling_error = diagnostics[accountant_id]['sampling_response_error']
    group_error = diagnostics[accountant_id]['group_response_error']
    
    # Normalize errors
    max_error = max(curve_error, inverse_error, tail_error, sampling_error, group_error)
    curve_score = 1 - curve_error / max_error if max_error != 0 else 1
    inverse_score = 1 - inverse_error / max_error if max_error != 0 else 1
    tail_score = 1 - tail_error / max_error if max_error != 0 else 1
    sampling_score = 1 - sampling_error / max_error if max_error != 0 else 1
    group_score = 1 - group_error / max_error if max_error != 0 else 1
    
    # Aggregate score
    accountant_score = (curve_score + inverse_score + tail_score + sampling_score + group_score) / 5
    accountant_scores[accountant_id] = accountant_score

# Select reliable accountant
selected_accountant_id = max(accountant_scores, key=accountant_scores.get)
selected_accountant_score = accountant_scores[selected_accountant_id]

# Classify failure modes
failure_modes = {}
for accountant_id in accountant_ids:
    if accountant_id == selected_accountant_id:
        failure_modes[accountant_id] = 'reliable_accountant'
        continue
    
    # Check monotonicity
    curve = accountant_curves[accountant_id]
    monotonicity_error = 0
    for i in range(1, len(epsilon_grid)):
        if curve[i] < curve[i-1]:
            monotonicity_error += 1
    if monotonicity_error > 0:
        failure_modes[accountant_id] = 'monotonicity_smoothing_bug'
        continue
    
    # Check inverse interpolation
    inverse_error = diagnostics[accountant_id]['inverse_panel_error']
    if inverse_error > 0.1:
        failure_modes[accountant_id] = 'inverse_interpolation_mismatch'
        continue
    
    # Check sampling response
    sampling_error = diagnostics[accountant_id]['sampling_response_error']
    if sampling_error > 0.1:
        failure_modes[accountant_id] = 'missing_sampling_amplification'
        continue
    
    # Check group response
    group_error = diagnostics[accountant_id]['group_response_error']
    if group_error > 0.1:
        failure_modes[accountant_id] = 'group_setting_mismatch'
        continue
    
    # Check tail mass
    tail_error = diagnostics[accountant_id]['tail_mass_error']
    if tail_error > 0.1:
        failure_modes[accountant_id] = 'tail_truncation_loss'
        continue
    
    # Check curve shape
    curve_error = diagnostics[accountant_id]['curve_panel_error']
    if curve_error > 0.1:
        failure_modes[accountant_id] = 'optimistic_curve_alias'
        continue
    
    failure_modes[accountant_id] = 'unknown'

# Save results
pd.DataFrame([{
    'accountant_id': accountant_id,
    'curve_panel_error': diagnostics[accountant_id]['curve_panel_error'],
    'inverse_panel_error': diagnostics[accountant_id]['inverse_panel_error'],
    'tail_mass_error': diagnostics[accountant_id]['tail_mass_error'],
    'sampling_response_error': diagnostics[accountant_id]['sampling_response_error'],
    'group_response_error': diagnostics[accountant_id]['group_response_error'],
    'accountant_score': accountant_scores[accountant_id]
} for accountant_id in accountant_ids]).to_csv(os.path.join(results_dir, 'accountant_audit.csv'), index=False)

pd.DataFrame([{
    'epsilon': epsilon_grid,
    'benchmark_curve': benchmark_curve,
    'candidate_curve': accountant_curves[accountant_id],
    'candidate_inverse': accountant_inverses[accountant_id],
    'candidate_sampling_response': accountant_sampling_responses[accountant_id],
    'candidate_group_response': accountant_group_responses[accountant_id]
} for accountant_id in accountant_ids]).to_csv(os.path.join(results_dir, 'privacy_curve.csv'), index=False)

pd.DataFrame([{
    'target_delta': target_deltas,
    'benchmark_inverse': benchmark_inverse,
    'candidate_inverse': accountant_inverses[accountant_id]
} for accountant_id in accountant_ids]).to_csv(os.path.join(results_dir, 'delta_inverse_audit.csv'), index=False)

pd.DataFrame([{
    'accountant_id': accountant_id,
    'curve_panel_error': diagnostics[accountant_id]['curve_panel_error'],
    'inverse_panel_error': diagnostics[accountant_id]['inverse_panel_error'],
    'tail_mass_error': diagnostics[accountant_id]['tail_mass_error'],
    'sampling_response_error': diagnostics[accountant_id]['sampling_response_error'],
    'group_response_error': diagnostics[accountant_id]['group_response_error']
} for accountant_id in accountant_ids]).to_csv(os.path.join(results_dir, 'mechanism_panel_diagnostics.csv'), index=False)

with open(os.path.join(results_dir, 'failure_modes.json'), 'w') as f:
    json.dump({
        'selected_accountant_id': selected_accountant_id,
        'failure_modes': failure_modes
    }, f)

with open(os.path.join(results_dir, 'selected_accountant.json'), 'w') as f:
    json.dump({
        'selected_accountant_id': selected_accountant_id,
        'selected_failure_mode': 'reliable_accountant',
        'selected_accountant_score': selected_accountant_score
    }, f)

with open(os.path.join(results_dir, 'reliability_report.json'), 'w') as f:
    json.dump({
        'selected_accountant_id': selected_accountant_id,
        'aggregate_curve_rmse_log10': np.mean([d['curve_panel_error'] for d in diagnostics.values()]),
        'aggregate_inverse_epsilon_mae': np.mean([d['inverse_panel_error'] for d in diagnostics.values()]),
        'aggregate_panel_agreement_score': np.mean([d['curve_panel_error'] for d in diagnostics.values()]),
        'accountant_ranking': sorted(accountant_ids, key=lambda x: accountant_scores[x], reverse=True),
        'case_warnings': {}
    }, f)

# Plot accountant overview
plt.figure(figsize=(12, 6))
for accountant_id in accountant_ids:
    plt.plot(epsilon_grid, accountant_curves[accountant_id], label=accountant_id)
plt.plot(epsilon_grid, benchmark_curve, label='Benchmark')
plt.legend()
plt.xlabel('Epsilon')
plt.ylabel('Privacy Curve')
plt.title('Privacy Curve Overview')
plt.savefig(os.path.join(results_dir, 'accountant_overview.png'))