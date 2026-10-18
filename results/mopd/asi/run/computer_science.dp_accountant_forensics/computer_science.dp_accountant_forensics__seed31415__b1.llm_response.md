```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    data = np.load('data/mechanism_panels.npz')
    mechanism_support_grid = data['mechanism_support_grid']
    baseline_increment_log_weights = data['baseline_increment_log_weights']
    sampling_probe_increment_log_weights = data['sampling_probe_increment_log_weights']
    group_probe_increment_log_weights = data['group_probe_increment_log_weights']
    diagnostic_recipe = data['diagnostic_recipe']
    
    candidate_data = np.load('data/accountant_candidate_panels.npz')
    candidate_curves = candidate_data['accountant_candidate_curves']
    candidate_inverse_epsilons = candidate_data['accountant_candidate_inverse_epsilons']
    candidate_sampling_epsilons = candidate_data['accountant_candidate_sampling_epsilons']
    candidate_group_epsilons = candidate_data['accountant_candidate_group_epsilons']
    
    epsilon_grid = diagnostic_recipe['epsilon_grid']
    target_deltas = diagnostic_recipe['target_deltas']
    primary_target_delta = diagnostic_recipe['primary_target_delta']
    primary_index = np.where(target_deltas == primary_target_delta)[0][0]
    
    return {
        'mechanism_support_grid': mechanism_support_grid,
        'baseline_increment_log_weights': baseline_increment_log_weights,
        'sampling_probe_increment_log_weights': sampling_probe_increment_log_weights,
        'group_probe_increment_log_weights': group_probe_increment_log_weights,
        'diagnostic_recipe': diagnostic_recipe,
        'candidate_curves': candidate_curves,
        'candidate_inverse_epsilons': candidate_inverse_epsilons,
        'candidate_sampling_epsilons': candidate_sampling_epsilons,
        'candidate_group_epsilons': candidate_group_epsilons,
        'epsilon_grid': epsilon_grid,
        'target_deltas': target_deltas,
        'primary_index': primary_index
    }

def compute_tail_mass(epsilon_grid, delta_curve):
    tail_start_index = max(0, int(0.72 * len(epsilon_grid)) - 1)
    tail_mass = np.trapz(delta_curve[tail_start_index:], epsilon_grid[tail_start_index:])
    return tail_mass

def compute_tail_mass_error(candidate_tail_mass, benchmark_tail_mass):
    return abs(candidate_tail_mass - benchmark_tail_mass) / max(abs(benchmark_tail_mass), 1e-14)

def compute_curve_panel_error(candidate_curve, benchmark_curve, epsilon_grid):
    log10_delta_residuals = np.log10(candidate_curve / benchmark_curve)
    return np.sqrt(np.mean(log10_delta_residuals**2))

def compute_inverse_panel_error(candidate_inverse, benchmark_inverse, target_deltas):
    epsilon_error = np.abs(candidate_inverse - benchmark_inverse)
    return np.mean(epsilon_error)

def compute_probe_response(candidate_sampling, candidate_group, benchmark_inverse):
    return candidate_sampling - benchmark_inverse, candidate_group - benchmark_inverse

def run_analysis():
    data = load_data()
    epsilon_grid = data['epsilon_grid']
    target_deltas = data['target_deltas']
    primary_index = data['primary_index']
    
    # Compute benchmark curves and inverse queries
    benchmark_curves = np.exp(data['baseline_increment_log_weights'])
    benchmark_inverses = np.zeros_like(data['candidate_inverse_epsilons'])
    for i in range(len(epsilon_grid)):
        benchmark_inverses[:, i] = data['candidate_inverse_epsilons'][:, i]
    
    # Compute candidate curves and inverse queries
    candidate_curves = np.exp(data['candidate_curves'])
    candidate_inverses = np.zeros_like(data['candidate_inverse_epsilons'])
    for i in range(len(epsilon_grid)):
        candidate_inverses[:, i] = data['candidate_inverse_epsilons'][:, i]
    
    # Compute benchmark tail mass
    benchmark_tail_mass = compute_tail_mass(epsilon_grid, benchmark_curves)
    
    # Compute candidate tail mass and error
    candidate_tail_mass = np.zeros(len(candidate_curves))
    for i in range(len(candidate_curves)):
        candidate_tail_mass[i] = compute_tail_mass(epsilon_grid, candidate_curves[i])
    tail_mass_error = np.zeros(len(candidate_curves))
    for i in range(len(candidate_curves)):
        tail_mass_error[i] = compute_tail_mass_error(candidate_tail_mass[i], benchmark_tail_mass)
    
    # Compute curve panel error
    curve_panel_error = np.zeros(len(candidate_curves))
    for i in range(len(candidate_curves)):
        curve_panel_error[i] = compute_curve_panel_error(candidate_curves[i], benchmark_curves, epsilon_grid)
    
    # Compute inverse panel error
    inverse_panel_error = np.zeros(len(candidate_curves))
    for i in range(len(candidate_curves)):
        inverse_panel_error[i] = compute_inverse_panel_error(candidate_inverses[i], benchmark_inverses[i], target_deltas)
    
    # Compute probe response
    sampling_response = np.zeros(len(candidate_curves))
    group_response = np.zeros(len(candidate_curves))
    for i in range(len(candidate_curves)):
        sampling_response[i] = data['candidate_sampling_epsilons'][i] - benchmark_inverses[i]
        group_response[i] = data['candidate_group_epsilons'][i] - benchmark_inverses[i]
    
    # Compute monotonicity
    monotonicity_error = np.zeros(len(candidate_curves))
    for i in range(len(candidate_curves)):
        monotonicity_error[i] = np.sum(np.abs(np.gradient(candidate_curves[i])) / np.gradient(epsilon_grid))
    
    # Compute reliability scores
    reliability_scores = np.zeros(len(candidate_curves))
    for i in range(len(candidate_curves)):
        reliability_scores[i] = (1 - (curve_panel_error[i] + inverse_panel_error[i] + tail_mass_error[i] + monotonicity_error[i])) * 100
    
    # Write results
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({
        'accountant_id': range(len(candidate_curves)),
        'reliability_score': reliability_scores
    }).to_csv('results/accountant_audit.csv', index=False)
    
    pd.DataFrame({
        'accountant_id': range(len(candidate_curves)),
        'curve_panel_error': curve_panel_error,
        'inverse_panel_error': inverse_panel_error,
        'tail_mass_error': tail_mass_error,
        'monotonicity_error': monotonicity_error
    }).to_csv('results/mechanism_panel_diagnostics.csv', index=False)
    
    pd.DataFrame({
        'accountant_id': range(len(candidate_curves)),
        'sampling_response': sampling_response,
        'group_response': group_response
    }).to_csv('results/delta_inverse_audit.csv', index=False)
    
    pd.DataFrame({
        'accountant_id': range(len(candidate_curves)),
        'curve': candidate_curves,
        'inverse': candidate_inverses
    }).to_csv('results/privacy_curve.csv', index=False)
    
    # Compute failure modes
    failure_modes = {}
    selected_accountant_id = np.argmin(reliability_scores)
    selected_accountant_score = reliability_scores[selected_accountant_id]
    
    for i in range(len(candidate_curves)):
        if i == selected_accountant_id:
            failure_modes[i] = 'reliable_accountant'
        else:
            if monotonicity_error[i] > 1:
                failure_modes[i] = 'monotonicity_smoothing_bug'
            elif inverse_panel_error[i] > 1:
                failure_modes[i] = 'inverse_interpolation_mismatch'
            elif sampling_response[i] > 1:
                failure_modes[i] = 'missing_sampling_amplification'
            elif group_response[i] > 1:
                failure_modes[i] = 'group_setting_mismatch'
            elif tail_mass_error[i] > 1:
                failure_modes[i] = 'tail_truncation_loss'
            elif curve_panel_error[i] > 1:
                failure_modes[i] = 'optimistic_curve_alias'
            else:
                failure_modes[i] = 'conservative_curve_alias'
    
    pd.DataFrame({
        'selected_accountant_id': [selected_accountant_id],
        'selected_failure_mode': [failure_modes[selected_accountant_id]],
        'selected_accountant_score': [selected_accountant_score]
    }).to_csv('results/selected_accountant.json', index=False)
    
    pd.DataFrame({
        'failure_mode': [failure_modes[i] for i in range(len(candidate_curves))],
        'accountant_id': range(len(candidate_curves))
    }).to_json('results/failure_modes.json')
    
    pd.DataFrame({
        'selected_accountant_id': [selected_accountant_id],
        'aggregate_curve_rmse_log10': [np.mean(np.log10(candidate_curves[selected_accountant_id] / benchmark_curves))],
        'aggregate_inverse_epsilon_mae': [np.mean(np.abs(candidate_inverses[selected_accountant_id] - benchmark_inverses[selected_accountant_id]))],
        'aggregate_panel_agreement_score': [reliability_scores[selected_accountant_id]],
        'accountant_ranking': [range(len(candidate_curves))],
        'case_warnings': {}
    }).to_json('results/reliability_report.json')
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_grid, benchmark_curves, label='Benchmark')
    plt.plot(epsilon_grid, candidate_curves[selected_accountant_id], label='Selected Accountant')
    plt.legend()
    plt.savefig('results/accountant_overview.png')
```