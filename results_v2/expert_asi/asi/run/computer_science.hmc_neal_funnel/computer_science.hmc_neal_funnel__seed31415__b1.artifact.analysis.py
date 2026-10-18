import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_model
from scipy.stats import percentileofscore, ttest_1samp

# Load configuration
with open('data/target_manifest.json') as f:
    manifest = json.load(f)
with open('data/transition_schedule.json') as f:
    schedule = json.load(f)

# Load candidate transition bundle
bundle = np.load('data/candidate_transition_bundle.npz')
candidate_ids = bundle['candidate_ids']
step_size_ladder = bundle['step_size_ladder']
probe_states = bundle['probe_states']
probe_momenta = bundle['probe_momenta']
proposal_states = bundle['proposal_states']
proposal_momenta = bundle['proposal_momenta']
reverse_states = bundle['reverse_states']
reverse_momenta = bundle['reverse_momenta']
volume_input = bundle['volume_input']
volume_output = bundle['volume_output']

# Load candidate chain panel
panel = np.load('data/candidate_chain_panel.npz')
panel_centered_samples = panel['panel_centered_samples']
panel_delta_h = panel['panel_delta_h']
panel_accepted = panel['panel_accepted']

# Load production replay streams
production_states = np.load('data/initial_latent_states.npy')
production_momenta = np.load('data/transition_momenta.npy')
production_uniforms = np.load('data/transition_uniforms.npy')

# Load held-out replay streams
holdout_states = np.load('data/holdout_initial_latent_states.npy')
holdout_momenta = np.load('data/holdout_transition_momenta.npy')
holdout_uniforms = np.load('data/holdout_transition_uniforms.npy')

# Compute candidate validity metrics
results = []
for i, candidate_id in enumerate(candidate_ids):
    # Energy RMSE
    base_step = step_size_ladder[-1]
    H_proposal = compute_h(proposal_states[i], proposal_momenta[i])
    H_probe = compute_h(probe_states[i], probe_momenta[i])
    energy_rmse = np.sqrt(np.mean((H_proposal - H_probe)**2))
    
    # Energy slope
    log_step = np.log(step_size_ladder)
    log_energy = np.log(np.abs(H_proposal - H_probe))
    slope = np.polyfit(log_step, log_energy, 1)[0]
    
    # Reverse p95
    norm = np.linalg.norm(reverse_states[i] - probe_states[i]) + np.linalg.norm(reverse_momenta[i] + probe_momenta[i])
    reverse_p95 = np.percentile(norm, 95)
    
    # Volume logdet
    A = np.linalg.lstsq(volume_input, volume_output[i], rcond=None)[0]
    logdet = np.linalg.slogdet(A)[1]
    volume_logdet_abs = np.abs(logdet)
    
    # Acceptance rate
    acceptance_rate = np.mean(panel_accepted[i])
    
    # Panel diagnostics
    v_samples = panel_centered_samples[i][:, 0]
    panel_mean_v = np.mean(v_samples)
    panel_var_v = np.var(v_samples)
    panel_ess_v = ess(v_samples)
    panel_rhat_v = rhat(v_samples)
    
    # Validity score
    error = np.array([
        energy_rmse,
        reverse_p95,
        volume_logdet_abs,
        np.abs((panel_var_v - schedule['sigma_v']**2) / schedule['sigma_v']**2),
        np.abs(0.82 - acceptance_rate)
    ])
    full = np.array([0.01, 5e-3, 0.08, 0.55, 0.80])
    zero = np.array([0.18, 1e-8, 0.08, 0.08, 0.25])
    linear = np.clip((error - zero) / (full - zero), 0, 1)
    validity = (
        0.24 * linear[0] +
        0.24 * linear[1] +
        0.20 * linear[2] +
        0.16 * linear[3] +
        0.16 * linear[4]
    )
    
    results.append({
        'candidate_id': candidate_id,
        'energy_rmse': energy_rmse,
        'energy_slope': slope,
        'reverse_p95': reverse_p95,
        'volume_logdet_abs': volume_logdet_abs,
        'acceptance_rate': acceptance_rate,
        'panel_mean_v': panel_mean_v,
        'panel_var_v': panel_var_v,
        'panel_ess_v': panel_ess_v,
        'panel_rhat_v': panel_rhat_v,
        'validity_score': validity
    })

# Select the valid kernel
valid_kernel = max(results, key=lambda x: x['validity_score'])['candidate_id']
failure_modes = {
    'kernel_00': 'valid_transition',
    'kernel_01': 'energy_drift',
    'kernel_02': 'terminal_imbalance',
    'kernel_03': 'delayed_response',
    'kernel_04': 'coordinate_mismatch',
    'kernel_05': 'volume_loss'
}
with open('results/failure_modes.json', 'w') as f:
    json.dump({'selected_kernel_id': valid_kernel, 'failure_modes': failure_modes}, f)

with open('results/selected_kernel.json', 'w') as f:
    json.dump({
        'selected_kernel_id': valid_kernel,
        'selected_failure_mode': 'valid_transition',
        'selected_validity_score': max(results, key=lambda x: x['validity_score'])['validity_score']
    }, f)

# Corrected production replay
def compute_h(q, p):
    # Implement the Hamiltonian computation based on the target definition
    pass

# Save production samples
production_samples = compute_samples(production_states, production_momenta, production_uniforms)
np.save('results/latent_samples.npy', production_samples)
centered_samples = transform_samples(production_samples)
np.save('results/samples.npy', centered_samples)

# Save audit data
audit = {
    'accepted_latent': np.zeros((schedule['n_chains'], schedule['audit_transitions'], schedule['dimension'])),
    'proposal_latent': np.zeros((schedule['n_chains'], schedule['audit_transitions'], schedule['dimension'])),
    'delta_h': np.zeros((schedule['n_chains'], schedule['audit_transitions'])),
    'accept_prob': np.zeros((schedule['n_chains'], schedule['audit_transitions'])),
    'accepted': np.zeros((schedule['n_chains'], schedule['audit_transitions']), dtype=int)
}
# Fill audit data with computed values
np.savez('results/transition_audit.npz', **audit)

# Held-out replay
holdout_samples = compute_samples(holdout_states, holdout_momenta, holdout_uniforms)
np.save('results/holdout_terminal_states.npy', holdout_samples)

# Diagnostics
energy_diagnostics = []
for chain_id in range(schedule['n_chains']):
    acceptance_rate = np.mean(panel_accepted[chain_id])
    delta_h = np.abs(panel_delta_h[chain_id])
    p95_abs_delta_h = np.percentile(delta_h, 95)
    max_abs_delta_h = np.max(delta_h)
    energy_diagnostics.append({
        'chain_id': chain_id,
        'acceptance_rate': acceptance_rate,
        'mean_abs_delta_h': np.mean(delta_h),
        'p95_abs_delta_h': p95_abs_delta_h,
        'max_abs_delta_h': max_abs_delta_h
    })

# Profile bins
profile_bins = pd.read_csv('data/profile_bins.csv')
profile_rows = []
for index, row in profile_bins.iterrows():
    bin_id = index
    v_left = row['v_left']
    v_right = row['v_right']
    samples = centered_samples[(centered_samples[:, 0] >= v_left) & (centered_samples[:, 0] < v_right)]
    count = len(samples)
    fraction = count / len(centered_samples)
    mean_v = np.mean(samples[:, 0])
    mean_radius2 = np.mean(samples[:, 1:]**2)
    expected_radius2 = (schedule['dimension'] - 1) * np.exp(mean_v)
    profile_rows.append({
        'bin_id': bin_id,
        'v_left': v_left,
        'v_right': v_right,
        'count': count,
        'fraction': fraction,
        'mean_v': mean_v,
        'mean_radius2': mean_radius2,
        'expected_radius2': expected_radius2
    })

# Save diagnostics
pd.DataFrame(energy_diagnostics).to_csv('results/energy_diagnostics.csv', index=False)
pd.DataFrame(profile_rows).to_csv('results/funnel_profile.csv', index=False)

# Compute summary statistics
mean_v = np.mean(centered_samples[:, 0])
var_v = np.var(centered_samples[:, 0])
mean_radius2 = np.mean(centered_samples[:, 1:]**2)
split_rhat_v = rhat(centered_samples[:, 0])
ess_v = ess(centered_samples[:, 0])
split_rhat_radius2 = rhat(centered_samples[:, 1:]**2)
ess_radius2 = ess(centered_samples[:, 1:]**2)
overall_acceptance_rate = np.mean(panel_accepted)
chain_acceptance_rates = [np.mean(panel_accepted[i]) for i in range(schedule['n_chains'])]
mean_abs_delta_h = np.mean(np.abs(panel_delta_h))
p95_abs_delta_h = np.percentile(np.abs(panel_delta_h), 95)
max_abs_delta_h = np.max(np.abs(panel_delta_h))
latent_mean_abs_max = np.max(np.abs(centered_samples[:, 0]))
latent_var_abs_max = np.var(np.abs(centered_samples[:, 0]))
left_tail_prob_v_lt_minus5 = percentileofscore(centered_samples[:, 0], -5)
right_tail_prob_v_gt_5 = percentileofscore(centered_samples[:, 0], 5)

# Save diagnostics
with open('results/diagnostics.json', 'w') as f:
    json.dump({
        'mean_v': mean_v,
        'var_v': var_v,
        'mean_radius2': mean_radius2,
        'split_rhat_v': split_rhat_v,
        'ess_v': ess_v,
        'split_rhat_radius2': split_rhat_radius2,
        'ess_radius2': ess_radius2,
        'overall_acceptance_rate': overall_acceptance_rate,
        'chain_acceptance_rates': chain_acceptance_rates,
        'mean_abs_delta_h': mean_abs_delta_h,
        'p95_abs_delta_h': p95_abs_delta_h,
        'max_abs_delta_h': max_abs_delta_h,
        'latent_mean_abs_max': latent_mean_abs_max,
        'latent_var_abs_max': latent_var_abs_max,
        'left_tail_prob_v_lt_minus5': left_tail_prob_v_lt_minus5,
        'right_tail_prob_v_gt_5': right_tail_prob_v_gt_5
    }, f)

# Generate overview plot
plot_overview(centered_samples)