```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
npz = np.load('data/mechanism_panels.npz')
epsilon_grid = npz['epsilon_grid']
target_deltas = npz['target_deltas']
mechanism_support_grid = npz['mechanism_support_grid']
baseline_increment_log_weights = npz['baseline_increment_log_weights']
sampling_probe_increment_log_weights = npz['sampling_probe_increment_log_weights']
group_probe_increment_log_weights = npz['group_probe_increment_log_weights']
diagnostic_recipe = npz['diagnostic_recipe']

# Load candidate panels
candidate_npz = np.load('data/accountant_candidate_panels.npz')
epsilon_curves = candidate_npz['epsilon_curves']
delta_inverses = candidate_npz['delta_inverses']
sampling_probe_epsilons = candidate_npz['sampling_probe_epsilons']
group_probe_epsilons = candidate_npz['group_probe_epsilons']
accountant_ids = candidate_npz['accountant_ids']

# Benchmark curve derivation
def derive_benchmark_curve(increment_log_weights, epsilon_grid, temperature=0.115, alpha=0.010, power=1.08):
    support = mechanism_support_grid
    weight = np.exp(increment_log_weights)
    tail = np.zeros_like(epsilon_grid)
    for i, eps in enumerate(epsilon_grid):
        temp = 0.115 + 0.010 * np.sqrt(max(support[i], 0) + 1)
        tail[i] = np.sum(weight / (1 + np.exp((eps - support) / temp)))
    tail = np.power(tail, power)
    tail = np.clip(tail, 1e-15, 1.0)
    return tail

baseline_tail = derive_benchmark_curve(baseline_increment_log_weights, epsilon_grid)
sampling_tail = derive_benchmark_curve(sampling_probe_increment_log_weights, epsilon_grid)
group_tail = derive_benchmark_curve(group_probe_increment_log_weights, epsilon_grid)

# Benchmark inverse derivation
def interpolate_inverse(epsilon_curve, epsilon_grid, delta_target, clip=True):
    if delta_target not in epsilon_curve:
        if clip:
            return np.nan
        else:
            return np.inf
    return epsilon_curve[delta_target]

# Benchmark probe response derivation
primary_index = np.where(target_deltas == np.max(target_deltas))[0][0]
primary_target_delta = target_deltas[primary_index]

# Candidate probe response derivation
def derive_candidate_probe_response(candidate_id, epsilon_grid, sampling_probe_epsilons, delta_inverses):
    sampling_response = sampling_probe_epsilons[candidate_id] - delta_inverses[candidate_id][primary_index]
    group_response = group_probe_epsilons[candidate_id] - delta_inverses[candidate_id][primary_index]
    return sampling_response, group_response

# Tail mass derivation
def derive_tail_mass(epsilon_curve, epsilon_grid):
    tail_start_index = max(0, int(np.floor(0.72 * len(epsilon_grid))) - 1)
    tail_mass = np.trapz(epsilon_curve[tail_start_index:], epsilon_grid[tail_start_index:])
    return tail_mass

# Curve panel error derivation
def derive_curve_panel_error(candidate_id, epsilon_curve, benchmark_curve):
    log10_residuals = np.log10(epsilon_curve / benchmark_curve)
    return np.sqrt(np.mean(log10_residuals ** 2))

# Inverse panel error derivation
def derive_inverse_panel_error(candidate_id, delta_inverse, benchmark_inverse):
    return np.abs(delta_inverse - benchmark_inverse)

# Tail mass error derivation
def derive_tail_mass_error(candidate_tail, benchmark_tail):
    return np.abs(candidate_tail - benchmark_tail) / np.maximum(np.abs(benchmark_tail), 1e-14)

# Monotonicity check
def is_monotonic(candidate_id, epsilon_curve):
    return np.all(np.diff(epsilon_curve[candidate_id]) <= 0)

# Benchmark delta inverse derivation
benchmark_delta_inverses = np.zeros_like(delta_inverses)
for i, delta in enumerate(target_deltas):
    for j, candidate_id in enumerate(accountant_ids):
        if delta in epsilon_curves[j]:
            benchmark_delta_inverses[j, i] = epsilon_curves[j][delta]
        else:
            benchmark_delta_inverses[j, i] = np.nan

# Candidate delta inverse derivation
candidate_delta_inverses = np.zeros((len(accountant_ids), len(target_deltas)))
for i, delta in enumerate(target_deltas):
    for j, candidate_id in enumerate(accountant_ids):
        if delta in epsilon_curves[j]:
            candidate_delta_inverses[j, i] = epsilon_curves[j][delta]
        else:
            candidate_delta_inverses[j, i] = np.nan

# Benchmark sampling response derivation
benchmark_sampling_responses = np.zeros_like(sampling_probe_epsilons)
for i, candidate_id in enumerate(accountant_ids):
    for j, delta in enumerate(target_deltas):
        if delta in epsilon_curves[i]:
            benchmark_sampling_responses[i, j] = sampling_probe
```