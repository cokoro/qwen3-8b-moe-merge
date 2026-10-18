import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
npz_path = "data/mechanism_panels.npz"
with np.load(npz_path) as npz:
    mechanism_support_grid = npz["mechanism_support_grid"]
    baseline_increment_log_weights = npz["baseline_increment_log_weights"]
    sampling_probe_increment_log_weights = npz["sampling_probe_increment_log_weights"]
    group_probe_increment_log_weights = npz["group_probe_increment_log_weights"]
    epsilon_grid = npz["epsilon_grid"]
    delta_grid = npz["delta_grid"]
    primary_target_delta = npz["primary_target_delta"]
    primary_index = npz["primary_index"]
    diagnostic_recipe = npz["diagnostic_recipe"]

npz_path = "data/accountant_candidate_panels.npz"
with np.load(npz_path) as npz:
    candidate_increment_log_weights = npz["candidate_increment_log_weights"]
    candidate_epsilon = npz["candidate_epsilon"]
    candidate_delta = npz["candidate_delta"]
    inverse_epsilon = npz["inverse_epsilon"]
    sampling_probe_epsilon = npz["sampling_probe_epsilon"]
    group_probe_epsilon = npz["group_probe_epsilon"]

# Compute benchmark curves
def compute_benchmark_curve(log_weights, epsilon_grid, temperature):
    support_grid = epsilon_grid
    weights = np.exp(log_weights)
    tail = np.sum(weights / (1 + np.exp((epsilon_grid - support_grid) / temperature)))
    curve = np.clip(tail ** 1.08, 0, 1)
    return curve

temperature = 0.115 + 0.010 * np.sqrt(np.maximum(support_grid, 0) + 1)
baseline_curve = compute_benchmark_curve(baseline_increment_log_weights, epsilon_grid, temperature)
sampling_probe_curve = compute_benchmark_curve(sampling_probe_increment_log_weights, epsilon_grid, temperature)
group_probe_curve = compute_benchmark_curve(group_probe_increment_log_weights, epsilon_grid, temperature)

# Compute inverse epsilon values
def interpolate_inverse(epsilon_grid, delta_grid, target_delta, clip_bounds):
    idx = np.abs(delta_grid - target_delta).argmin()
    if delta_grid[idx] == target_delta:
        return epsilon_grid[idx]
    if idx == 0:
        return epsilon_grid[0]
    if idx == len(delta_grid) - 1:
        return epsilon_grid[-1]
    # Linear interpolation
    delta1, delta2 = delta_grid[idx - 1], delta_grid[idx]
    epsilon1, epsilon2 = epsilon_grid[idx - 1], epsilon_grid[idx]
    slope = (epsilon2 - epsilon1) / (delta2 - delta1)
    return epsilon1 + slope * (target_delta - delta1)

benchmark_inverse = np.zeros(len(delta_grid))
for i in range(len(delta_grid)):
    benchmark_inverse[i] = interpolate_inverse(epsilon_grid, delta_grid, delta_grid[i], (epsilon_grid[0], epsilon_grid[-1]))

# Compute benchmark probe responses
def compute_benchmark_probe_response(log_weights, epsilon_grid, delta_grid, primary_index):
    target_delta = delta_grid[primary_index]
    inverse_values = interpolate_inverse(epsilon_grid, delta_grid, target_delta, (epsilon_grid[0], epsilon_grid[-1]))
    return inverse_values

benchmark_sampling_response = compute_benchmark_probe_response(sampling_probe_increment_log_weights, epsilon_grid, delta_grid, primary_index)
benchmark_group_response = compute_benchmark_probe_response(group_probe_increment_log_weights, epsilon_grid, delta_grid, primary_index)

# Compute candidate curves
candidate_curves = []
for i in range(len(candidate_increment_log_weights)):
    log_weights = candidate_increment_log_weights[i]
    weights = np.exp(log_weights)
    tail = np.sum(weights / (1 + np.exp((epsilon_grid - support_grid) / temperature)))
    curve = np.clip(tail ** 1.08, 0, 1)
    candidate_curves.append(curve)

# Compute candidate inverse values
candidate_inverse_values = []
for i in range(len(candidate_increment_log_weights)):
    log_weights = candidate_increment_log_weights[i]
    weights = np.exp(log_weights)
    tail = np.sum(weights / (1 + np.exp((epsilon_grid - support_grid) / temperature)))
    curve = np.clip(tail ** 1.08, 0, 1)
    inverse_values = interpolate_inverse(epsilon_grid, delta_grid, primary_target_delta, (epsilon_grid[0], epsilon_grid[-1]))
    candidate_inverse_values.append(inverse_values)

# Compute candidate probe responses
candidate_sampling_responses = []
candidate_group_responses = []
for i in range(len(candidate_increment_log_weights)):
    log_weights = candidate_increment_log_weights[i]
    weights = np.exp(log_weights)
    tail = np.sum(weights / (1 + np.exp((epsilon_grid - support_grid) / temperature)))
    curve = np.clip(tail ** 1.08, 0, 1)
    inverse_values = interpolate_inverse(epsilon_grid, delta_grid, primary_target_delta, (epsilon_grid[0], epsilon_grid[-1]))
    sampling_response = sampling_probe_epsilon[i] - inverse_values
    group_response = group_probe_epsilon[i] - inverse_values
    candidate_sampling_responses.append(sampling_response)
    candidate_group_responses.append(group_response)

# Compute tail mass
def compute_tail_mass(epsilon_grid, curve, tail_start_index):
    tail_mass = np.trapz(curve[tail_start_index:], epsilon_grid[tail_start_index:])
    return tail_mass

tail_start_index = max(0, int(0.72 * len(epsilon_grid)) - 1)
benchmark_tail_mass = compute_tail_mass(epsilon_grid, baseline_curve, tail_start_index)
candidate_tail_masses = []
for curve in candidate_curves:
    candidate_tail_masses.append(compute_tail_mass(epsilon_grid, curve, tail_start_index))

# Compute diagnostics
def compute_curve_panel_error(log_delta_residuals):
    return np.sqrt(np.mean(np.log10(np.abs(log_delta_residuals)) ** 2))

def compute_inverse_panel_error(epsilon_errors):
    return np.mean(epsilon_errors)

def compute_tail_mass_error(candidate_tail_mass, benchmark_tail_mass):
    return abs(candidate_tail_mass - benchmark_tail_mass) / max(abs(benchmark_tail_mass), 1e-14)

log_delta_residuals = []
for i in range(len(candidate_curves)):
    log_weights = candidate_increment_log_weights[i]
    weights = np.exp(log_weights)
    tail = np.sum(weights / (1 + np.exp((epsilon_grid - support_grid) / temperature)))
    curve = np.clip(tail ** 1.08, 0, 1)
    log_delta_residuals.append(np.log10(np.abs(curve - baseline_curve)))

curve_panel_errors = []
for i in range(len(candidate_curves)):
    curve_panel_errors.append(compute_curve_panel_error(log_delta_residuals[i]))

inverse_errors = []
for i in range(len(candidate_inverse_values)):
    inverse_errors.append(np.abs(candidate_inverse_values[i] - benchmark_inverse))

tail_mass_errors = []
for i in range(len(candidate_tail_masses)):
    tail_mass_errors.append(compute_tail_mass_error(candidate_tail_masses[i], benchmark_tail_mass))

# Compute monotonicity violations
monotonicity_violations = []
for i in range(len(candidate_curves)):
    curve = candidate_curves[i]
    violations = np.sum(np.diff(curve) > 0)
    monotonicity_violations.append(violations)

# Compute sampling and group response errors
sampling_response_errors = []
group_response_errors = []
for i in range(len(candidate_sampling_responses)):
    sampling_response_errors.append(np.abs(candidate_sampling_responses[i] - benchmark_sampling_response))
    group_response_errors.append(np.abs(candidate_group_responses[i] - benchmark_group_response))

# Compute reliability scores
curve_panel_scores = []
inverse_panel_scores = []
tail_mass_scores = []
monotonicity_scores = []
sampling_response_scores = []
group_response_scores = []

for i in range(len(candidate_curves)):
    curve_panel_score = 1 - curve_panel_errors[i] / np.max(curve_panel_errors)
    inverse_panel_score = 1 - inverse_errors[i] / np.max(inverse_errors)
    tail_mass_score = 1 - tail_mass_errors[i] / np.max(tail_mass_errors)
    monotonicity_score = 1 - monotonicity_violations[i] / np.max(monotonicity_violations)
    sampling_response_score = 1 - sampling_response_errors[i] / np.max(sampling_response_errors)
    group_response_score = 1 - group_response_errors[i] / np.max(group_response_errors)
    curve_panel_scores.append(curve_panel_score)
    inverse_panel_scores.append(inverse_panel_score)
    tail_mass_scores.append(tail_mass_score)
    monotonicity_scores.append(monotonicity_score)
    sampling_response_scores.append(sampling_response_score)
    group_response_scores.append(group_response_score)

# Compute accountant scores
accountant_scores = []
for i in range(len(candidate_curves)):
    score = (curve_panel_scores[i] + inverse_panel_scores[i] + tail_mass_scores[i] + monotonicity_scores[i] + sampling_response_scores[i] + group_response_scores[i]) / 6
    accountant_scores.append(score)

# Select reliable accountant
reliable_accountant_id = np.argmax(accountant_scores)

# Classify failure modes
failure_modes = {}
for i in range(len(candidate_curves)):
    if i == reliable_accountant_id:
        failure_modes[i] = "reliable_accountant"
        continue
    # First, check for monotonicity validity defect
    if monotonicity_scores[i] < 0.5:
        failure_modes[i] = "monotonicity_smoothing_bug"
        continue
    # Check for inverse-only drift with close curves
    if curve_panel_scores[i] > 0.7 and inverse_panel_scores[i] < 0.5:
        failure_modes[i] = "inverse_interpolation_mismatch"
        continue
    # Check for sampling-response primary defect
    if sampling_response_scores[i] < 0.5 and group_response_scores[i] > 0.7:
        failure_modes[i] = "missing_sampling_amplification"
        continue
    # Check for group-response primary defect
    if group_response_scores[i] < 0.5 and sampling_response_scores[i] > 0.7:
        failure_modes[i] = "group_setting_mismatch"
        continue
    # Check for tail-truncation curve-shape defect
    tail_mass_error = tail_mass_errors[i]
    if tail_mass_error > 0.5 and curve_panel_scores[i] > 0.7 and inverse_panel_scores[i] > 0.7:
        failure_modes[i] = "tail_truncation_loss"
        continue
    # Check for optimistic_curve_alias
    if curve_panel_scores[i] < 0.5 and inverse_panel_scores[i] < 0.5 and sampling_response_scores[i] < 0.5 and group_response_scores[i] < 0.5:
        failure_modes[i] = "optimistic_curve_alias"
        continue
    # Check for conservative_curve_alias
    if curve_panel_scores[i] < 0.5 and inverse_panel_scores[i] < 0.5 and sampling_response_scores[i] < 0.5 and group_response_scores[i] < 0.5:
        failure_modes[i] = "conservative_curve_alias"
        continue

# Write results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Write accountant_audit.csv
accountant_audit = pd.DataFrame({
    "accountant_id": list(range(len(candidate_curves))),
    "curve_panel_score": curve_panel_scores,
    "inverse_panel_score": inverse_panel_scores,
    "tail_mass_score": tail_mass_scores,
    "monotonicity_score": monotonicity_scores,
    "sampling_response_score": sampling_response_scores,
    "group_response_score": group_response_scores,
    "accountant_score": accountant_scores
})
accountant_audit.to_csv(f"{results_dir}/accountant_audit.csv", index=False)

# Write privacy_curve.csv
privacy_curve = pd.DataFrame({
    "epsilon": epsilon_grid,
    "baseline": baseline_curve,
    "sampling_probe": sampling_probe_curve,
    "group_probe": group_probe_curve
})
privacy_curve.to_csv(f"{results_dir}/privacy_curve.csv", index=False)

# Write delta_inverse_audit.csv
delta_inverse_audit = pd.DataFrame({
    "delta": delta_grid,
    "benchmark_inverse": benchmark_inverse,
    "candidate_inverse": [inverse_values[i][primary_index] for i in range(len(candidate_inverse_values))],
    "candidate_inverse_error": [np.abs(candidate_inverse_values[i][primary_index] - benchmark_inverse[primary_index]) for i in range(len(candidate_inverse_values))]
})
delta_inverse_audit.to_csv(f"{results_dir}/delta_inverse_audit.csv", index=False)

# Write mechanism_panel_diagnostics.csv
mechanism_panel_diagnostics = pd.DataFrame({
    "case_id": list(range(len(candidate_curves))),
    "curve_panel_error": curve_panel_errors,
    "inverse_panel_error": inverse_errors,
    "tail_mass_error": tail_mass_errors,
    "monotonicity_violations": monotonicity_violations,
    "sampling_response_error": sampling_response_errors,
    "group_response_error": group_response_errors
})
mechanism_panel_diagnostics.to_csv(f"{results_dir}/mechanism_panel_diagnostics.csv", index=False)

# Write failure_modes.json
failure_modes_json = {
    "selected_accountant_id": reliable_accountant_id,
    "failure_modes": failure_modes
}
with open(f"{results_dir}/failure_modes.json", "w") as f:
    json.dump(failure_modes_json, f)

# Write selected_accountant.json
selected_accountant_json = {
    "selected_accountant_id": reliable_accountant_id,
    "selected_failure_mode": failure_modes[reliable_accountant_id],
    "selected_accountant_score": accountant_scores[reliable_accountant_id]
}
with open(f"{results_dir}/selected_accountant.json", "w") as f:
    json.dump(selected_accountant_json, f)

# Write reliability_report.json
reliability_report_json = {
    "selected_accountant_id": reliable_accountant_id,
    "aggregate_curve_rmse_log10": np.sqrt(np.mean(np.log10(np.abs(np.array(log_delta_residuals) - np.array(baseline_curve))) ** 2)),
    "aggregate_inverse_epsilon_mae": np.mean(inverse_errors),
    "aggregate_panel_agreement_score": np.mean(accountant_scores),
    "accountant_ranking": np.argsort(accountant_scores)[::-1].tolist(),
    "case_warnings": {}
}
with open(f"{results_dir}/reliability_report.json", "w") as f:
    json.dump(reliability_report_json, f)

# Write accountant_overview.png
plt.figure(figsize=(12, 6))
plt.plot(epsilon_grid, baseline_curve, label="Benchmark Curve")
for i in range(len(candidate_curves)):
    plt.plot(epsilon_grid, candidate_curves[i], label=f"Accountant {i}")
plt.legend()
plt.xlabel("Epsilon")
plt.ylabel("Privacy Curve")
plt.title("Privacy Curve Comparison")
plt.savefig(f"{results_dir}/accountant_overview.png")