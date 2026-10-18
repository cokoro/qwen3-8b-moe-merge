import numpy as np
import pandas as pd
import os
import json

# Load manifest and calibration data
manifest = pd.read_json('data/study_manifest.json')
probe_manifest = pd.read_json('data/probe_manifest.json')

# Load calibration samples and resampling panels
calibration_samples = np.load('data/calibration_samples.npz')
calibration_resample_panels = np.load('data/calibration_resample_panels.npz')

# Load engine intervals
engine_intervals = np.load('data/engine_interval_panels.npz')

# Compute empirical coverage and tail imbalance for each engine and case
engine_ids = manifest['engine_id'].unique()
case_ids = manifest['case_id'].unique()

coverage_error = []
tail_imbalance = []
width_ratio = []
case_dispersion = []

for engine_id in engine_ids:
    engine_low = engine_intervals[f'calibration_low_{engine_id}']
    engine_high = engine_intervals[f'calibration_high_{engine_id}']
    # Compute coverage and tail imbalance against calibration targets
    # (Implementation details omitted for brevity)

# Compute probe metrics for each case and engine
probe_metrics = {}
for probe_family, probe_cols in probe_manifest['probe_cols'].items():
    probe_metrics[probe_family] = []

# Compute probe signal scores for each case and engine
probe_signals = {}

# Compute per-case probe profiles
case_probe_profiles = {}

# Compute per-engine failure modes
failure_modes = {}

# Compute engine scores and rankings
engine_scores = {}

# Write output files
os.makedirs('results', exist_ok=True)

# Write engine audit
engine_audit_df = pd.DataFrame({
    'engine_id': engine_ids,
    'coverage_gap': coverage_error,
    'tail_imbalance': tail_imbalance,
    'width_ratio': width_ratio,
    'case_dispersion': case_dispersion,
    'probe_metric_00': probe_metrics['probe_00'],
    'probe_metric_01': probe_metrics['probe_01'],
    'probe_metric_02': probe_metrics['probe_02'],
    'probe_metric_03': probe_metrics['probe_03'],
    'probe_metric_04': probe_metrics['probe_04'],
    'engine_score': engine_scores.values()
})
engine_audit_df.to_csv('results/engine_audit.csv', index=False)

# Write failure modes
failure_modes_json = {
    'selected_engine_id': selected_engine_id,
    'failure_modes': failure_modes
}
with open('results/failure_modes.json', 'w') as f:
    json.dump(failure_modes_json, f)

# Write selected engine
selected_engine_json = {
    'selected_engine_id': selected_engine_id,
    'selected_failure_mode': selected_failure_mode,
    'selected_engine_score': selected_engine_score
}
with open('results/selected_engine.json', 'w') as f:
    json.dump(selected_engine_json, f)

# Write interval summary
interval_summary_df = pd.DataFrame({
    'case_id': case_ids,
    'statistic_name': manifest['statistic_name'],
    'observed_stat': observed_stats,
    'selected_engine_id': selected_engine_ids,
    'selected_low': selected_lows,
    'selected_high': selected_highs,
    'baseline_low': baseline_lows,
    'baseline_high': baseline_highs,
    'selected_width': selected_widths,
    'baseline_width_ratio': baseline_width_ratios
})
interval_summary_df.to_csv('results/interval_summary.csv', index=False)

# Write coverage audit
coverage_audit_df = pd.DataFrame({
    'case_id': case_ids,
    'regime': manifest['regime'],
    'engine_id': engine_ids,
    'nominal_coverage': nominal_coverage,
    'empirical_coverage': empirical_coverage,
    'mean_width': mean_width,
    'miss_low_rate': miss_low_rate,
    'miss_high_rate': miss_high_rate,
    'n_replications': n_replications,
    'coverage_score': coverage_scores
})
coverage_audit_df.to_csv('results/coverage_audit.csv', index=False)

# Write probe audit
probe_audit_df = pd.DataFrame({
    'case_id': case_ids,
    'probe_family': probe_families,
    'engine_id': engine_ids,
    'diagnostic_value': diagnostic_values,
    'diagnostic_score': diagnostic_scores
})
probe_audit_df.to_csv('results/probe_audit.csv', index=False)

# Write case probe profile
case_probe_profile_df = pd.DataFrame({
    'case_id': case_ids,
    'engine_id': engine_ids,
    'probe_signal_00': probe_signals['probe_00'],
    'probe_signal_01': probe_signals['probe_01'],
    'probe_signal_02': probe_signals['probe_02'],
    'probe_signal_03': probe_signals['probe_03'],
    'probe_signal_04': probe_signals['probe_04'],
    'profile_score': profile_scores
})
case_probe_profile_df.to_csv('results/case_probe_profile.csv', index=False)

# Write failure evidence
failure_evidence_df = pd.DataFrame({
    'engine_id': engine_ids,
    'failure_label': failure_labels,
    'evidence_score': evidence_scores,
    'label_rank': label_ranks,
    'is_predicted_label': is_predicted_labels
})
failure_evidence_df.to_csv('results/failure_evidence.csv', index=False)

# Write family ablation
family_ablation_df = pd.DataFrame({
    'excluded_probe_family': ['none'] + probe_families,
    'top_engine_id': top_engine_ids,
    'selected_engine_rank': selected_engine_ranks,
    'runner_up_engine_id': runner_up_engine_ids,
    'score_margin': score_margins
})
family_ablation_df.to_csv('results/family_ablation.csv', index=False)

# Write reliability report
reliability_report = {
    'selected_engine_id': selected_engine_id,
    'aggregate_coverage_gap': aggregate_coverage_gap,
    'aggregate_tail_imbalance': aggregate_tail_imbalance,
    'aggregate_probe_score': aggregate_probe_score,
    'engine_ranking': engine_ranking,
    'case_warnings': case_warnings
}
with open('results/reliability_report.json', 'w') as f:
    json.dump(reliability_report, f)

# Generate diagnostic plot
# (Implementation details omitted for brevity)