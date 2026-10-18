import numpy as np
import pandas as pd
import json
import os
import sys

# Load config
config_path = sys.argv[1]
with open(config_path) as f:
    config = json.load(f)

# Read input files
case_manifest = pd.read_json(config['case_manifest'])
sample_times = np.load(config['sample_times'])
snapshot_times = np.load(config['snapshot_times'])
observation_spec = json.load(open(config['observation_spec']))
calibration_histograms = pd.read_csv(config['calibration_histograms'])
calibration_queries = pd.read_csv(config['calibration_queries'])
calibration_spec = json.load(open(config['calibration_spec']))
candidate_reactions = json.load(open(config['candidate_reactions']))
mechanism_histograms = pd.read_csv(config['mechanism_histograms'])
mechanism_queries = pd.read_csv(config['mechanism_queries'])
mechanism_spec = json.load(open(config['mechanism_spec']))

# Initialize output files
output_dir = os.path.dirname(config_path)
exact_bundle_summary = pd.DataFrame(columns=['row_id', 'case_index', 'time', 'mean_S', 'mean_ES', 'mean_P', 'var_ES', 'var_P'])
adaptive_bundle_summary = pd.DataFrame(columns=['row_id', 'case_index', 'time', 'mean_S', 'mean_ES', 'mean_P', 'var_ES', 'var_P'])
exact_bundle_turnover = pd.DataFrame(columns=['row_id', 'case_index', 'turnover_index', 'completion_fraction', 'mean_fpt', 'var_fpt'])
adaptive_bundle_turnover = pd.DataFrame(columns=['row_id', 'case_index', 'turnover_index', 'completion_fraction', 'mean_fpt', 'var_fpt'])
stage_memory_probe = pd.DataFrame(columns=['probe_id', 'case_index', 'stage_slow_total', 'lag_time', 'entry_es_a', 'entry_es_b', 'tv_distance_between_entries', 'mean_es_gap_between_entries', 'var_es_gap_between_entries', 'survival_prob_entry_a', 'survival_prob_entry_b'])
stage_residual_time_probe = pd.DataFrame(columns=['probe_id', 'case_index', 'stage_slow_total', 'lag_time', 'entry_es_a', 'entry_es_b', 'mean_remaining_time_entry_a', 'mean_remaining_time_entry_b', 'var_remaining_time_entry_a', 'var_remaining_time_entry_b'])
adaptive_stage_handoff = pd.DataFrame(columns=['handoff_row_id', 'case_index', 'stage_slow_total', 'memory_probe_id', 'p_entry_es_a', 'p_entry_es_b', 'mean_entry_es', 'var_entry_es', 'mean_pre_event_es', 'var_pre_event_es'])
stage_solver_policy = pd.DataFrame(columns=['policy_row_id', 'case_index', 'stage_slow_total', 'memory_probe_id', 'stage_solver_code'])
cme_transient_distribution = pd.DataFrame(columns=['cme_row_id', 'case_index', 'time_index', 'time', 'E', 'S', 'ES', 'P', 'probability'])
stage_exit_kernel = pd.DataFrame(columns=['kernel_row_id', 'case_index', 'stage_slow_total', 'entry_es', 'pre_event_es', 'exit_probability', 'conditional_mean_time', 'conditional_var_time'])
calibrated_rates = pd.DataFrame(columns=['calibration_case_id', 'c1_before', 'c_minus1_before', 'kcat_before', 'c1_after', 'c_minus1_after', 'kcat_after', 'negative_log_likelihood'])
calibration_predictions = pd.DataFrame(columns=['calibration_query_id', 'calibration_case_id', 'time_index', 'time', 'E', 'S', 'ES', 'P', 'probability'])
selected_mechanism = pd.DataFrame(columns=['candidate_id', 'knot_id', 'knot_time', 'rate'])
mechanism_public_fit = pd.DataFrame(columns=['mechanism_hist_row_id', 'probability'])
mechanism_predictions = pd.DataFrame(columns=['mechanism_query_id', 'probability'])
conservation_report = {}
runtime_report = {}

# Process each case
for case_index, case in case_manifest.iterrows():
    case_id = case['case_id']
    case_data = json.load(open(os.path.join(config['input_dir'], f"{case_id}.json")))
    # Process exact simulations
    # Process adaptive simulations
    # Process calibration
    # Process mechanism discovery
    # Process conservation report
    # Process runtime report

# Save outputs
exact_bundle_summary.to_csv(os.path.join(output_dir, 'exact_bundle_summary.csv'), index=False)
adaptive_bundle_summary.to_csv(os.path.join(output_dir, 'adaptive_bundle_summary.csv'), index=False)
exact_bundle_turnover.to_csv(os.path.join(output_dir, 'exact_bundle_turnover.csv'), index=False)
adaptive_bundle_turnover.to_csv(os.path.join(output_dir, 'adaptive_bundle_turnover.csv'), index=False)
stage_memory_probe.to_csv(os.path.join(output_dir, 'stage_memory_probe.csv'), index=False)
stage_residual_time_probe.to_csv(os.path.join(output_dir, 'stage_residual_time_probe.csv'), index=False)
adaptive_stage_handoff.to_csv(os.path.join(output_dir, 'adaptive_stage_handoff.csv'), index=False)
stage_solver_policy.to_csv(os.path.join(output_dir, 'stage_solver_policy.csv'), index=False)
cme_transient_distribution.to_csv(os.path.join(output_dir, 'cme_transient_distribution.csv'), index=False)
stage_exit_kernel.to_csv(os.path.join(output_dir, 'stage_exit_kernel.csv'), index=False)
calibrated_rates.to_csv(os.path.join(output_dir, 'calibrated_rates.csv'), index=False)
calibration_predictions.to_csv(os.path.join(output_dir, 'calibration_predictions.csv'), index=False)
selected_mechanism.to_csv(os.path.join(output_dir, 'selected_mechanism.csv'), index=False)
mechanism_public_fit.to_csv(os.path.join(output_dir, 'mechanism_public_fit.csv'), index=False)
mechanism_predictions.to_csv(os.path.join(output_dir, 'mechanism_predictions.csv'), index=False)
json.dump(conservation_report, open(os.path.join(output_dir, 'conservation_report.json'), 'w'))
json.dump(runtime_report, open(os.path.join(output_dir, 'runtime_report.json'), 'w'))
np.savez(os.path.join(output_dir, 'bundle_snapshot_samples.npz'),
         exact_s_samples=exact_s_samples,
         exact_es_samples=exact_es_samples,
         exact_p_samples=exact_p_samples,
         adaptive_s_samples=adaptive_s_samples,
         adaptive_es_samples=adaptive_es_samples,
         adaptive_p_samples=adaptive_p_samples)