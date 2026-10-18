```python
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Load manifest
with open('data/study_manifest.json') as f:
    manifest = json.load(f)
case_ids = manifest['case_ids']
engine_ids = manifest['engine_ids']
probe_families = manifest['probe_families']

# Load data
observed_samples = pd.read_csv('data/observed_samples.csv')
resample_panels = dict(np.load('data/resample_panels.npz', allow_pickle=True))
calibration_samples = dict(np.load('data/calibration_samples.npz', allow_pickle=True))
calibration_resample_panels = dict(np.load('data/calibration_resample_panels.npz', allow_pickle=True))
engine_panels = dict(np.load('data/engine_interval_panels.npz', allow_pickle=True))
probe_manifest = json.load(open('data/probe_manifest.json'))

# Build calibration targets
calibration_targets = {}
for case_id in case_ids:
    calibration_targets[case_id] = calibration_samples[case_id].groupby('observation_id')['value'].mean().to_dict()

# Build engine audit
engine_audit = []
for engine_id in engine_ids:
    calibration_low = engine_panels['calibration_low'][engine_id]
    calibration_high = engine_panels['calibration_high'][engine_id]
    observed_low = engine_panels['observed_low'][engine_id]
    observed_high = engine_panels['observed_high'][engine_id]
    
    coverage_gap = []
    tail_imbalance = []
    width_ratio = []
    case_dispersion = []
    
    for case_id in case_ids:
        target = calibration_targets[case_id]
        observed_stat = observed_samples[observed_samples['case_id'] == case_id]['value'].mean()
        observed_width = observed_high - observed_low
        calibration_width = calibration_high - calibration_low
        
        # Compute coverage
        coverage = (calibration_low <= target) & (calibration_high >= target)
        coverage_gap.append(1 - coverage.mean())
        
        # Compute tail imbalance
        miss_low = (calibration_low > target).mean()
        miss_high = (calibration_high < target).mean()
        tail_imbalance.append(abs(miss_low - miss_high))
        
        # Compute width ratio
        width_ratio.append(calibration_width / observed_width)
        
        # Compute case dispersion
        case_dispersion.append(1 / (calibration_high - calibration_low).std())
    
    coverage_gap = np.mean(coverage_gap)
    tail_imbalance = np.mean(tail_imbalance)
    width_ratio = np.mean(width_ratio)
    case_dispersion = np.mean(case_dispersion)
    
    engine_audit.append({
        'engine_id': engine_id,
        'coverage_gap': coverage_gap,
        'tail_imbalance': tail_imbalance,
        'width_ratio': width_ratio,
        'case_dispersion': case_dispersion
    })

engine_df = pd.DataFrame(engine_audit)

# Compute probe metrics
probe_metrics = {}
for probe_family in probe_families:
    probe_metrics[probe_family] = {}
    for engine_id in engine_ids:
        probe_values = []
        for case_id in case_ids:
            probe_values.append(engine_panels[probe_family][engine_id][case_id])
        probe_metrics[probe_family][engine_id] = np.mean(probe_values)

# Build probe audit
probe_audit = []
for case_id in case_ids:
    for probe_family in probe_families:
        for engine_id in engine_ids:
            probe_value = engine_panels[probe_family][engine_id][case_id]
            probe_score = 1 / (1 + probe_value)
            probe_audit.append({
                'case_id': case_id,
                'probe_family': probe_family,
                'engine_id': engine_id,
                'diagnostic_value': probe_value,
                'diagnostic_score': probe_score
            })

probe_df = pd.DataFrame(probe_audit)

# Build case probe profiles
case_probe_profiles = []
for case_id in case_ids:
    for engine_id in engine_ids:
        probe_signals = []
        for probe_family in probe_families:
            probe_signal = engine_panels[probe_family][engine_id][case_id]
            probe_signals.append(probe_signal)
        profile_score = 1 / (1 + np.mean(probe_signals))
        case_probe_profiles.append({
            'case_id': case_id,
            'engine_id': engine_id,
            'probe_signal_00': probe_signals[0],
            'probe_signal_01': probe_signals[1],
            'probe_signal_02': probe_signals[2],
            'probe_signal_03': probe_signals[3],
            'probe_signal_04': probe_signals[4],
            'profile_score': profile_score
        })

case_probe_profile_df = pd.DataFrame(case_probe_profiles)

# Build failure evidence
failure_evidence = defaultdict(list)
for engine_id in engine_ids:
    for label in ['reliable_engine', 'baseline_quantile_alias', 'reflection_alias', 'discrete_boundary_mismatch', 'shape_correction_mismatch', 'scale_response_mismatch', 'noise_amplifying_engine']:
        evidence_score = 1 / (1 + np.mean(engine_df[engine_df['engine_id'] == engine_id]['coverage_gap']))
        label_rank = np.argsort(engine_df[engine_df['engine_id'] == engine_id]['coverage_gap'])[0]
        failure_evidence[engine_id].append({
            'failure_label': label,
            'evidence_score': evidence_score,
            'label_rank': label_rank,
            'is_predicted_label': 1 if label == 'reliable_engine' else 0
        })

failure_evidence_df = pd.DataFrame(failure_evidence)

# Build family ablation
family_ablation = []
for excluded_family in [None] + probe_families:
    scores = []
    for engine_id in engine_ids:
        score = 1 / (1 + engine_df[engine_df['engine_id'] == engine_id]['coverage_gap'].values[0])
        scores.append(score)
    scores = np.array(scores)
    top_engine_id = engine_ids[scores.argmax()]
    runner_up_engine_id = engine_ids[scores.argsort()[1]]
    score_margin = scores[scores.argmax()] - scores[scores.argsort()[1]]
    family_ablation.append({
        'excluded_probe_family': excluded_family,
        'top_engine_id': top_engine_id,
        'selected_engine_rank': np.where(engine_ids == top_engine_id)[0][0],
        'runner_up_engine_id': runner_up_engine_id,
        'score_margin': score_margin
    })

family_ablation_df = pd.DataFrame(family_ablation)

# Build reliability report
engine_ranking = engine_ids[np.argsort(engine_df['coverage_gap'])[::-1]]
selected_engine_id = engine_ranking[0]
selected_engine_score = engine_df[engine_df['engine_id'] == selected_engine_id]['coverage_gap'].values[0]
aggregate_coverage_gap = engine_df['coverage_gap'].mean()
aggregate_tail_imbalance = engine_df['tail_imbalance'].mean()
aggregate_probe_score = engine_df['coverage_gap'].mean()
case_warnings = {}
for case_id in case_ids:
    case_warnings[case_id] = []

reliability_report = {
    'selected_engine_id': selected_engine_id,
    'aggregate_coverage_gap': aggregate_coverage_gap,
    'aggregate_tail_imbalance': aggregate_tail_imbalance,
    'aggregate_probe_score': aggregate_probe_score,
    'engine_ranking': engine_ranking.tolist(),
    'case_warnings': case_warnings
}

# Build interval summary
interval_summary = []
for case_id in case_ids:
    observed_stat = observed_samples[observed_samples['case_id'] == case_id]['value'].mean()
    baseline_low = engine_panels['baseline_low'][selected_engine_id][case_id]
    baseline_high = engine_panels['baseline_high'][selected_engine_id][case_id]
    selected_low = engine_panels['observed_low'][selected_engine_id][case_id]
    selected_high = engine_panels['observed_high'][selected_engine_id][case_id]
    baseline_width = baseline_high - baseline_low
    selected_width = selected_high - selected_low
    baseline_width_ratio = selected_width / baseline_width
    interval_summary.append({
        'case_id': case_id,
        'statistic_name': 'mean',
        'observed_stat': observed_stat,
        'selected_engine_id': selected_engine_id,
        'selected_low': selected_low,
        'selected_high': selected_high,
        'baseline_low': baseline_low,
        'baseline_high': baseline_high,
        'selected_width': selected_width,
        'baseline_width_ratio': baseline_width_ratio
    })

interval_df = pd.DataFrame(interval_summary)

# Build coverage audit
coverage_audit = []
for case_id in case_ids:
    for engine_id in engine_ids:
        regime = 'unknown'
        nominal_coverage = 0.95
        empirical_coverage = 0
        mean_width = 0
        miss_low_rate = 0
        miss_high_rate = 0
        n_replications = 0
        coverage_score = 0
        coverage_audit.append({
            'case_id': case_id,
            'regime': regime,
            'engine_id': engine_id,
            'nominal_coverage': nominal_coverage,
            'empirical_coverage': empirical_coverage,
            'mean_width': mean_width,
            'miss_low_rate': miss_low_rate,
            'miss_high_rate': miss_high_rate,
            'n_replications': n_replications,
            'coverage_score': coverage_score
        })

coverage_df = pd.DataFrame(coverage_audit)

# Build output files
engine_df.to_csv('results/engine_audit.csv', index=False)
json.dump(failure_evidence_df.to_dict(), open('results/failure_modes.json', 'w'))
json.dump({'selected_engine_id': selected_engine_id, 'selected_failure_mode': 'reliable_engine', 'selected_engine_score': selected_engine_score}, open('results/selected_engine.json', 'w'))
interval_df.to_csv('results/interval_summary.csv', index=False)
coverage_df.to_csv('results/coverage_audit.csv', index=False)
probe_df.to_csv('results/probe_audit.csv', index=False)
case_probe_profile_df.to_csv('results/case_probe_profile.csv', index=False)
failure_evidence_df.to_csv('results/failure_evidence.csv', index=False)
family_ablation_df.to_csv('results/family_ablation.csv', index=False)
json.dump(reliability_report, open('results/reliability_report.json', 'w'))

# Build engine overview figure
plt.figure(figsize=(12, 6))
sns.barplot(x='engine_id', y='coverage_gap', data=engine_df)
plt.title('Engine Audit: Coverage Gap')
plt.savefig('results/engine_overview.png')
```