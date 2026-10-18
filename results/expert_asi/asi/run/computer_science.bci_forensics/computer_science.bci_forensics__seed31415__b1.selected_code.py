import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load manifest and probe manifest
manifest = pd.read_json('data/study_manifest.json')
probe_manifest = pd.read_json('data/probe_manifest.json')

# Load resampling panels
observed_panels = np.load('data/resample_panels.npz')
calibration_panels = np.load('data/calibration_resample_panels.npz')
engine_panels = np.load('data/engine_interval_panels.npz')

# Extract key arrays
observed_low = engine_panels['observed_low']
observed_high = engine_panels['observed_high']
calibration_low = engine_panels['calibration_low']
calibration_high = engine_panels['calibration_high']
baseline_low = engine_panels['baseline_low']
baseline_high = engine_panels['baseline_high']
probe_02_low = engine_panels['probe_panel_02_low']
probe_02_high = engine_panels['probe_panel_02_high']
probe_03_low = engine_panels['probe_panel_03_low']
probe_03_high = engine_panels['probe_panel_03_high']

# Load calibration samples
calibration_samples = np.load('data/calibration_samples.npz')
calibration_values = calibration_samples['values']

# Compute empirical coverage for each engine and case
def compute_coverage(engine_low, engine_high, calibration_values):
    n = len(calibration_values)
    lower = np.sum(calibration_values < engine_low) / n
    upper = np.sum(calibration_values > engine_high) / n
    return lower, upper

coverage_data = []
for case_id in manifest['case_ids']:
    for engine_id in manifest['engine_ids']:
        cl = calibration_low[case_id, engine_id]
        ch = calibration_high[case_id, engine_id]
        lower, upper = compute_coverage(cl, ch, calibration_values)
        coverage = 1 - (lower + upper)
        coverage_data.append({
            'case_id': case_id,
            'engine_id': engine_id,
            'nominal_coverage': 0.95,
            'empirical_coverage': coverage,
            'mean_width': (ch - cl) / 2,
            'miss_low_rate': lower,
            'miss_high_rate': upper,
            'n_replications': n,
            'coverage_score': coverage
        })

# Compute probe metrics
probe_metrics = {}
for probe_family in probe_manifest['probe_families']:
    probe_values = {}
    for case_id in manifest['case_ids']:
        for engine_id in manifest['engine_ids']:
            if probe_family == 'probe_00':
                # Coverage behavior
                probe_values[(case_id, engine_id)] = 1 - (lower + upper)
            elif probe_family == 'probe_01':
                # Boundary behavior
                probe_values[(case_id, engine_id)] = (ch - cl) / 2
            elif probe_family == 'probe_02':
                # Scale-response gap
                probe_values[(case_id, engine_id)] = (probe_02_high - probe_02_low) / (ch - cl)
            elif probe_family == 'probe_03':
                # Split stability
                probe_values[(case_id, engine_id)] = (probe_03_high - probe_03_low) / (ch - cl)
            elif probe_family == 'probe_04':
                # Baseline distance
                probe_values[(case_id, engine_id)] = (baseline_high - ch) / (ch - cl)
    probe_metrics[probe_family] = probe_values

# Score engines and rank them
engine_scores = {}
for engine_id in manifest['engine_ids']:
    score = 0
    for probe_family in probe_manifest['probe_families']:
        score += probe_metrics[probe_family][(case_id, engine_id)]
    engine_scores[engine_id] = score

engine_rankings = sorted(manifest['engine_ids'], key=lambda x: -engine_scores[x])

# Select the most reliable engine
selected_engine_id = engine_rankings[0]
selected_failure_mode = 'reliable_engine'

# Generate final outputs
pd.DataFrame(coverage_data).to_csv('results/coverage_audit.csv', index=False)
pd.DataFrame(probe_metrics).to_csv('results/probe_audit.csv', index=False)
pd.DataFrame(engine_scores.items(), columns=['engine_id', 'score']).to_csv('results/engine_audit.csv', index=False)
pd.DataFrame({'engine_id': engine_rankings}).to_csv('results/engine_ranking.csv', index=False)

# Plot reliability profile
plt.figure(figsize=(10, 6))
plt.bar(engine_rankings, [engine_scores[id] for id in engine_rankings])
plt.xlabel('Engine ID')
plt.ylabel('Score')
plt.title('Engine Reliability Score')
plt.savefig('results/engine_overview.png')
plt.close()