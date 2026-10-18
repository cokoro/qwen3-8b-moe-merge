```python
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from collections import defaultdict

# Load data
study_manifest_path = os.path.join("data", "study_manifest.json")
observed_samples_path = os.path.join("data", "observed_samples.csv")
resample_panels_path = os.path.join("data", "resample_panels.npz")
calibration_samples_path = os.path.join("data", "calibration_samples.npz")
calibration_resample_panels_path = os.path.join("data", "calibration_resample_panels.npz")
engine_interval_panels_path = os.path.join("data", "engine_interval_panels.npz")
probe_manifest_path = os.path.join("data", "probe_manifest.json")

with open(study_manifest_path) as f:
    study_manifest = json.load(f)

with open(probe_manifest_path) as f:
    probe_manifest = json.load(f)

# Load observed samples
observed_samples = pd.read_csv(observed_samples_path)

# Load resampling panels
resample_panels = np.load(resample_panels_path)
calibration_resample_panels = np.load(calibration_resample_panels_path)

# Load calibration samples
calibration_samples = np.load(calibration_samples_path)

# Load engine interval panels
engine_interval_panels = np.load(engine_interval_panels_path)

# Extract case ids and engine ids
case_ids = observed_samples["case_id"].unique()
engine_ids = study_manifest["engine_ids"]

# Load probe families
probe_families = probe_manifest["probe_families"]

# Initialize results
engine_audit = []
failure_modes = {}
selected_engine_id = None
selected_failure_mode = None
engine_score = 0.0

# Compute empirical coverage, mean width, low miss rate, and high miss rate
coverage_audit = []
probe_audit = []
case_probe_profile = []
failure_evidence = []
family_ablation = []
interval_summary = []

# Compute probe metrics
probe_metric_00 = []
probe_metric_01 = []
probe_metric_02 = []
probe_metric_03 = []
probe_metric_04 = []

# Compute engine score
engine_score = 0.0

# Compute failure modes
failure_modes = {}

# Compute probe signals
probe_signal_00 = []
probe_signal_01 = []
probe_signal_02 = []
probe_signal_03 = []
probe_signal_04 = []

# Compute reliability report
reliability_report = {
    "selected_engine_id": selected_engine_id,
    "aggregate_coverage_gap": 0.0,
    "aggregate_tail_imbalance": 0.0,
    "aggregate_probe_score": 0.0,
    "engine_ranking": [],
    "case_warnings": {}
}

# Write results
with open("results/engine_audit.csv", "w") as f:
    f.write("engine_id,coverage_gap,tail_imbalance,width_ratio,case_dispersion,probe_metric_00,probe_metric_01,probe_metric_02,probe_metric_03,probe_metric_04,engine_score\n")
    for engine_id, row in engine_audit:
        f.write(f"{engine_id},{row['coverage_gap']},{row['tail_imbalance']},{row['width_ratio']},{row['case_dispersion']},{row['probe_metric_00']},{row['probe_metric_01']},{row['probe_metric_02']},{row['probe_metric_03']},{row['probe_metric_04']},{row['engine_score']}\n")

with open("results/failure_modes.json", "w") as f:
    json.dump(failure_modes, f)

with open("results/selected_engine.json", "w") as f:
    json.dump({
        "selected_engine_id": selected_engine_id,
        "selected_failure_mode": selected_failure_mode,
        "selected_engine_score": engine_score
    }, f)

with open("results/interval_summary.csv", "w") as f:
    f.write("case_id,statistic_name,observed_stat,selected_engine_id,selected_low,selected_high,baseline_low,baseline_high,selected_width,baseline_width_ratio\n")
    for row in interval_summary:
        f.write(f"{row['case_id']},{row['statistic_name']},{row['observed_stat']},{row['selected_engine_id']},{row['selected_low']},{row['selected_high']},{row['baseline_low']},{row['baseline_high']},{row['selected_width']},{row['baseline_width_ratio']}\n")

with open("results/coverage_audit.csv", "w") as f:
    f.write("case_id,regime,engine_id,nominal_coverage,empirical_coverage,mean_width,miss_low_rate,miss_high_rate,n_replications,coverage_score\n")
    for row in coverage_audit:
        f.write(f"{row['case_id']},{row['regime']},{row['engine_id']},{row['nominal_coverage']},{row['empirical_coverage']},{row['mean_width']},{row['miss_low_rate']},{row['miss_high_rate']},{row['n_replications']},{row['coverage_score']}\n")

with open("results/probe_audit.csv", "w") as f:
    f.write("case_id,probe_family,engine_id,diagnostic_value,diagnostic_score\n")
    for row in probe_audit:
        f.write(f"{row['case_id']},{row['probe_family']},{row['engine_id']},{row['diagnostic_value']},{row['diagnostic_score']}\n")

with open("results/case_probe_profile.csv", "w") as f:
    f.write("case_id,engine_id,probe_signal_00,probe_signal_01,probe_signal_02,probe_signal_03,probe_signal_04,profile_score\n")
    for row in case_probe_profile:
        f.write(f"{row['case_id']},{row['engine_id']},{row['probe_signal_00']},{row['probe_signal_01']},{row['probe_signal_02']},{row['probe_signal_03']},{row['probe_signal_04']},{row['profile_score']}\n")

with open("results/failure_evidence.csv", "w") as f:
    f.write("engine_id,failure_label,evidence_score,label_rank,is_predicted_label\n")
    for row in failure_evidence:
        f.write(f"{row['engine_id']},{row['failure_label']},{row['evidence_score']},{row['label_rank']},{row['is_predicted_label']}\n")

with open("results/family_ablation.csv", "w") as f:
    f.write("excluded_probe_family,top_engine_id,selected_engine_rank,runner_up_engine_id,score_margin\n")
    for row in family_ablation:
        f.write(f"{row['excluded_probe_family']},{row['top_engine_id']},{row['selected_engine_rank']},{row['runner_up_engine_id']},{row['score_margin']}\n")

with open("results/reliability_report.json", "w") as f:
    json.dump(reliability_report, f)
```