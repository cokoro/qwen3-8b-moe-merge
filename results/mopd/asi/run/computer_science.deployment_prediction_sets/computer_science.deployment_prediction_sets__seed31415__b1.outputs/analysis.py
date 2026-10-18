import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import rankdata

# Load metadata and risk specification
metadata_path = 'data/metadata.json'
risk_spec_path = 'data/risk_spec.json'
with open(metadata_path) as f:
    metadata = json.load(f)
with open(risk_spec_path) as f:
    risk_spec = json.load(f)

# Load calibration and deployment data
calibration_data_path = 'data/calibration.csv'
deployment_data_path = 'data/deployment.csv'
calibration_df = pd.read_csv(calibration_data_path)
deployment_df = pd.read_csv(deployment_data_path)

# Step 3: Discover proxy strata
Z = np.vstack((calibration_df.drop(columns=['label']).values, deployment_df.drop(columns=['label']).values))
Z_standardized = (Z - Z.mean(axis=0)) / Z.std(axis=0)
pca = PCA(n_components=1)
pca.fit(Z_standardized)
projection = pca.transform(Z_standardized)
initial_centers = np.quantile(projection, [0.08, 0.36, 0.64, 0.92], axis=0)
kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=10)
kmeans.fit(Z_standardized)
proxy_strata = kmeans.labels_
proxy_strata = np.sort(proxy_strata)[::-1]
proxy_strata = np.unique(proxy_strata)

# Step 4: Find optimal cutoffs for each proxy stratum
def find_optimal_cutoffs(df, proxy_strata, risk_spec):
    cutoffs = {}
    for s in proxy_strata:
        stratum_df = df[df['proxy_stratum'] == s]
        stratum_df.sort_values('score', inplace=True)
        stratum_df['cumulative_weighted_miss'] = 0
        stratum_df['cumulative_weighted_miss'] = stratum_df['score'].cumsum() * stratum_df['label_cost']
        stratum_df['cumulative_weighted_miss'] = stratum_df['cumulative_weighted_miss'] / stratum_df['score'].sum()
        cutoffs[s] = stratum_df[stratum_df['cumulative_weighted_miss'] <= risk_spec['target_risk']]['score'].iloc[-1]
    return cutoffs

cutoffs = find_optimal_cutoffs(deployment_df, proxy_strata, risk_spec)

# Step 5: Apply cutoffs to deployment records
def apply_cutoffs(df, cutoffs, proxy_strata):
    df['prediction_set'] = df['label']
    for s in proxy_strata:
        stratum_df = df[df['proxy_stratum'] == s]
        stratum_df['prediction_set'] = stratum_df['prediction_set'][stratum_df['score'] >= cutoffs[s]]
    return df

deployment_df = apply_cutoffs(deployment_df, cutoffs, proxy_strata)

# Step 6: Evaluate risk and set size
def evaluate_risk(df, risk_spec):
    risk = 0
    for label in df['label'].unique():
        risk += df[df['label'] == label]['label_cost'].sum() * df[df['label'] == label]['score'].sum()
    risk /= df['score'].sum()
    return risk

risk = evaluate_risk(deployment_df, risk_spec)

# Step 7: Write outputs
os.makedirs('results', exist_ok=True)

# Write prediction sets
deployment_df.to_csv('results/prediction_sets.csv', index=False)

# Write policy parameters
policy_params = []
for s in proxy_strata:
    stratum_df = deployment_df[deployment_df['proxy_stratum'] == s]
    for label in stratum_df['label'].unique():
        policy_params.append({
            'proxy_stratum': s,
            'threshold': cutoffs[s],
            'n_calibration': len(stratum_df),
            'empirical_weighted_miss_risk': risk,
            'target_risk': risk_spec['target_risk']
        })
pd.DataFrame(policy_params).to_csv('results/policy_parameters.csv', index=False)

# Write audit summary
audit_summary = {
    'scope': 'overall',
    'empirical_weighted_miss_risk': risk,
    'target_risk': risk_spec['target_risk'],
    'set_size': len(deployment_df)
}
pd.DataFrame([audit_summary]).to_csv('results/audit_summary.csv', index=False)

# Write set size summary
set_size_summary = {
    'record_id': deployment_df['record_id'],
    'set_size': deployment_df['prediction_set'].str.split().str.len()
}
pd.DataFrame(set_size_summary).to_csv('results/set_size_summary.csv', index=False)

# Write partition diagnostics
partition_diagnostics = {
    'proxy_stratum': proxy_strata,
    'risk_margin': risk - risk_spec['target_risk'],
    'set_size': len(deployment_df)
}
pd.DataFrame(partition_diagnostics).to_csv('results/partition_diagnostics.csv', index=False)

# Write distribution diagnostics
distribution_diagnostics = {
    'proxy_stratum': proxy_strata,
    'calibration_fraction': len(calibration_df[calibration_df['proxy_stratum'] == s]) / len(calibration_df),
    'deployment_fraction': len(deployment_df[deployment_df['proxy_stratum'] == s]) / len(deployment_df),
    'density_ratio': len(deployment_df[deployment_df['proxy_stratum'] == s]) / len(calibration_df[calibration_df['proxy_stratum'] == s])
}
pd.DataFrame(distribution_diagnostics).to_csv('results/distribution_diagnostics.csv', index=False)

# Write calibration diagnostics
calibration_diagnostics = {
    'proxy_stratum': proxy_strata,
    'score': calibration_df['score'],
    'label': calibration_df['label']
}
pd.DataFrame(calibration_diagnostics).to_csv('results/calibration_diagnostics.csv', index=False)

# Write constraint audit
constraint_audit = {
    'scope': 'overall',
    'empirical_weighted_miss_risk': risk,
    'target_risk': risk_spec['target_risk'],
    'set_size': len(deployment_df)
}
pd.DataFrame([constraint_audit]).to_csv('results/constraint_audit.csv', index=False)

# Write set size tradeoff plot
import matplotlib.pyplot as plt
plt.plot(deployment_df['set_size'])
plt.savefig('results/set_size_tradeoff.png')

# Write partition audit plot
plt.bar(proxy_strata, [risk - risk_spec['target_risk'] for _ in proxy_strata])
plt.savefig('results/partition_audit.png')

# Write score diagnostics plot
plt.hist(deployment_df['score'])
plt.savefig('results/score_diagnostics.png')